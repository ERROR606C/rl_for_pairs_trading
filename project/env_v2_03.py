import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PairTradingEnv(gym.Env):
    """
    Pairs trading environment — simplified observation, 3 actions.

    Observation (12 floats):
        z_score          – current spread z-score
        position         – -1 short / 0 flat / +1 long
        pct_1 … pct_10   – percentage moves of z-score over past 10 steps
                           pct_k = (z[t-k+1] - z[t-k]) / (|z[t-k]| + 0.1) * 100
                           ordered oldest → most recent

    Actions:
        0 – go flat   (close any open position)
        1 – long  the spread  (long A, short β·B)
        2 – short the spread  (short A, long β·B)

    Financials identical to env_v2_01:
        - Beta-neutral sizing, TC on notional, cash earns rf every step
        - Reward = step change in total wealth
    """

    _ACT2POS  = {0: 0, 1: 1, 2: -1}
    _LOOKBACK = 10   # number of z-score pct moves in obs

    def __init__(
        self,
        prices: np.ndarray,
        beta: float,
        intercept: float,
        fixed_std: float,
        initial_wealth: float = 10_000.0,
        tc: float = 0.0001,
        rf_annual: float = 0.05,
        past_window: int = 60,
        train_end: int = None,
        episode_len: int = None,
    ):
        super().__init__()
        N = prices.shape[1]
        self.prices       = prices.astype(np.float64)
        self.beta         = float(beta)
        self.intercept    = float(intercept)
        self.initial_wealth = float(initial_wealth)
        self.tc           = float(tc)
        self.rf_daily     = (1.0 + rf_annual) ** (1.0 / 252) - 1.0
        self.past_window  = int(past_window)
        self._N           = N

        self._min_t = past_window + self._LOOKBACK   # need history for pct moves

        self.train_end   = int(train_end)   if train_end   is not None else N
        self.episode_len = int(episode_len) if episode_len is not None else (N - self._min_t)

        # Precompute full rolling z-score series
        spread  = np.log(prices[0]) - self.beta * np.log(prices[1]) - float(intercept)
        s       = pd.Series(spread)
        rm      = s.rolling(past_window).mean()
        rs      = s.rolling(past_window).std().clip(lower=1e-8)
        self._z = ((s - rm) / rs).to_numpy(dtype=np.float64)

        obs_dim = 2 + self._LOOKBACK   # z_score, position, 10 pct moves
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

        self._t      = self._min_t
        self._end    = self._min_t + self.episode_len
        self._cash   = float(initial_wealth)
        self._shares = np.zeros(2, dtype=np.float64)
        self._pos    = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _safe_t(self) -> int:
        return min(self._t, self._N - 1)

    def _pct_moves(self, t: int) -> np.ndarray:
        """10 percentage moves of z-score: oldest → most recent."""
        zw    = self._z[t - self._LOOKBACK: t + 1]          # 11 values
        denom = np.abs(zw[:-1]) + 0.1                        # floor avoids ÷0
        pct   = (zw[1:] - zw[:-1]) / denom * 100.0
        return np.clip(pct, -200.0, 200.0).astype(np.float32)

    def _obs(self) -> np.ndarray:
        t = self._safe_t()
        return np.concatenate([
            [self._z[t], float(self._pos)],
            self._pct_moves(t),
        ]).astype(np.float32)

    def _wealth(self) -> float:
        t = self._safe_t()
        return float(self._cash + self._shares @ self.prices[:, t])

    # ── position management ───────────────────────────────────────────────────

    def _enter(self, direction: int):
        t      = self._t
        p0, p1 = self.prices[0, t], self.prices[1, t]
        C, b   = self._wealth(), self.beta
        d0     = C / (1.0 + b)
        d1     = b * C / (1.0 + b)

        self._shares[0] =  direction * (d0 / p0)
        self._shares[1] = -direction * (d1 / p1)
        self._cash      += direction * (d1 - d0)
        self._cash      -= self.tc * (d0 + d1)
        self._pos        = direction

    def _exit(self):
        t        = self._t
        p0, p1   = self.prices[0, t], self.prices[1, t]
        notional = abs(self._shares[0]) * p0 + abs(self._shares[1]) * p1
        self._cash    = self._wealth() - self.tc * notional
        self._shares[:] = 0.0
        self._pos       = 0

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None):
        super().reset(seed=seed)
        min_t = self._min_t
        max_t = max(min_t, self.train_end - self.episode_len)
        start = int(self.np_random.integers(min_t, max_t + 1))

        self._t      = start
        self._end    = start + self.episode_len
        self._cash   = float(self.initial_wealth)
        self._shares = np.zeros(2, dtype=np.float64)
        self._pos    = 0
        return self._obs(), {}

    def step(self, action: int):
        old_wealth  = self._wealth()
        desired_pos = self._ACT2POS[action]

        if desired_pos != self._pos:
            if self._pos != 0:
                self._exit()
            if desired_pos != 0:
                self._enter(desired_pos)

        self._cash *= (1.0 + self.rf_daily)
        self._t    += 1

        reward = float(self._wealth() - old_wealth)
        done   = self._t >= self._end
        return self._obs(), reward, done, False, {}
