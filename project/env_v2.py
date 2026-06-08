import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PairTradingEnv(gym.Env):
    """
    Pairs trading environment.

    Observation (8 floats, all signed/normalised):
        z          – current spread z-score
        entry_z    – z-score at entry (0 if flat)
        position   – 0 flat / 1 in position
        macd_n     – MACD of spread / rolling_std
        signal_n   – MACD signal line / rolling_std
        hist_n     – MACD histogram / rolling_std
        mean_z     – rolling_mean / rolling_std
        vol_ratio  – rolling_std / fixed_std

    Actions: 0 = hold, 1 = toggle (enter if flat, exit if in position)

    Financials:
        - Beta-neutral sizing: all wealth deployed on entry
        - Transaction cost: tc * notional charged on each trade
        - Cash earns rf_daily every step (whether in position or not)
        - Reward = step change in total wealth
    """

    def __init__(
        self,
        prices: np.ndarray,       # shape (2, N)
        beta: float,
        intercept: float,
        fixed_std: float,
        initial_wealth: float = 10_000.0,
        tc: float = 0.0001,       # 0.01%
        rf_annual: float = 0.05,
        past_window: int = 60,
        train_end: int = None,    # upper bound for episode-start sampling (exclusive)
        episode_len: int = None,  # steps per episode
    ):
        super().__init__()
        N = prices.shape[1]
        self.prices       = prices.astype(np.float64)
        self.beta         = float(beta)
        self.intercept    = float(intercept)
        self.fixed_std    = max(float(fixed_std), 1e-8)
        self.initial_wealth = float(initial_wealth)
        self.tc           = float(tc)
        self.rf_daily     = (1.0 + rf_annual) ** (1.0 / 252) - 1.0
        self.past_window  = int(past_window)
        self.train_end    = int(train_end)   if train_end   is not None else N
        self.episode_len  = int(episode_len) if episode_len is not None else (N - past_window)

        self._spread = np.log(prices[0]) - self.beta * np.log(prices[1]) - self.intercept
        self._N      = N

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

        # initialise state so the env is usable before first reset
        self._t          = past_window
        self._end        = past_window + self.episode_len
        self._cash       = float(initial_wealth)
        self._shares     = np.zeros(2, dtype=np.float64)
        self._in_pos     = 0
        self._entry_z    = 0.0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _safe_t(self) -> int:
        """Clamp t to valid price-array index (guards terminal-step OOB)."""
        return min(self._t, self._N - 1)

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        k = 2.0 / (span + 1)
        out = np.empty(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
        return out

    def _roll_stats(self, t: int):
        w = self._spread[t - self.past_window: t]
        return w.mean(), max(w.std(), 1e-8)

    def _obs(self) -> np.ndarray:
        t = self._safe_t()
        roll_mean, roll_std = self._roll_stats(t)
        z = (self._spread[t] - roll_mean) / roll_std

        # MACD on signed spread, normalised by roll_std
        sw       = self._spread[t - self.past_window: t + 1]
        ema12    = self._ema(sw, 12)
        ema26    = self._ema(sw, 26)
        macd_arr = ema12 - ema26
        macd     = macd_arr[-1]
        signal   = self._ema(macd_arr, 9)[-1]

        return np.array([
            z,
            self._entry_z,
            float(self._in_pos),
            macd / roll_std,
            signal / roll_std,
            (macd - signal) / roll_std,
            roll_mean / roll_std,
            roll_std / self.fixed_std,
        ], dtype=np.float32)

    def _wealth(self) -> float:
        t = self._safe_t()
        return float(self._cash + self._shares @ self.prices[:, t])

    # ── position management ───────────────────────────────────────────────────

    def _enter(self):
        t       = self._t
        p0, p1  = self.prices[0, t], self.prices[1, t]
        C, b    = self._wealth(), self.beta
        d0      = C / (1.0 + b)
        d1      = b * C / (1.0 + b)

        if self._spread[t] > 0:          # s0 rich: short s0, long s1
            self._shares[0] = -(d0 / p0)
            self._shares[1] =  (d1 / p1)
            self._cash += d0 - d1
        else:                             # s0 cheap: long s0, short s1
            self._shares[0] =  (d0 / p0)
            self._shares[1] = -(d1 / p1)
            self._cash += d1 - d0

        self._cash -= self.tc * (d0 + d1)
        rm, rs = self._roll_stats(t)
        self._entry_z = float((self._spread[t] - rm) / rs)
        self._in_pos  = 1

    def _exit(self):
        t       = self._t
        p0, p1  = self.prices[0, t], self.prices[1, t]
        notional = abs(self._shares[0]) * p0 + abs(self._shares[1]) * p1
        self._cash    = self._wealth() - self.tc * notional
        self._shares[:] = 0.0
        self._entry_z   = 0.0
        self._in_pos    = 0

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None):
        super().reset(seed=seed)
        min_t = self.past_window
        max_t = max(min_t, self.train_end - self.episode_len)
        start = int(self.np_random.integers(min_t, max_t + 1))

        self._t       = start
        self._end     = start + self.episode_len
        self._cash    = float(self.initial_wealth)
        self._shares  = np.zeros(2, dtype=np.float64)
        self._in_pos  = 0
        self._entry_z = 0.0
        return self._obs(), {}

    def step(self, action: int):
        old_wealth = self._wealth()

        if action == 1:
            if self._in_pos == 0:
                self._enter()
            else:
                self._exit()

        self._cash *= (1.0 + self.rf_daily)
        self._t    += 1

        reward = float(self._wealth() - old_wealth)
        done   = self._t >= self._end
        return self._obs(), reward, done, False, {}
