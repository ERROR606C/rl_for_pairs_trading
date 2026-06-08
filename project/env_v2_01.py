import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PairTradingEnv(gym.Env):
    """
    Pairs trading environment — 3-action version.

    Observation (8 floats, all signed/normalised):
        z          – current spread z-score
        entry_z    – z-score at entry (0 if flat)
        position   – -1 short spread / 0 flat / +1 long spread
        macd_n     – MACD of spread / rolling_std
        signal_n   – MACD signal line / rolling_std
        hist_n     – MACD histogram / rolling_std
        mean_z     – rolling_mean / rolling_std
        vol_ratio  – rolling_std / fixed_std

    Actions:
        0 – go flat   (close any open position)
        1 – long  the spread  (long A, short β·B)
        2 – short the spread  (short A, long β·B)

    Transitions:
        same action as current position → do nothing
        0/1/2 → different → exit current (if any), enter new (if not flat)
        Reversals (1→2 or 2→1) pay TC twice (exit + re-enter).

    Financials:
        - Beta-neutral sizing: all wealth deployed on entry
        - Transaction cost: tc * notional on each trade leg
        - Cash earns rf_daily every step
        - Reward = step change in total wealth
    """

    # maps action index to position value used internally
    _ACT2POS = {0: 0, 1: 1, 2: -1}

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
        self.action_space      = spaces.Discrete(3)

        self._t       = past_window
        self._end     = past_window + self.episode_len
        self._cash    = float(initial_wealth)
        self._shares  = np.zeros(2, dtype=np.float64)
        self._pos     = 0      # -1 short / 0 flat / +1 long
        self._entry_z = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _safe_t(self) -> int:
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

        sw       = self._spread[t - self.past_window: t + 1]
        ema12    = self._ema(sw, 12)
        ema26    = self._ema(sw, 26)
        macd_arr = ema12 - ema26
        macd     = macd_arr[-1]
        signal   = self._ema(macd_arr, 9)[-1]

        return np.array([
            z,
            self._entry_z,
            float(self._pos),          # -1 / 0 / +1
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

    def _enter(self, direction: int):
        """direction: +1 = long spread (long A, short B), -1 = short spread."""
        t      = self._t
        p0, p1 = self.prices[0, t], self.prices[1, t]
        C, b   = self._wealth(), self.beta
        d0     = C / (1.0 + b)
        d1     = b * C / (1.0 + b)

        self._shares[0] =  direction * (d0 / p0)
        self._shares[1] = -direction * (d1 / p1)
        self._cash      += direction * (d1 - d0)   # net cash change

        self._cash -= self.tc * (d0 + d1)
        rm, rs = self._roll_stats(t)
        self._entry_z = float((self._spread[t] - rm) / rs)
        self._pos = direction

    def _exit(self):
        t        = self._t
        p0, p1   = self.prices[0, t], self.prices[1, t]
        notional = abs(self._shares[0]) * p0 + abs(self._shares[1]) * p1
        self._cash    = self._wealth() - self.tc * notional
        self._shares[:] = 0.0
        self._entry_z   = 0.0
        self._pos       = 0

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
        self._pos     = 0
        self._entry_z = 0.0
        return self._obs(), {}

    def step(self, action: int):
        old_wealth  = self._wealth()
        desired_pos = self._ACT2POS[action]

        if desired_pos != self._pos:
            if self._pos != 0:
                self._exit()                  # close current position
            if desired_pos != 0:
                self._enter(desired_pos)      # open new position (may be reversal)

        self._cash *= (1.0 + self.rf_daily)
        self._t    += 1

        reward = float(self._wealth() - old_wealth)
        done   = self._t >= self._end
        return self._obs(), reward, done, False, {}
