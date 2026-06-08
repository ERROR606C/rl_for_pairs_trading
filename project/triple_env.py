import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from enum import Enum
from numpy.lib.stride_tricks import sliding_window_view


class Actions(Enum):
    Maintain = 0
    Swap     = 1


class TripleData:
    """Container for 3-asset price data, mirrors PairData interface."""
    def __init__(self, path: str):
        df = pd.read_csv(path, index_col=0)
        self.labels     = df.columns.tolist()[:3]
        self.timestamps = df.index.to_numpy()
        self.s          = df.iloc[:, :3].to_numpy(dtype=np.float64).T  # (3, N)

    def __len__(self):
        return self.s.shape[1]

    def slice(self, start=0, end=None):
        new            = object.__new__(TripleData)
        new.labels     = self.labels
        new.timestamps = self.timestamps[start:end]
        new.s          = self.s[:, start:end]
        return new


class TripleMarket(gym.Env):
    """
    Pairs-trading environment for three cointegrated assets.

    Spread:  ξ_t = coint_vec @ log(prices_t) − spread_intercept
    Z-score: z_t = ξ_t / fixed_std

    Position sizing (enter at time t):
        shares_k = −sign(ξ_t) · v_k · C / (‖v‖₁ · p_k)

    Speed: on each reset the full spread, MACD, and rolling stats series are
    precomputed once with vectorised NumPy. Each step() then does O(1) lookups
    instead of re-running EMA loops over a sliding window.
    """

    def __init__(self, triple_data: TripleData,
                 coint_vec,
                 spread_intercept: float = 0.0,
                 fixed_std: float        = None,
                 past_window: int        = 60,
                 initial_capital: float  = 10_000,
                 transaction_cost: float = 0.0):

        self._data             = triple_data
        self._v                = np.asarray(coint_vec, dtype=np.float64)
        self._v_norm           = float(np.abs(self._v).sum())
        self._spread_intercept = spread_intercept
        self._fixed_std        = fixed_std
        self._past_window      = past_window
        self._initial_capital  = initial_capital
        self._transaction_cost = transaction_cost

        self._capital             = initial_capital
        self._position            = np.zeros(3)
        self._in_position         = 0
        self._entry_z_score       = 0.0
        self._entry_spread        = 0.0
        self._last_trade_notional = 0.0
        self._t                   = past_window

        # Precomputed series (populated on reset)
        self._spread_pre    = None
        self._macd_pre      = None
        self._sig_pre       = None
        self._hist_pre      = None
        self._roll_mean_pre = None
        self._roll_std_pre  = None

        self.observation_space = spaces.Dict({
            "spread":         spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "spread_z_score": spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "entry_spread":   spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "entry_z_score":  spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "position":       spaces.Discrete(2),
            "macd":           spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "macd_signal":    spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "macd_hist":      spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "rolling_mean":   spaces.Box(0.0,     np.inf, (1,), np.float32),
            "rolling_std":    spaces.Box(0.0,     np.inf, (1,), np.float32),
        })
        self.action_space = spaces.Discrete(2)

    # ── Precomputation (called once per episode) ──────────────────────────────

    @staticmethod
    def _ema_series(values: np.ndarray, span: int) -> np.ndarray:
        k   = 2.0 / (span + 1)
        ema = np.empty(len(values))
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = values[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _precompute(self):
        """Vectorised precomputation of spread, MACD, and rolling stats."""
        log_p = np.log(self._data.s)                              # (3, T)
        spr   = self._v @ log_p - self._spread_intercept          # (T,)
        self._spread_pre = spr

        # MACD on |spread| — three EMA passes over the full series (fast)
        abs_spr        = np.abs(spr)
        ema12          = self._ema_series(abs_spr, 12)
        ema26          = self._ema_series(abs_spr, 26)
        macd_          = ema12 - ema26
        sig_           = self._ema_series(macd_, 9)
        self._macd_pre = macd_
        self._sig_pre  = sig_
        self._hist_pre = macd_ - sig_

        # Rolling mean / std (window = 50) — pure NumPy, no Python loop
        W    = 50
        wins = sliding_window_view(spr, W)      # (T-W+1, W); wins[i] = spr[i:i+W]
        self._roll_mean_pre = wins.mean(axis=1)
        self._roll_std_pre  = wins.std(axis=1)
        # For time t: index = t - W  (valid ∀ t ≥ past_window = 60 > W = 50)

    # ── Spread / obs helpers (O(1) after precompute) ──────────────────────────

    def _raw_spread(self, t: int) -> float:
        return float(self._spread_pre[t])

    def _get_spread(self) -> np.ndarray:
        return np.array([self._spread_pre[self._t]], dtype=np.float32)

    def _get_spread_z_score(self) -> np.ndarray:
        s = self._spread_pre[self._t]
        if self._fixed_std is not None:
            return np.array([s / (self._fixed_std + 1e-8)], dtype=np.float32)
        idx = self._t - self._past_window
        std = self._roll_std_pre[idx] if idx < len(self._roll_std_pre) else 1.0
        return np.array([s / (std + 1e-8)], dtype=np.float32)

    def _get_macd(self):
        t = self._t
        return float(self._macd_pre[t]), float(self._sig_pre[t]), float(self._hist_pre[t])

    def _get_rolling_stats(self):
        idx = self._t - 50
        if idx < 0 or idx >= len(self._roll_mean_pre):
            return 0.0, 1.0
        return float(self._roll_mean_pre[idx]), float(self._roll_std_pre[idx])

    def _get_obs(self) -> dict:
        macd, sig, hist = self._get_macd()
        rm, rs          = self._get_rolling_stats()
        return {
            "spread":         self._get_spread(),
            "spread_z_score": self._get_spread_z_score(),
            "entry_spread":   np.array([self._entry_spread],  dtype=np.float32),
            "entry_z_score":  np.array([self._entry_z_score], dtype=np.float32),
            "position":       self._in_position,
            "macd":           np.array([macd],  dtype=np.float32),
            "macd_signal":    np.array([sig],   dtype=np.float32),
            "macd_hist":      np.array([hist],  dtype=np.float32),
            "rolling_mean":   np.array([rm],    dtype=np.float32),
            "rolling_std":    np.array([rs],    dtype=np.float32),
        }

    # ── Portfolio value ───────────────────────────────────────────────────────

    def _get_pnl(self) -> float:
        return self._capital + float(self._position @ self._data.s[:, self._t])

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._precompute()
        self._t                   = self._past_window
        self._in_position         = 0
        self._position            = np.zeros(3)
        self._capital             = self._initial_capital
        self._entry_capital       = None
        self._entry_z_score       = 0.0
        self._entry_spread        = 0.0
        self._last_trade_notional = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        old_pnl       = self._get_pnl()
        entry_capital = None
        entry_z       = 0.0

        if action == Actions.Swap.value:
            if self._in_position == 0:                      # ── enter ──
                prices  = self._data.s[:, self._t]
                spread  = self._spread_pre[self._t]
                sign    = -np.sign(spread) if spread != 0.0 else 1.0
                C       = self._capital
                self._position = sign * self._v * C / (self._v_norm * prices)
                self._capital -= float(self._position @ prices)
                self._last_trade_notional = float(np.abs(self._position) @ prices)
                self._capital -= self._transaction_cost * self._last_trade_notional
                self._entry_capital = self._capital
                self._entry_z_score = float(self._get_spread_z_score()[0])
                self._entry_spread  = float(self._spread_pre[self._t])

            else:                                           # ── exit ──
                entry_capital = self._entry_capital
                entry_z       = abs(self._entry_z_score)
                prices        = self._data.s[:, self._t]
                self._last_trade_notional = float(np.abs(self._position) @ prices)
                self._capital  = self._get_pnl()
                self._capital -= self._transaction_cost * self._last_trade_notional
                self._position      = np.zeros(3)
                self._entry_capital = None
                self._entry_z_score = 0.0
                self._entry_spread  = 0.0

            self._in_position = 1 - self._in_position

        self._t   += 1
        terminated = self._t >= len(self._data) - 1
        new_pnl    = self._get_pnl()

        # Dense unrealized PnL reward — agent feels every losing tick immediately.
        # TC is already deducted from capital on entry/exit, so the step where a
        # trade fires shows a negative reward spike proportional to TC * notional.
        reward = new_pnl - old_pnl

        return self._get_obs(), reward, terminated, False, {}
