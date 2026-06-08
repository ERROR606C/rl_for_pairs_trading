import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch


class PairTradingEnv(gym.Env):
    """
    Pairs trading environment — z-score + rolling HL + Hybrid model predictions.

    Observation (5 floats):
        z_score    – rolling past_window z-score of the spread
        position   – -1 / 0 / +1
        hl_norm    – rolling half-life normalised to [0, 1]
        hybrid_mu  – Hybrid model predicted next-day z-score
        hybrid_var – Hybrid model predicted variance (uncertainty)

    hybrid_pack (dict, optional):
        model        – GaussianHybrid instance (eval mode)
        ff_mean/std  – shape (1, 6) numpy arrays
        cnn_mean/std – shape (1, seq_len) numpy arrays
        seq_len      – int

    If hybrid_pack is None, mu=0 and var=1 are used as placeholders.
    """

    _ACT2POS = {0: 0, 1: 1, 2: -1}

    def __init__(
        self,
        prices:         np.ndarray,
        beta:           float,
        intercept:      float,
        hybrid_pack:    dict  = None,
        initial_wealth: float = 10_000.0,
        tc:             float = 0.0001,
        rf_annual:      float = 0.05,
        past_window:    int   = 60,
        train_end:      int   = None,
        episode_len:    int   = None,
    ):
        super().__init__()
        N = prices.shape[1]
        self.prices         = prices.astype(np.float64)
        self.beta           = float(beta)
        self.intercept      = float(intercept)
        self.initial_wealth = float(initial_wealth)
        self.tc             = float(tc)
        self.rf_daily       = (1.0 + rf_annual) ** (1.0 / 252) - 1.0
        self.past_window    = int(past_window)
        self._N             = N

        self._min_t  = past_window + 3
        self.train_end   = int(train_end)   if train_end   is not None else N
        self.episode_len = int(episode_len) if episode_len is not None else (N - self._min_t)

        spread = np.log(prices[0]) - self.beta * np.log(prices[1]) - float(intercept)
        s      = pd.Series(spread)

        # Rolling z-score
        rm      = s.rolling(past_window).mean()
        rs      = s.rolling(past_window).std().clip(lower=1e-8)
        self._z = ((s - rm) / rs).to_numpy(dtype=np.float64)

        # Rolling half-life
        min_p   = max(10, past_window // 3)
        rl_phi  = s.rolling(past_window, min_periods=min_p).corr(s.shift(1))
        phi_c   = rl_phi.abs().clip(lower=1e-4, upper=1 - 1e-4)
        rl_hl   = -np.log(2) / np.log(phi_c)
        hl_norm = (rl_hl.clip(lower=1, upper=252) / 252).fillna(0.5)
        self._hl_norm = hl_norm.to_numpy(dtype=np.float64)

        # Hybrid predictions (one forward pass over all valid t)
        self._mu_pred  = np.zeros(N, dtype=np.float32)
        self._var_pred = np.ones(N,  dtype=np.float32)
        if hybrid_pack is not None:
            self._run_hybrid(s, hybrid_pack, N)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

        self._t      = self._min_t
        self._end    = self._min_t + self.episode_len
        self._cash   = float(initial_wealth)
        self._shares = np.zeros(2, dtype=np.float64)
        self._pos    = 0

    # ── Hybrid inference ─────────────────────────────────────────────────

    def _run_hybrid(self, s: pd.Series, hp: dict, N: int):
        z_s     = pd.Series(self._z)
        seq_len = hp['seq_len']

        ema12     = s.ewm(span=12, adjust=False).mean()
        ema26     = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
        momentum  = z_s.diff(5)
        roll_mean = z_s.rolling(20).mean()
        roll_std  = z_s.rolling(20).std()
        z_diff    = z_s.diff()

        X_ff = np.column_stack([
            z_s.values, momentum.values, macd_line.values,
            macd_hist.values, roll_mean.values, roll_std.values,
        ])
        X_cnn = np.full((N, seq_len), np.nan)
        for t in range(seq_len - 1, N):
            X_cnn[t] = z_diff.values[t - seq_len + 1 : t + 1]

        valid = ~np.any(np.isnan(X_ff), axis=1) & ~np.any(np.isnan(X_cnn), axis=1)
        if valid.sum() == 0:
            return

        ff_mean  = torch.tensor(hp['ff_mean'],  dtype=torch.float32)
        ff_std   = torch.tensor(hp['ff_std'],   dtype=torch.float32)
        cnn_mean = torch.tensor(hp['cnn_mean'], dtype=torch.float32)
        cnn_std  = torch.tensor(hp['cnn_std'],  dtype=torch.float32)

        X_ff_n  = (torch.tensor(X_ff[valid].astype(np.float32))  - ff_mean)  / ff_std
        X_cnn_n = (torch.tensor(X_cnn[valid].astype(np.float32)) - cnn_mean) / cnn_std
        z_now   = X_ff_n[:, 0]

        hp['model'].eval()
        with torch.no_grad():
            mu_t, lv_t = hp['model'](X_cnn_n, z_now)

        self._mu_pred[valid]  = mu_t.numpy()
        self._var_pred[valid] = np.exp(lv_t.numpy()).clip(1e-6, 10.0)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _safe_t(self) -> int:
        return min(self._t, self._N - 1)

    def _obs(self) -> np.ndarray:
        t = self._safe_t()
        return np.array([
            self._z[t],
            float(self._pos),
            self._hl_norm[t],
            float(self._mu_pred[t]),
            float(self._var_pred[t]),
        ], dtype=np.float32)

    def _wealth(self) -> float:
        return float(self._cash + self._shares @ self.prices[:, self._safe_t()])

    # ── Position management ───────────────────────────────────────────────

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
        self._cash      = self._wealth() - self.tc * notional
        self._shares[:] = 0.0
        self._pos       = 0

    # ── Gym API ───────────────────────────────────────────────────────────

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
