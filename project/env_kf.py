import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PairTradingEnv(gym.Env):
    """
    Pairs trading environment — spread-agnostic.

    The env knows nothing about spreads or z-scores.
    The agent (or strategy) decides when and at what beta to enter.

    Action: (direction, beta)
        direction : 0 = flat/exit, 1 = long, 2 = short
        beta      : hedge ratio — dollars in B = beta * dollars in A
                    (ignored when direction=0)

    Observation: [log_p0, log_p1, position, entry_beta]
        log_p0/p1   : current log prices
        position    : -1 / 0 / +1
        entry_beta  : beta locked in at entry (0 if flat)
    """

    _DIR = {0: 0, 1: 1, 2: -1}

    def __init__(
        self,
        prices:         np.ndarray,
        initial_wealth: float = 10_000.0,
        tc:             float = 0.0001,
        episode_len:    int   = None,  # type: ignore[assignment]
    ):
        super().__init__()
        self.prices         = prices.astype(np.float64)
        self.N              = prices.shape[1]
        self.initial_wealth = float(initial_wealth)
        self.tc             = float(tc)
        self.episode_len    = int(episode_len) if episode_len else self.N

        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        ))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(4,), dtype=np.float32
        )

        self._t          = 0
        self._end        = self.episode_len
        self._cash       = float(initial_wealth)
        self._shares     = np.zeros(2, dtype=np.float64)
        self._pos        = 0
        self._entry_beta = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────

    def _t_safe(self):
        return min(self._t, self.N - 1)

    def _wealth(self):
        return self._cash + self._shares @ self.prices[:, self._t_safe()]

    def _obs(self):
        t = self._t_safe()
        return np.array([
            np.log(self.prices[0, t]),
            np.log(self.prices[1, t]),
            float(self._pos),
            self._entry_beta,
        ], dtype=np.float32)

    # ── Position management ───────────────────────────────────────────────

    def _enter(self, direction: int, beta: float):
        t        = self._t_safe()
        p0, p1   = self.prices[0, t], self.prices[1, t]
        C        = self._wealth()
        d0       = C / (1.0 + beta)
        d1       = beta * C / (1.0 + beta)
        self._shares[0]  =  direction * d0 / p0
        self._shares[1]  = -direction * d1 / p1
        self._cash      +=  direction * (d1 - d0)
        self._cash      -=  self.tc * (d0 + d1)
        self._pos        =  direction
        self._entry_beta =  beta

    def _exit(self):
        t        = self._t_safe()
        p0, p1   = self.prices[0, t], self.prices[1, t]
        notional = abs(self._shares[0]) * p0 + abs(self._shares[1]) * p1
        self._cash       = self._wealth() - self.tc * notional
        self._shares[:]  = 0.0
        self._pos        = 0
        self._entry_beta = 0.0

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None, t_start=0):
        super().reset(seed=seed)
        self._t          = t_start
        self._end        = t_start + self.episode_len
        self._cash       = self.initial_wealth
        self._shares[:]  = 0.0
        self._pos        = 0
        self._entry_beta = 0.0
        return self._obs(), {}

    def step(self, action):
        direction_idx, beta_val = action
        desired = self._DIR[int(direction_idx)]
        beta    = float(beta_val[0]) if hasattr(beta_val, '__len__') else float(beta_val)

        old_wealth = self._wealth()

        if desired != self._pos:
            if self._pos != 0:
                self._exit()
            if desired != 0:
                self._enter(desired, beta)

        self._t   += 1
        done       = self._t >= self._end
        reward     = float(self._wealth() - old_wealth)
        return self._obs(), reward, done, False, {}
