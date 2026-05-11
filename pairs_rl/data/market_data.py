"""MarketData abstractions.

A MarketData provider feeds the env one step at a time. It is responsible for
the raw view of the market: prices for each leg of the pair, plus any derived
quantities it wants to expose as features (spread, z-score, etc.).

We deliberately keep the contract small: at every step we hand back a
`MarketStep` containing prices + a free-form feature dict. The StateBuilder
later picks which feature keys end up in the observation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MarketStep:
    """One snapshot of the market.

    prices : shape (n_legs,) raw prices for each leg of the pair.
    features : free-form dict of derived scalars (e.g. spread, z_score).
               StateBuilder selects from this by key.
    done : whether this is the terminal step (e.g. end of data).
    """

    prices: np.ndarray
    features: dict[str, float]
    done: bool = False


class MarketData(ABC):
    """Iterator-style market data provider.

    Lifecycle:
        reset() -> MarketStep   # initial snapshot at t=0
        step()  -> MarketStep   # advance to t+1
    """

    n_legs: int = 2

    @abstractmethod
    def reset(self, seed: int | None = None) -> MarketStep: ...

    @abstractmethod
    def step(self) -> MarketStep: ...

    @property
    @abstractmethod
    def feature_keys(self) -> tuple[str, ...]:
        """Names of features this provider exposes in MarketStep.features."""


# ---------------------------------------------------------------------------
# Synthetic generator: cointegrated pair driven by an OU spread
# ---------------------------------------------------------------------------


class SyntheticCointegratedPair(MarketData):
    """Two-asset cointegrated process for smoke testing the framework.

    Asset A is a geometric random walk; asset B is constructed so that
    log(A) - beta * log(B) is an Ornstein-Uhlenbeck process. This is the
    textbook cointegrated-pair toy.

    The point is not realism — it is to give the env something to consume so
    we can verify the wiring end-to-end.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        beta: float = 1.0,
        ou_mean: float = 0.0,
        ou_kappa: float = 0.05,
        ou_sigma: float = 0.01,
        drift_a: float = 0.0001,
        sigma_a: float = 0.01,
        zscore_window: int = 50,
        seed: int | None = None,
    ):
        self.n_steps = n_steps
        self.beta = beta
        self.ou_mean = ou_mean
        self.ou_kappa = ou_kappa
        self.ou_sigma = ou_sigma
        self.drift_a = drift_a
        self.sigma_a = sigma_a
        self.zscore_window = zscore_window
        self._seed = seed
        self._t = 0
        self._log_a: np.ndarray | None = None
        self._log_b: np.ndarray | None = None
        self._spread: np.ndarray | None = None

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return ("spread", "z_score", "log_a", "log_b")

    def _generate(self, seed: int | None) -> None:
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        n = self.n_steps
        # Asset A: log price as random walk with drift
        eps_a = rng.normal(self.drift_a, self.sigma_a, size=n)
        log_a = np.cumsum(eps_a)
        # Spread (OU) drives the deviation of log_b from beta-scaled log_a
        spread = np.zeros(n)
        spread[0] = self.ou_mean
        eps_s = rng.normal(0.0, self.ou_sigma, size=n)
        for t in range(1, n):
            spread[t] = (
                spread[t - 1]
                + self.ou_kappa * (self.ou_mean - spread[t - 1])
                + eps_s[t]
            )
        # log_b such that spread = log_a - beta * log_b  =>  log_b = (log_a - spread) / beta
        log_b = (log_a - spread) / self.beta
        self._log_a = log_a
        self._log_b = log_b
        self._spread = spread

    def _snapshot(self) -> MarketStep:
        t = self._t
        assert self._log_a is not None and self._log_b is not None and self._spread is not None
        log_a = float(self._log_a[t])
        log_b = float(self._log_b[t])
        spread = float(self._spread[t])
        # Rolling z-score of the spread (causal: uses [max(0,t-w+1):t+1])
        w = self.zscore_window
        lo = max(0, t - w + 1)
        window = self._spread[lo : t + 1]
        mu = float(window.mean())
        sd = float(window.std(ddof=0))
        z = 0.0 if sd < 1e-12 else (spread - mu) / sd

        prices = np.array([np.exp(log_a), np.exp(log_b)], dtype=np.float64)
        features = {
            "spread": spread,
            "z_score": z,
            "log_a": log_a,
            "log_b": log_b,
        }
        return MarketStep(prices=prices, features=features, done=(t >= self.n_steps - 1))

    def reset(self, seed: int | None = None) -> MarketStep:
        if seed is not None:
            self._seed = seed
        self._generate(self._seed)
        self._t = 0
        return self._snapshot()

    def step(self) -> MarketStep:
        if self._t >= self.n_steps - 1:
            # already at the end; return terminal snapshot
            return self._snapshot()
        self._t += 1
        return self._snapshot()
