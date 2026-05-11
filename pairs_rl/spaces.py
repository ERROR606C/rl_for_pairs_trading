"""Minimal action/observation space primitives.

These exist so the framework has no hard dependency on gymnasium. They expose
the small surface area we actually use: `sample()`, `shape`/`n`, and `contains()`.
If you want full gymnasium compatibility, the env's `action_space` /
`observation_space` attributes can be swapped for `gymnasium.spaces.*` instances
without other changes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any


class Space:
    """Abstract space."""

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        raise NotImplementedError


@dataclass
class Discrete(Space):
    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            return False
        return 0 <= xi < self.n


@dataclass
class Box(Space):
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype = np.float32

    @classmethod
    def make(cls, low: float, high: float, shape: tuple[int, ...]) -> "Box":
        return cls(
            low=np.full(shape, low, dtype=np.float32),
            high=np.full(shape, high, dtype=np.float32),
            shape=shape,
            dtype=np.float32,
        )

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            return False
        if x.shape != self.shape:
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))


@dataclass
class Dict(Space):
    """Heterogeneous composite space, keyed by string."""

    spaces: dict[str, Space]

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        return {k: s.sample(rng) for k, s in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(self.spaces[k].contains(v) for k, v in x.items())
