"""Regime detector interface.

Conceptually a RegimeDetector watches the recent history of market features
and emits some additional state (a label, a probability vector, an embedding —
anything). Its output is concatenated into the observation by StateBuilder, so
the rest of the pipeline is unaware that regime detection exists.

The default NoOpRegimeDetector emits an empty vector, which is the "no regime
info" mode. Real implementations (HMM, change-point, etc.) plug in here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RegimeDetector(ABC):
    """Emits a fixed-length feature vector summarising the current regime."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, features: dict[str, float]) -> np.ndarray:
        """Consume the latest market features; return regime feature vector.

        Implementations may also be stateful (e.g. an HMM filter). The returned
        array must have length == self.output_dim and dtype float32.
        """

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Optional: human-readable names for the regime features."""
        return tuple(f"regime_{i}" for i in range(self.output_dim))


class NoOpRegimeDetector(RegimeDetector):
    """Default: no regime info. Returns an empty vector."""

    def reset(self) -> None:
        pass

    def update(self, features: dict[str, float]) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return 0
