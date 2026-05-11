"""ActionTranslator strategies.

Every translator converts an agent's raw action into the canonical
(position_signal, weights) pair the env uses. Each translator also exposes
the matching action_space so agents know what to emit.

Three implementations corresponding to the three training modes:
    StoppingOnlyTranslator : Discrete(3) -> {-1, 0, +1} ; weights fixed
    SizingOnlyTranslator   : Box(n_legs) -> weights     ; signal fixed (+1)
    CompositeTranslator    : Dict{stopping, sizing}     ; both

Implementations are intentionally tiny — they are decision-space adapters,
nothing more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..spaces import Box, Discrete, Dict as DictSpace, Space


# discrete index -> position_signal
_STOPPING_INDEX_TO_SIGNAL = {0: -1, 1: 0, 2: +1}


class ActionTranslator(ABC):
    n_legs: int = 2

    @property
    @abstractmethod
    def action_space(self) -> Space: ...

    @abstractmethod
    def translate(self, action: Any) -> tuple[int, np.ndarray]:
        """Return (position_signal, weights) for the env to apply."""


class StoppingOnlyTranslator(ActionTranslator):
    """Agent picks position_signal; weights are fixed (typ. hedge-ratio weights)."""

    def __init__(self, fixed_weights: tuple[float, ...] = (1.0, -1.0)):
        self._weights = np.array(fixed_weights, dtype=np.float32)
        self.n_legs = len(fixed_weights)

    @property
    def action_space(self) -> Space:
        return Discrete(3)

    def translate(self, action: Any) -> tuple[int, np.ndarray]:
        idx = int(action)
        if idx not in _STOPPING_INDEX_TO_SIGNAL:
            raise ValueError(f"Stopping action must be in {{0,1,2}}, got {idx}")
        return _STOPPING_INDEX_TO_SIGNAL[idx], self._weights.copy()


class SizingOnlyTranslator(ActionTranslator):
    """Agent picks continuous weights; position_signal is fixed (+1 default).

    Useful for training a sizer in isolation: we pretend we're always in a
    long-spread position and just learn how to allocate.
    """

    def __init__(
        self,
        n_legs: int = 2,
        weight_low: float = -1.0,
        weight_high: float = 1.0,
        fixed_signal: int = 1,
    ):
        self.n_legs = n_legs
        self._low = weight_low
        self._high = weight_high
        self._signal = fixed_signal

    @property
    def action_space(self) -> Space:
        return Box.make(self._low, self._high, shape=(self.n_legs,))

    def translate(self, action: Any) -> tuple[int, np.ndarray]:
        w = np.asarray(action, dtype=np.float32)
        if w.shape != (self.n_legs,):
            raise ValueError(f"Sizing action shape {w.shape} != ({self.n_legs},)")
        return self._signal, w


class CompositeTranslator(ActionTranslator):
    """Agent picks both stopping and sizing as a Dict action."""

    def __init__(
        self,
        n_legs: int = 2,
        weight_low: float = -1.0,
        weight_high: float = 1.0,
    ):
        self.n_legs = n_legs
        self._stopping = StoppingOnlyTranslator(fixed_weights=tuple([0.0] * n_legs))
        self._sizing = SizingOnlyTranslator(
            n_legs=n_legs,
            weight_low=weight_low,
            weight_high=weight_high,
        )

    @property
    def action_space(self) -> Space:
        return DictSpace(
            spaces={
                "stopping": self._stopping.action_space,
                "sizing": self._sizing.action_space,
            }
        )

    def translate(self, action: Any) -> tuple[int, np.ndarray]:
        if not isinstance(action, dict):
            raise TypeError("CompositeTranslator expects a dict action")
        signal, _ = self._stopping.translate(action["stopping"])
        _, weights = self._sizing.translate(action["sizing"])
        return signal, weights
