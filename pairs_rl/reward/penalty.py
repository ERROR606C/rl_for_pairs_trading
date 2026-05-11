"""PenaltyTerm interface.

Each penalty is a small object that, given the env transition, returns a
nonnegative scalar "violation" amount. The reward function aggregates these
with their weights into the final reward; for now the framework ships with
the interface only (no concrete penalties implemented, per the user's brief).

We keep penalties *named* and *separately tracked* (not pre-summed) so that
later, when we add a Lagrangian / CMDP solver, the per-constraint signals are
already available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionContext:
    """All quantities a penalty might want, packed into one object."""

    prev_position: np.ndarray
    new_position: np.ndarray
    weights: np.ndarray
    position_signal: int
    asset_returns: np.ndarray
    pnl: float
    time_in_position: int


class PenaltyTerm(ABC):
    """Pluggable penalty: returns a *nonnegative* violation amount."""

    name: str = "penalty"
    weight: float = 1.0

    @abstractmethod
    def __call__(self, ctx: TransitionContext) -> float: ...
