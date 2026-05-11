"""RewardFunction.

Composed of:
    - a base PnL term (mandatory)
    - a list of PenaltyTerm objects (empty by default; populated later)

The reward function returns a scalar reward AND an `info` dict that breaks
down the components. This is useful both for debugging and for any future
Lagrangian / multi-objective extensions, which need the per-term values
without aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .penalty import PenaltyTerm, TransitionContext


class BasePnL:
    """Default PnL term: dot(position, asset_returns).

    `asset_returns` here is a vector of per-leg simple returns over the step.
    Position is the per-leg dollar exposure (= position_signal * weights).
    """

    name: str = "pnl"

    def __call__(self, ctx: TransitionContext) -> float:
        return float(ctx.pnl)


@dataclass
class RewardFunction:
    """Aggregates base reward + (weighted) penalties.

    The framework deliberately leaves `penalties` empty until you add some.
    """

    base: BasePnL = field(default_factory=BasePnL)
    penalties: Sequence[PenaltyTerm] = field(default_factory=tuple)

    def __call__(self, ctx: TransitionContext) -> tuple[float, dict[str, float]]:
        components: dict[str, float] = {self.base.name: self.base(ctx)}
        for p in self.penalties:
            components[p.name] = float(p(ctx))
        reward = components[self.base.name] - sum(
            p.weight * components[p.name] for p in self.penalties
        )
        components["total"] = reward
        return reward, components

    @staticmethod
    def compute_pnl(prev_position: np.ndarray, asset_returns: np.ndarray) -> float:
        """Convenience helper used by the env."""
        return float(np.dot(prev_position, asset_returns))
