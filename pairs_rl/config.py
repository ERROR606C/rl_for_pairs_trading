"""Top-level config dataclasses.

Kept deliberately small. Each subsystem (data, state, action, reward, etc.)
takes its own config object so they can be swapped/extended without touching
this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    """Which decision the agent is responsible for.

    STOPPING_ONLY: agent picks position_signal in {-1, 0, +1}; weights fixed.
    SIZING_ONLY:   agent picks continuous weights; position_signal fixed (+1).
    COMPOSITE:     agent picks both (Dict action space).
    """

    STOPPING_ONLY = "stopping_only"
    SIZING_ONLY = "sizing_only"
    COMPOSITE = "composite"


@dataclass
class PairEnvConfig:
    """Configuration for a single-pair environment.

    Only env-level concerns live here. Sub-component configs (e.g. data,
    regime detector) are passed in as objects rather than wedged into this
    dataclass — keeps things composable.
    """

    mode: Mode = Mode.STOPPING_ONLY
    episode_length: int | None = None      # None = until data runs out
    initial_position_signal: int = 0       # starts flat by default
    initial_weights: tuple[float, float] = (1.0, -1.0)
    feature_keys: tuple[str, ...] = ("z_score", "spread")
    include_position_in_state: bool = True
    include_time_in_position: bool = True
    extras: dict[str, Any] = field(default_factory=dict)
