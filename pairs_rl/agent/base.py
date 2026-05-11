"""Agent abstract base class.

This is a minimal interface — `act()` and `observe()` — chosen so it is
trivially adaptable to any RL library later. Concrete RL algorithms (DQN,
PPO, SAC) are intentionally not implemented at this stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: Any
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> Any: ...

    def observe(self, transition: Transition) -> None:
        """Called by the training loop after every env.step.

        Default: no-op. Learning agents override this to push to a replay
        buffer / update parameters.
        """
