"""RandomAgent: samples from the action space.

Used for smoke-testing the pipeline. It doesn't learn anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..spaces import Space
from .base import Agent


class RandomAgent(Agent):
    def __init__(self, action_space: Space, seed: int | None = None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> Any:
        return self.action_space.sample(self.rng)
