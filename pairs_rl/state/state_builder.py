"""StateBuilder: composes the env observation.

Inputs at each step:
    - MarketStep.features  (dict of named scalars from MarketData)
    - current position_signal and weights (from env's internal state)
    - time_in_position counter
    - RegimeDetector.update(...) output

Output: a 1-D float32 numpy array, plus a parallel list of feature names for
debuggability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regime import RegimeDetector, NoOpRegimeDetector
from ..data.market_data import MarketStep


@dataclass
class StateBuilder:
    feature_keys: tuple[str, ...] = ("z_score", "spread")
    include_position: bool = True
    include_time_in_position: bool = True
    regime_detector: RegimeDetector = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.regime_detector is None:
            self.regime_detector = NoOpRegimeDetector()

    def reset(self) -> None:
        self.regime_detector.reset()

    @property
    def observation_dim(self) -> int:
        return (
            len(self.feature_keys)
            + (2 if self.include_position else 0)   # position_signal + ||weights||
            + (1 if self.include_time_in_position else 0)
            + self.regime_detector.output_dim
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = list(self.feature_keys)
        if self.include_position:
            names += ["position_signal", "position_gross"]
        if self.include_time_in_position:
            names += ["time_in_position"]
        names += list(self.regime_detector.feature_names)
        return tuple(names)

    def build(
        self,
        market_step: MarketStep,
        position_signal: int,
        weights: np.ndarray,
        time_in_position: int,
    ) -> np.ndarray:
        parts: list[float | np.ndarray] = []
        for key in self.feature_keys:
            parts.append(float(market_step.features.get(key, 0.0)))
        if self.include_position:
            parts.append(float(position_signal))
            parts.append(float(np.sum(np.abs(weights))))
        if self.include_time_in_position:
            parts.append(float(time_in_position))
        regime_vec = self.regime_detector.update(market_step.features)
        obs = np.array(parts, dtype=np.float32)
        if regime_vec.size:
            obs = np.concatenate([obs, regime_vec.astype(np.float32)])
        return obs
