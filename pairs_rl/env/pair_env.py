"""PairEnv: gym-style environment for a single cointegrated pair.

Step contract:
    obs, reward, terminated, truncated, info = env.step(action)

`action` shape depends on the configured ActionTranslator. The env itself
does not know whether it is in stopping-only / sizing-only / composite mode;
that is entirely encapsulated by the translator object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..action.translator import ActionTranslator
from ..config import PairEnvConfig
from ..data.market_data import MarketData, MarketStep
from ..reward.penalty import TransitionContext
from ..reward.reward import RewardFunction
from ..spaces import Box, Space
from ..state.state_builder import StateBuilder


@dataclass
class StepInfo:
    """Returned in info dict each step (also serialised for logging)."""

    market_features: dict[str, float]
    position_signal: int
    weights: np.ndarray
    final_position: np.ndarray
    asset_returns: np.ndarray
    reward_components: dict[str, float]


class PairEnv:
    """Single-pair environment.

    Composes:
        market : MarketData            (state of the world)
        state  : StateBuilder          (observation construction)
        action : ActionTranslator      (action-space adapter)
        reward : RewardFunction        (PnL + penalties)
        config : PairEnvConfig         (episode-level knobs)

    The env is gymnasium-compatible in spirit: reset()/step() return the
    canonical 5-tuple shape. We don't import gymnasium so the package stays
    dependency-light.
    """

    def __init__(
        self,
        market: MarketData,
        state_builder: StateBuilder,
        action_translator: ActionTranslator,
        reward_fn: RewardFunction,
        config: PairEnvConfig | None = None,
    ):
        self.market = market
        self.state_builder = state_builder
        self.action_translator = action_translator
        self.reward_fn = reward_fn
        self.config = config or PairEnvConfig()

        # Mutable per-episode state
        self._position_signal: int = self.config.initial_position_signal
        self._weights: np.ndarray = np.array(self.config.initial_weights, dtype=np.float32)
        self._time_in_position: int = 0
        self._steps_taken: int = 0
        self._last_step: MarketStep | None = None

    # ----- gym-style API --------------------------------------------------

    @property
    def action_space(self) -> Space:
        return self.action_translator.action_space

    @property
    def observation_space(self) -> Box:
        d = self.state_builder.observation_dim
        return Box.make(-np.inf, np.inf, shape=(d,))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._position_signal = self.config.initial_position_signal
        self._weights = np.array(self.config.initial_weights, dtype=np.float32)
        self._time_in_position = 0
        self._steps_taken = 0
        self.state_builder.reset()
        self._last_step = self.market.reset(seed=seed)
        obs = self.state_builder.build(
            self._last_step,
            self._position_signal,
            self._weights,
            self._time_in_position,
        )
        return obs, {"market_features": dict(self._last_step.features)}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._last_step is not None, "must call reset() before step()"

        # 1. translate action into the canonical (signal, weights) pair
        new_signal, new_weights = self.action_translator.translate(action)

        # 2. position that will be exposed to the NEXT price move
        prev_position = float(self._position_signal) * self._weights
        new_position = float(new_signal) * new_weights

        # 3. advance the market by one step
        prices_before = self._last_step.prices.copy()
        next_step = self.market.step()
        prices_after = next_step.prices

        # 4. compute per-leg simple returns over the step
        with np.errstate(divide="ignore", invalid="ignore"):
            asset_returns = np.where(
                prices_before > 0,
                (prices_after - prices_before) / prices_before,
                0.0,
            )

        # 5. PnL is earned on the position we *held* through the move.
        #    Convention: the action chosen at time t determines the position
        #    held from t to t+1, so PnL uses new_position. (Equivalent to the
        #    common "trade at close" convention.)
        pnl = RewardFunction.compute_pnl(new_position, asset_returns)

        # 6. build the reward
        ctx = TransitionContext(
            prev_position=prev_position,
            new_position=new_position,
            weights=new_weights,
            position_signal=new_signal,
            asset_returns=asset_returns,
            pnl=pnl,
            time_in_position=self._time_in_position,
        )
        reward, components = self.reward_fn(ctx)

        # 7. update env state
        if new_signal == 0:
            self._time_in_position = 0
        elif new_signal == self._position_signal:
            self._time_in_position += 1
        else:
            self._time_in_position = 1  # just entered (or flipped)
        self._position_signal = new_signal
        self._weights = new_weights
        self._steps_taken += 1
        self._last_step = next_step

        # 8. termination conditions
        terminated = bool(next_step.done)
        truncated = bool(
            self.config.episode_length is not None
            and self._steps_taken >= self.config.episode_length
        )

        # 9. build next observation
        obs = self.state_builder.build(
            next_step,
            self._position_signal,
            self._weights,
            self._time_in_position,
        )

        info: dict[str, Any] = {
            "market_features": dict(next_step.features),
            "position_signal": self._position_signal,
            "weights": self._weights.copy(),
            "final_position": new_position,
            "asset_returns": asset_returns,
            "reward_components": components,
        }
        return obs, float(reward), terminated, truncated, info
