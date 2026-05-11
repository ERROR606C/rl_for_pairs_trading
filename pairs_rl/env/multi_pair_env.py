"""MultiPairEnv: wraps N independent PairEnvs.

Boilerplate-level extension to the multi-pair case. The contract is:
    - observation: dict {pair_id: per-pair obs vector}
    - action:      dict {pair_id: per-pair action}
    - reward:      sum of per-pair rewards (the simplest aggregation; a real
                   capital-allocator agent will replace this term)
    - info:        dict {pair_id: per-pair info}

A separate cross-pair "Allocator" agent that distributes capital across pairs
is left as a future extension — placeholder hook below.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pair_env import PairEnv


class MultiPairEnv:
    def __init__(self, envs: dict[str, PairEnv]):
        if not envs:
            raise ValueError("MultiPairEnv requires at least one pair env")
        self.envs = envs

    @property
    def pair_ids(self) -> list[str]:
        return list(self.envs.keys())

    @property
    def action_space(self) -> dict[str, Any]:
        return {pid: env.action_space for pid, env in self.envs.items()}

    @property
    def observation_space(self) -> dict[str, Any]:
        return {pid: env.observation_space for pid, env in self.envs.items()}

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs: dict[str, np.ndarray] = {}
        info: dict[str, Any] = {}
        for i, (pid, env) in enumerate(self.envs.items()):
            sub_seed = None if seed is None else seed + i
            o, inf = env.reset(seed=sub_seed)
            obs[pid] = o
            info[pid] = inf
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs: dict[str, np.ndarray] = {}
        info: dict[str, Any] = {}
        total_reward = 0.0
        terminated = True
        truncated = True
        for pid, env in self.envs.items():
            o, r, term, trunc, inf = env.step(action[pid])
            obs[pid] = o
            info[pid] = inf
            total_reward += r
            terminated = terminated and term
            truncated = truncated and trunc
        return obs, total_reward, terminated, truncated, info

    # Placeholder hook for cross-pair capital allocation. A future Allocator
    # would reweight each env's effective exposure before stepping.
    def set_capital_weights(self, weights: dict[str, float]) -> None:  # noqa: D401
        """Reserved for the future cross-pair allocator. No-op for now."""
        _ = weights
