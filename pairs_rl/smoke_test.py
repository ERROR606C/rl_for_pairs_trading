"""Smoke test: run an episode of each mode end-to-end with a RandomAgent.

Verifies that every component wires together: data flows, observations have
the right shape, actions translate correctly, rewards aggregate, and the
multi-pair wrapper composes correctly.

Run from the project root:
    python -m pairs_rl.smoke_test
"""

from __future__ import annotations

import numpy as np

from .action import CompositeTranslator, SizingOnlyTranslator, StoppingOnlyTranslator
from .agent import RandomAgent, Transition
from .config import Mode, PairEnvConfig
from .data import SyntheticCointegratedPair
from .env import MultiPairEnv, PairEnv
from .reward import BasePnL, RewardFunction
from .state import NoOpRegimeDetector, StateBuilder


def make_env(mode: Mode, seed: int = 0) -> PairEnv:
    market = SyntheticCointegratedPair(n_steps=200, seed=seed)
    state_builder = StateBuilder(
        feature_keys=("z_score", "spread"),
        include_position=True,
        include_time_in_position=True,
        regime_detector=NoOpRegimeDetector(),
    )
    if mode == Mode.STOPPING_ONLY:
        translator = StoppingOnlyTranslator(fixed_weights=(1.0, -1.0))
    elif mode == Mode.SIZING_ONLY:
        translator = SizingOnlyTranslator(n_legs=2)
    else:
        translator = CompositeTranslator(n_legs=2)

    reward_fn = RewardFunction(base=BasePnL(), penalties=())
    config = PairEnvConfig(
        mode=mode,
        episode_length=100,
        feature_keys=("z_score", "spread"),
    )
    return PairEnv(
        market=market,
        state_builder=state_builder,
        action_translator=translator,
        reward_fn=reward_fn,
        config=config,
    )


def run_episode(env: PairEnv, max_steps: int = 100) -> dict[str, float]:
    agent = RandomAgent(env.action_space, seed=42)
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    n = 0
    for _ in range(max_steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(
            Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        total_reward += reward
        obs = next_obs
        n += 1
        if terminated or truncated:
            break
    return {"steps": float(n), "total_reward": float(total_reward), "obs_dim": float(obs.shape[0])}


def run_multipair() -> dict[str, float]:
    envs = {
        "pair_A": make_env(Mode.STOPPING_ONLY, seed=1),
        "pair_B": make_env(Mode.SIZING_ONLY, seed=2),
    }
    multi = MultiPairEnv(envs)
    agents = {pid: RandomAgent(env.action_space, seed=7 + i) for i, (pid, env) in enumerate(envs.items())}
    obs, _ = multi.reset(seed=0)
    total = 0.0
    steps = 0
    for _ in range(100):
        action = {pid: agents[pid].act(obs[pid]) for pid in multi.pair_ids}
        obs, r, term, trunc, _ = multi.step(action)
        total += r
        steps += 1
        if term or trunc:
            break
    return {"steps": float(steps), "total_reward": float(total)}


def main() -> None:
    print("=== single-pair smoke tests ===")
    for mode in (Mode.STOPPING_ONLY, Mode.SIZING_ONLY, Mode.COMPOSITE):
        env = make_env(mode)
        result = run_episode(env)
        print(f"  {mode.value:>16s}: {result}")
    print("=== multi-pair smoke test ===")
    print(f"  {run_multipair()}")
    print("OK")


if __name__ == "__main__":
    main()
