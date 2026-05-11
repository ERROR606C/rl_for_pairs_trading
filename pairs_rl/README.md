# pairs_rl — baseline framework

Modular RL framework for pairs trading. This is **boilerplate only**: no policies, no constraints, no real regime detector. Every extension point is exposed as an interface with a trivial default implementation.

## Layout

```
pairs_rl/
  config.py           Mode enum, PairEnvConfig
  spaces.py           Minimal Discrete/Box/Dict (no gymnasium dependency)
  data/
    market_data.py    MarketData ABC + SyntheticCointegratedPair
  state/
    regime.py         RegimeDetector ABC + NoOpRegimeDetector
    state_builder.py  Composes obs from market features + position + regime
  action/
    translator.py     StoppingOnly / SizingOnly / Composite action adapters
  reward/
    penalty.py        PenaltyTerm ABC + TransitionContext
    reward.py         RewardFunction (base PnL + composable penalties)
  env/
    pair_env.py       Gym-style single-pair env
    multi_pair_env.py Wraps N PairEnvs
  agent/
    base.py           Agent ABC + Transition dataclass
    random_agent.py   Samples from action_space (for smoke tests)
  smoke_test.py       End-to-end wiring check
```

## The contract

Every step the env resolves an action into a canonical pair:

```
position_signal ∈ {-1, 0, +1}    # short-spread / flat / long-spread
weights         ∈ R^n             # per-leg allocation
final_position  = position_signal * weights
reward          = final_position · asset_returns − Σ λ_i · penalty_i
```

`ActionTranslator` decides which side(s) of this pair the agent controls. Switching modes never touches env code.

## Mode switch

```python
from pairs_rl.action import StoppingOnlyTranslator, SizingOnlyTranslator, CompositeTranslator
```

- `StoppingOnlyTranslator` — agent picks `Discrete(3)`, weights fixed (typ. hedge ratio).
- `SizingOnlyTranslator`   — agent picks `Box(n_legs,)`, signal fixed (default +1).
- `CompositeTranslator`    — agent picks `Dict{stopping, sizing}`.

## Extensions wired in but not implemented

- **Regime detection** — `RegimeDetector.update(features) -> np.ndarray`. Output is concatenated into the observation by `StateBuilder`. Default `NoOpRegimeDetector` returns an empty vector.
- **Constraint penalties** — `PenaltyTerm.__call__(ctx) -> float`. `RewardFunction.penalties` is an empty tuple by default. Per-term values are passed through in `info["reward_components"]` so a future Lagrangian/CMDP layer can read them.
- **Multi-pair** — `MultiPairEnv` already composes N `PairEnv`s. A future capital-allocator agent will use `set_capital_weights(...)`.

## Smoke test

```
python -m pairs_rl.smoke_test
```

Runs the three single-pair modes plus the multi-pair wrapper with a `RandomAgent`. Should print step counts and total rewards for each mode and finish with `OK`.

## Dependencies

Only `numpy`. The minimal `spaces.py` module exists so we don't have a hard dependency on `gymnasium`. Drop in `gymnasium.spaces` later if you want full gym compatibility.
