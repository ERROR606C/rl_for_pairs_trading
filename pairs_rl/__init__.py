"""pairs_rl: a modular RL framework for pairs trading.

Top-level layout:
    data/    - market data providers (the supplier of raw prices/features)
    state/   - state/observation construction + regime detectors
    action/  - action-space strategies (stopping-only / sizing-only / composite)
    reward/  - reward function with composable penalty terms
    env/     - gym-style environments (single pair + multi-pair wrapper)
    agent/   - agent abstract base class + RandomAgent for smoke tests
    config   - top-level dataclass configs

The design contract that ties everything together:
    every step, the env resolves an action into a (position_signal, weights) pair
        position_signal in {-1, 0, +1}   (short-spread / flat / long-spread)
        weights         in R^n           (per-leg allocation magnitudes/signs)
    the final per-leg position is:  position_signal * weights
    reward is computed from this position interacting with the next price move.
"""

from .config import PairEnvConfig, Mode
