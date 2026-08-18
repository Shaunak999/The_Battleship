"""
Neural AI sub-package — DQN + CNN agent for Battleship.

Modules
-------
battleship_env — Gymnasium-compatible Battleship environment
train          — DQN training loop with replay buffer & target network
evaluate       — Evaluate trained agent against baselines
"""

from .battleship_env import BattleshipEnv

__all__ = ["BattleshipEnv"]
