"""
Battleship AI strategies package.

Provides three AI strategies of increasing sophistication:

    RandomAI        — uniformly random baseline
    HuntTargetAI    — checkerboard hunting + directional targeting
    ProbabilityAI   — probability-density heat-map analysis

All strategies implement the BaseAI interface:
    choose_move()    → (row, col)
    process_result() → update internal state
    reset()          → reinitialise for a new game

Usage
-----
    from ai import get_ai

    ai = get_ai("probability")
    move = ai.choose_move()
    ai.process_result(*move, "hit")
"""

from .base_ai import BaseAI, SHIP_DEFINITIONS, BOARD_SIZE
from .random_ai import RandomAI
from .hunt_target_ai import HuntTargetAI
from .probability_ai import ProbabilityAI

AI_STRATEGIES = {
    "random": RandomAI,
    "hunt_target": HuntTargetAI,
    "probability": ProbabilityAI,
}

AI_STRATEGY_NAMES = {
    "random": "Random",
    "hunt_target": "Hunt & Target",
    "probability": "Probability",
}

_DESCRIPTIONS = {
    "random": "Fires at a random un-attacked cell each turn. Baseline strategy.",
    "hunt_target": (
        "Uses a checkerboard pattern to hunt for ships, then switches to "
        "directional targeting to sink discovered ships."
    ),
    "probability": (
        "Calculates a probability heat-map of valid ship placements for "
        "every cell and fires at the most likely location."
    ),
}


def get_ai(strategy: str, board_size: int = BOARD_SIZE) -> BaseAI:
    """Factory function to create an AI instance by strategy name.

    Parameters
    ----------
    strategy : str
        One of "random", "hunt_target", "probability", or a human-friendly
        variant such as "Hunt & Target".
    board_size : int
        Board size (default 10).

    Raises
    ------
    ValueError
        If *strategy* is not a recognised strategy name.
    """
    key = strategy.strip().lower().replace(" ", "_").replace("&", "").replace("-", "_")
    key = key.replace("__", "_")
    key = key.replace("hunttarget", "hunt_target")

    if key not in AI_STRATEGIES:
        valid = ", ".join(AI_STRATEGIES.keys())
        raise ValueError(f"Unknown AI strategy '{strategy}'. Valid strategies: {valid}")

    return AI_STRATEGIES[key](board_size=board_size)


def list_strategies() -> list[dict]:
    """Return available strategies with metadata, for the UI to render."""
    return [
        {
            "key": key,
            "name": AI_STRATEGY_NAMES[key],
            "description": _DESCRIPTIONS.get(key, ""),
        }
        for key in AI_STRATEGIES
    ]


__all__ = [
    "BaseAI",
    "RandomAI",
    "HuntTargetAI",
    "ProbabilityAI",
    "get_ai",
    "list_strategies",
    "AI_STRATEGIES",
    "AI_STRATEGY_NAMES",
    "SHIP_DEFINITIONS",
    "BOARD_SIZE",
]