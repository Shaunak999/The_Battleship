"""
Random AI strategy for Battleship.

The simplest possible baseline — picks a random untested cell each turn.
Used as the control group when comparing AI performance.
"""

import random
from typing import Optional

from .base_ai import BaseAI


class RandomAI(BaseAI):
    """Baseline AI that fires at a uniformly random un-attacked cell.

    Algorithm
    ---------
    1. Build a set of all cells not yet attacked.
    2. Pick one at random.
    3. Return it.

    This gives an expected game length of roughly 95-97 shots on average
    (since 17 of 100 cells contain a ship, and randomness converges slowly).
    """

    def __init__(self, board_size: int = 10):
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Random"

    def reset(self) -> None:
        super().reset()
        # Pre-build a shuffled list for O(1) pops
        self._remaining: list[tuple[int, int]] = self._all_cells()
        random.shuffle(self._remaining)

    def choose_move(self) -> tuple[int, int]:
        """Pick a random cell that hasn't been attacked yet.

        Returns
        -------
        tuple[int, int]
            (row, col) of the chosen cell.

        Raises
        ------
        RuntimeError
            If there are no cells left (should never happen in a valid game).
        """
        if not self._remaining:
            raise RuntimeError("RandomAI: no cells left to attack")

        # Pop from the end of the shuffled list — O(1)
        return self._remaining.pop()

    def process_result(
        self,
        row: int,
        col: int,
        result: str,
        sunk_ship_name: Optional[str] = None,
        sunk_ship_size: Optional[int] = None,
    ) -> None:
        """Record the shot result.  Random AI doesn't change behaviour
        based on hits/misses, but we still record them for statistics."""
        self._record_shot(row, col, result, sunk_ship_name, sunk_ship_size)
