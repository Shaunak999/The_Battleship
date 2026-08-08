"""
Hunt & Target AI strategy for Battleship.

A classic two-phase algorithm that is the most commonly described
"smart" Battleship strategy:

    HUNT  — search for ships using a parity/checkerboard pattern
    TARGET — once a hit is found, explore adjacent cells to sink the ship

This strategy significantly outperforms Random and typically averages
around 60-65 shots per game.
"""

import random
from collections import deque
from typing import Optional

from .base_ai import BaseAI


class HuntTargetAI(BaseAI):
    """Hunt & Target AI with checkerboard parity hunting and directional targeting.

    Hunt Phase
    ----------
    Instead of random shots, the AI uses a **checkerboard pattern** so that
    every ship of size ≥ 2 is guaranteed to be crossed.  Cells where
    ``(row + col) % 2 == 0`` are tried first (the "even parity" squares),
    which cuts the hunt space roughly in half.

    Target Phase
    ------------
    When a hit is scored:

    1. Push the four cardinal neighbours of the hit onto a target queue.
    2. Pop cells from the queue and fire.
    3. If a second consecutive hit is scored, lock onto that orientation
       (horizontal or vertical) and continue in that direction.
    4. When the ship is sunk, clear the target state and return to hunting.
    """

    def __init__(self, board_size: int = 10):
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Hunt & Target"

    def reset(self) -> None:
        super().reset()

        # --- Hunt state ---
        # Build the checkerboard hunt list (parity 0 first, then parity 1)
        parity0 = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if (r + c) % 2 == 0
        ]
        parity1 = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if (r + c) % 2 == 1
        ]
        random.shuffle(parity0)
        random.shuffle(parity1)
        # parity0 is tried first; parity1 is the fallback
        self._hunt_cells: list[tuple[int, int]] = parity1 + parity0  # pop from end → parity0 first

        # --- Target state ---
        self._target_queue: deque[tuple[int, int]] = deque()
        self._current_hits: list[tuple[int, int]] = []  # hits for the ship currently being targeted
        self._orientation: Optional[str] = None  # "horizontal" or "vertical" once determined
        self._is_targeting: bool = False

    # ------------------------------------------------------------------
    # choose_move
    # ------------------------------------------------------------------

    def choose_move(self) -> tuple[int, int]:
        """Return the next cell to attack.

        - In TARGET mode: use the target queue.
        - In HUNT mode: use the checkerboard list.
        """
        # --- Target mode ---
        while self._target_queue:
            cell = self._target_queue.popleft()
            if cell not in self.shots_taken and self._is_valid(*cell):
                return cell

        # If the queue is empty but we were targeting, fall through to hunt
        self._is_targeting = False

        # --- Hunt mode ---
        while self._hunt_cells:
            cell = self._hunt_cells.pop()
            if cell not in self.shots_taken:
                return cell

        # Fallback: shouldn't happen in a normal game, but be safe
        available = self._available_cells()
        if not available:
            raise RuntimeError("HuntTargetAI: no cells left to attack")
        return random.choice(available)

    # ------------------------------------------------------------------
    # process_result
    # ------------------------------------------------------------------

    def process_result(
        self,
        row: int,
        col: int,
        result: str,
        sunk_ship_name: Optional[str] = None,
        sunk_ship_size: Optional[int] = None,
    ) -> None:
        """Update AI state after a shot."""
        self._record_shot(row, col, result, sunk_ship_name, sunk_ship_size)

        if result == "miss":
            # If we were targeting along an orientation and missed,
            # we may need to reverse direction.
            return

        if result == "hit":
            self._current_hits.append((row, col))
            self._is_targeting = True
            self._update_orientation()
            self._enqueue_neighbours(row, col)

        elif result == "sunk":
            self._current_hits.append((row, col))
            # Ship is done — clear targeting state
            self._finish_targeting()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue_neighbours(self, row: int, col: int) -> None:
        """Add valid, un-attacked cardinal neighbours to the target queue.

        If an orientation is locked, only add cells along that axis.
        """
        directions: list[tuple[int, int]]
        if self._orientation == "horizontal":
            directions = [(0, -1), (0, 1)]
        elif self._orientation == "vertical":
            directions = [(-1, 0), (1, 0)]
        else:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if self._is_valid(nr, nc) and (nr, nc) not in self.shots_taken:
                if (nr, nc) not in self._target_queue:
                    self._target_queue.append((nr, nc))

    def _update_orientation(self) -> None:
        """Determine orientation once we have ≥ 2 hits on the current ship."""
        if self._orientation is not None:
            return
        if len(self._current_hits) < 2:
            return

        r1, c1 = self._current_hits[-2]
        r2, c2 = self._current_hits[-1]
        if r1 == r2:
            self._orientation = "horizontal"
        elif c1 == c2:
            self._orientation = "vertical"

        # Once orientation is known, prune the queue to only keep
        # cells aligned with the orientation.
        if self._orientation is not None:
            self._prune_queue()

    def _prune_queue(self) -> None:
        """Remove cells from the target queue that don't match the
        locked orientation."""
        if self._orientation is None or not self._current_hits:
            return

        hit_rows = {r for r, _ in self._current_hits}
        hit_cols = {c for _, c in self._current_hits}

        pruned: deque[tuple[int, int]] = deque()
        for r, c in self._target_queue:
            if self._orientation == "horizontal" and r in hit_rows:
                pruned.append((r, c))
            elif self._orientation == "vertical" and c in hit_cols:
                pruned.append((r, c))
        self._target_queue = pruned

    def _finish_targeting(self) -> None:
        """Reset targeting state after a ship is sunk."""
        # Mark the cells belonging to the sunk ship so we can
        # identify any "stale" hits left from a different ship.
        sunk_set = set(self._current_hits)

        # Check if there are any un-sunk hits remaining from
        # a previous ship that was hit while targeting this one.
        leftover_hits = [
            h for h in self.hits
            if h not in self.misses and h not in sunk_set
            and self._has_unsunk_neighbours(h)
        ]

        self._current_hits = []
        self._orientation = None
        self._target_queue.clear()

        # If there are leftover un-sunk hits, re-enter target mode
        if leftover_hits:
            self._is_targeting = True
            for h in leftover_hits:
                self._current_hits.append(h)
                self._enqueue_neighbours(h[0], h[1])
        else:
            self._is_targeting = False

    def _has_unsunk_neighbours(self, cell: tuple[int, int]) -> bool:
        """Return True if a hit cell has at least one un-attacked neighbour,
        suggesting the ship it belongs to may not be sunk yet."""
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc) and (nr, nc) not in self.shots_taken:
                return True
        return False
