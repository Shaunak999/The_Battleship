"""
Probability-density AI strategy for Battleship.

The most sophisticated of the three strategies.  For every un-attacked cell
it calculates how many *valid* ship placements (across all remaining ships)
pass through that cell.  The cell with the highest count is the best shot.

When hits exist that haven't been resolved to a sunk ship, the AI boosts
cells adjacent to those hits so that it aggressively finishes off wounded
ships before hunting new ones.

Expected average game length: ~42-50 shots (far better than Random ~96
or Hunt & Target ~62).
"""

import random

import numpy as np
from typing import Optional

from .base_ai import BaseAI


class ProbabilityAI(BaseAI):
    """Probability-density map AI.

    Algorithm
    ---------
    On every turn the AI builds a **probability heat-map** over the 10×10
    board.  For each surviving ship size *s*:

    1.  Try every possible horizontal placement of length *s*.
    2.  Try every possible vertical placement of length *s*.
    3.  Reject any placement that covers a known **miss** cell.
    4.  Reject any placement that covers an already-**sunk** cell.
    5.  For the remaining valid placements, increment the heat-map
        value of every cell in that placement by 1.

    The result is a 10×10 matrix where each cell's value is proportional
    to the probability that it conceals part of a surviving ship.

    **Target-mode boost**: any un-sunk hit cell gets its neighbours'
    probability values amplified, so the AI prioritises finishing off
    ships it has already wounded.

    The AI fires at the highest-value cell.
    """

    def __init__(self, board_size: int = 10):
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Probability"

    def reset(self) -> None:
        super().reset()
        # Track which hit cells have been accounted for by a sunk ship
        self._unsunk_hits: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # choose_move
    # ------------------------------------------------------------------

    def choose_move(self) -> tuple[int, int]:
        """Build the probability heat-map and return the highest-value cell."""

        heat = self._build_heatmap()

        # Zero out cells we've already shot at
        for r, c in self.shots_taken:
            heat[r, c] = 0

        # Find the cell with the maximum probability
        max_val = heat.max()

        if max_val == 0:
            # Shouldn't happen in a normal game — fallback
            available = self._available_cells()

            if not available:
                raise RuntimeError("ProbabilityAI: no cells left")

            return random.choice(available)

        # Convert NumPy int64 values to normal Python ints
        candidates = [
            (int(row), int(col))
            for row, col in zip(*np.where(heat == max_val))
        ]

        return random.choice(candidates)

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
        """Update internal state after a shot."""
        self._record_shot(row, col, result, sunk_ship_name, sunk_ship_size)

        if result == "hit":
            self._unsunk_hits.add((row, col))

        elif result == "sunk":
            # The sunk cell itself was also a hit
            self._unsunk_hits.add((row, col))
            # Now figure out which hits belong to this sunk ship and remove them.
            self._resolve_sunk_ship(row, col, sunk_ship_size)

    # ------------------------------------------------------------------
    # Heat-map construction
    # ------------------------------------------------------------------

    def _build_heatmap(self) -> np.ndarray:
        """Return a 10×10 float array of probability weights."""
        heat = np.zeros((self.board_size, self.board_size), dtype=float)

        for size in self.remaining_ship_sizes:
            self._add_ship_placements(heat, size)

        # --- Target-mode boost ---
        # If there are un-sunk hits, heavily boost their neighbours so
        # the AI finishes wounded ships first.
        if self._unsunk_hits:
            boost = np.zeros_like(heat)
            for r, c in self._unsunk_hits:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        self._is_valid(nr, nc)
                        and (nr, nc) not in self.shots_taken
                    ):
                        boost[nr, nc] += 1

            # The boost factor is large enough to dominate the base probability
            # so that the AI always prioritises finishing ships.
            max_heat = heat.max() if heat.max() > 0 else 1
            heat += boost * max_heat * 10

        return heat

    def _add_ship_placements(self, heat: np.ndarray, size: int) -> None:
        """For a single ship size, add valid placement counts to *heat*.

        A placement is valid if none of its cells are:
        - a known miss
        - a known sunk-ship cell (already accounted for)

        A placement IS still valid if it passes through an un-sunk hit,
        because that hit might belong to this very ship.
        """
        # Cells we know are blocked for new placements
        blocked = self.misses | (self.hits - self._unsunk_hits)

        # --- Horizontal placements ---
        for r in range(self.board_size):
            for c in range(self.board_size - size + 1):
                cells = [(r, c + i) for i in range(size)]
                if not any(cell in blocked for cell in cells):
                    # Bonus: if the placement overlaps an unsunk hit, weight it higher
                    overlap = sum(1 for cell in cells if cell in self._unsunk_hits)
                    weight = 1 + overlap * 5  # amplify placements consistent with hits
                    for cell in cells:
                        if cell not in self.shots_taken:
                            heat[cell[0], cell[1]] += weight

        # --- Vertical placements ---
        for r in range(self.board_size - size + 1):
            for c in range(self.board_size):
                cells = [(r + i, c) for i in range(size)]
                if not any(cell in blocked for cell in cells):
                    overlap = sum(1 for cell in cells if cell in self._unsunk_hits)
                    weight = 1 + overlap * 5
                    for cell in cells:
                        if cell not in self.shots_taken:
                            heat[cell[0], cell[1]] += weight

    # ------------------------------------------------------------------
    # Sunk-ship resolution
    # ------------------------------------------------------------------

    def _resolve_sunk_ship(
        self, last_row: int, last_col: int, ship_size: Optional[int]
    ) -> None:
        """When a ship is sunk, figure out which hit cells belonged to it
        and remove them from ``_unsunk_hits``.

        We use a simple flood-fill on contiguous unsunk hits touching the
        final sinking cell.  If *ship_size* is provided we stop once we
        have exactly that many cells.
        """
        if ship_size is None:
            # Without size info, just clear all unsunk hits (less precise)
            self._unsunk_hits.clear()
            return

        # BFS / flood-fill from (last_row, last_col) over unsunk hits
        from collections import deque

        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()
        queue.append((last_row, last_col))
        found: list[tuple[int, int]] = []

        while queue and len(found) < ship_size:
            cell = queue.popleft()
            if cell in visited:
                continue
            visited.add(cell)
            if cell in self._unsunk_hits:
                found.append(cell)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cell[0] + dr, cell[1] + dc
                    if (nr, nc) not in visited and (nr, nc) in self._unsunk_hits:
                        queue.append((nr, nc))

        for cell in found:
            self._unsunk_hits.discard(cell)

    # ------------------------------------------------------------------
    # Debugging / analysis helpers
    # ------------------------------------------------------------------

    def get_heatmap(self) -> list[list[float]]:
        """Return the current probability heat-map as a 2D Python list.

        Useful for the frontend to visualise what the AI is "thinking".
        """
        heat = self._build_heatmap()
        # Zero out already-shot cells
        for r, c in self.shots_taken:
            heat[r, c] = 0
        return heat.tolist()
