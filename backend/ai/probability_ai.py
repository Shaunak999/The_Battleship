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

from .base_ai import BaseAI, axis_extension_cells


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

    The AI fires at the best cell, de-predicted so it does not replay the
    same opening every game: hunt-mode shots add tiny random jitter, and the
    very first shot is drawn from a band around the peak instead of always
    picking the peak cell. Both knobs are configurable in the constructor and
    tuned so variety costs nothing in average shots-to-win.
    """

    def __init__(
        self,
        board_size: int = 10,
        hunt_jitter: float = 1.5,
        opening_band: float = 2.0,
    ):
        """Probability-density AI.

        Parameters
        ----------
        board_size : int
            Side length of the square board.
        hunt_jitter : float
            Max random noise (0..hunt_jitter) added to every cell in hunt mode.
            Large enough to let near-ties resolve differently each game, small
            enough not to distort the real probability ordering.
        opening_band : float
            On the very first shot, pick randomly among cells whose value is
            within *opening_band* of the peak instead of the peak itself. On
            an empty board the top cells are near-equivalent, so this widens
            the opening at no accuracy cost.
        """
        # Stored before super().__init__() because BaseAI's constructor calls
        # reset(), which needs to know the knobs.
        self._hunt_jitter = hunt_jitter
        self._opening_band = opening_band
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Probability"

    def reset(self) -> None:
        super().reset()
        # Track which hit cells have been accounted for by a sunk ship
        self._unsunk_hits: set[tuple[int, int]] = set()
        self._first_move = True

    # ------------------------------------------------------------------
    # choose_move
    # ------------------------------------------------------------------

    def choose_move(self) -> tuple[int, int]:
        """Build the probability heat-map and return the highest-value cell."""

        heat = self._build_heatmap()

        # Hunt mode (no active unsunk hits): add a little random jitter so
        # near-ties between cells resolve differently each game. The jitter is
        # tiny relative to real heat-map differences, so it does not degrade
        # shot quality — it only stops the AI replaying the same search.
        if not self._unsunk_hits:
            heat = heat + np.random.uniform(0.0, self._hunt_jitter, size=heat.shape)

        # Zero out cells we've already shot at or marked as sunk perimeters
        for r, c in self.blocked_cells:
            heat[r, c] = 0

        # Find the best remaining cell
        max_val = heat.max()

        if max_val == 0:
            # Shouldn't happen in a normal game — fallback
            available = [c for c in self._available_cells() if c not in self.sunk_perimeter]
            if not available:
                available = self._available_cells()

            if not available:
                raise RuntimeError("ProbabilityAI: no cells left")

            return random.choice(available)

        if self._unsunk_hits:
            # Target mode — a ship is wounded, so always finish it first:
            # fire at the (boosted) peak neighbour, strict maximum.
            candidates = [
                (int(row), int(col))
                for row, col in zip(*np.where(heat == max_val))
            ]
            return random.choice(candidates)

        if self._first_move:
            # Opening shot: pick randomly among every cell within *opening_band*
            # of the peak instead of demanding the single maximum. On an empty
            # board the centre cells are near-equivalent, so this widens the
            # opening from a handful of cells to many — at no accuracy cost.
            self._first_move = False
            flat = heat.ravel()
            band_idxs = np.flatnonzero(flat >= max_val - self._opening_band)
            i = band_idxs[np.random.randint(len(band_idxs))]
            return (int(i // self.board_size), int(i % self.board_size))

        # Hunt mode, past the opening: strict maximum of the jittered map
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
            blocked = self.blocked_cells
            for r, c in self._unsunk_hits:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        self._is_valid(nr, nc)
                        and (nr, nc) not in blocked
                    ):
                        boost[nr, nc] += 1

            # Orientation-aware boost: a straight run of >=2 collinear hits
            # reveals the ship's axis, so the only cells that can continue it
            # are the ones just beyond the run along that axis. Give them a
            # bigger boost than every pure neighbour probe so the AI never
            # wastes a shot perpendicular to a ship it is already lined up on.
            # (Perpendicular probes are still used for isolated single hits,
            # whose orientation is not yet known.)
            for r, c in axis_extension_cells(
                self._unsunk_hits, self.shots_taken, self.board_size
            ):
                boost[r, c] += 6

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
        # Cells we know are blocked for new placements. Sunk cells cannot hold
        # a new ship, but the ring around a sunk ship can (ships may touch), so
        # it is intentionally NOT treated as blocked.
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
                        if cell not in self.blocked_cells:
                            heat[cell[0], cell[1]] += weight

        # --- Vertical placements ---
        for r in range(self.board_size - size + 1):
            for c in range(self.board_size):
                cells = [(r + i, c) for i in range(size)]
                if not any(cell in blocked for cell in cells):
                    overlap = sum(1 for cell in cells if cell in self._unsunk_hits)
                    weight = 1 + overlap * 5
                    for cell in cells:
                        if cell not in self.blocked_cells:
                            heat[cell[0], cell[1]] += weight

    # ------------------------------------------------------------------
    # Sunk-ship resolution
    # ------------------------------------------------------------------

    def _resolve_sunk_ship(
        self, last_row: int, last_col: int, ship_size: Optional[int]
    ) -> None:
        """When a ship is sunk, figure out which hit cells belonged to it
        and remove them from ``_unsunk_hits``.
        """
        if ship_size is None:
            self._unsunk_hits.discard((last_row, last_col))
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
        # Note: we deliberately do NOT mark the sunk ship's surrounding ring as
        # guaranteed water. Real opponents (human players) are allowed to place
        # ships touching each other, so a ring cell may still hide part of
        # another ship. The probability heat-map already excludes the sunk
        # ship's own cells, which is all that is safely known.

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
