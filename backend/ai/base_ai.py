"""
Base AI class for Battleship AI strategies.

All AI strategies inherit from BaseAI and implement the same interface.
The AI module is fully independent of the game/ module — it uses only
standard Python types (tuples of ints, strings, lists) so that it can be
developed and tested separately.

Interface contract with the game engine / API layer
----------------------------------------------------
Coordinates are (row, col) tuples where row ∈ [0, 9] and col ∈ [0, 9].
    Row 0 = "A", Row 9 = "J"   (letter rows)
    Col 0 = "1", Col 9 = "10"  (number columns)

Attack results are one of three strings:
    "miss"  — shot hit water
    "hit"   — shot hit a ship but did not sink it
    "sunk"  — shot hit the last remaining cell of a ship

When a ship is sunk the caller may optionally pass the ship name and size
via process_result() so the AI can update its remaining-ships list.
"""

from abc import ABC, abstractmethod
import random
from typing import Optional


# Standard Battleship ship definitions
SHIP_DEFINITIONS = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]

BOARD_SIZE = 10


class BaseAI(ABC):
    """Abstract base class that every Battleship AI strategy must extend.

    Attributes
    ----------
    board_size : int
        Side length of the square board (default 10).
    ships : list[tuple[str, int]]
        Ship definitions as (name, size) pairs.
    shots_taken : set[tuple[int, int]]
        All cells the AI has already fired at.
    hits : set[tuple[int, int]]
        Cells where the AI scored a hit.
    misses : set[tuple[int, int]]
        Cells where the AI missed.
    remaining_ship_sizes : list[int]
        Sizes of opponent ships that have NOT been sunk yet.
    sunk_ships : list[str]
        Names of opponent ships that have been sunk.
    """

    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        self.ships = list(SHIP_DEFINITIONS)
        self.reset()

    # ------------------------------------------------------------------
    # Public interface — must be implemented by every strategy
    # ------------------------------------------------------------------

    @abstractmethod
    def choose_move(self) -> tuple[int, int]:
        """Return the next (row, col) cell to attack.

        The returned cell MUST NOT be in ``self.shots_taken``.

        Returns
        -------
        tuple[int, int]
            (row, col) with 0-based indexing.
        """
        ...

    @abstractmethod
    def process_result(
        self,
        row: int,
        col: int,
        result: str,
        sunk_ship_name: Optional[str] = None,
        sunk_ship_size: Optional[int] = None,
    ) -> None:
        """Inform the AI of the outcome of its last shot.

        Parameters
        ----------
        row, col : int
            The cell that was attacked.
        result : str
            One of ``"miss"``, ``"hit"``, or ``"sunk"``.
        sunk_ship_name : str or None
            If *result* is ``"sunk"``, the name of the ship that was sunk.
        sunk_ship_size : int or None
            If *result* is ``"sunk"``, the size of the ship that was sunk.
        """
        ...

    def reset(self) -> None:
        """Reset the AI to a clean state for a new game.

        Subclasses that add extra state **must** call ``super().reset()``
        and then reinitialise their own attributes.
        """
        self.shots_taken: set[tuple[int, int]] = set()
        self.hits: set[tuple[int, int]] = set()
        self.misses: set[tuple[int, int]] = set()
        self.sunk_perimeter: set[tuple[int, int]] = set()
        self.remaining_ship_sizes: list[int] = [s for _, s in self.ships]
        self.sunk_ships: list[str] = []

    def _mark_sunk_perimeter(self, sunk_cells: list[tuple[int, int]]) -> None:
        """Mark surrounding 1-cell perimeter of a sunk ship as unusable dead water."""
        for r, c in sunk_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        if (nr, nc) not in self.hits:
                            self.sunk_perimeter.add((nr, nc))

    @property
    def blocked_cells(self) -> set[tuple[int, int]]:
        """Cells that cannot contain ships (shots taken or sunk ship perimeters)."""
        return self.shots_taken | self.sunk_perimeter

    # ------------------------------------------------------------------
    # Shared helper: random valid ship placement
    # ------------------------------------------------------------------

    def place_ships_randomly(self) -> list[dict]:
        """Generate a smart, non-clustered placement for the AI's own ships.

        Prefers placing ships with at least a 1-cell buffer spacing between them
        so ships do not touch or form easy clusters for opponents to sweep.
        """
        # Try up to 100 times to place all ships with a 1-cell buffer zone
        for attempt in range(100):
            occupied: set[tuple[int, int]] = set()
            buffer_zone: set[tuple[int, int]] = set()
            placements: list[dict] = []
            success = True

            for name, size in self.ships:
                placed = False
                for _ in range(200):
                    orientation = random.choice(["horizontal", "vertical"])
                    if orientation == "horizontal":
                        row = random.randint(0, self.board_size - 1)
                        col = random.randint(0, self.board_size - size)
                        coords = [(row, col + i) for i in range(size)]
                    else:
                        row = random.randint(0, self.board_size - size)
                        col = random.randint(0, self.board_size - 1)
                        coords = [(row + i, col) for i in range(size)]

                    # Check that no cell overlaps existing ships or their 1-cell surrounding buffer
                    if not any(c in occupied or c in buffer_zone for c in coords):
                        for c in coords:
                            occupied.add(c)
                            # Mark surrounding 1-cell neighborhood as buffer
                            for dr in (-1, 0, 1):
                                for dc in (-1, 0, 1):
                                    nr, nc = c[0] + dr, c[1] + dc
                                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                        buffer_zone.add((nr, nc))

                        placements.append({
                            "name": name,
                            "size": size,
                            "coordinates": coords,
                        })
                        placed = True
                        break

                if not placed:
                    success = False
                    break

            if success:
                return placements

        # Fallback: standard non-overlapping placement if buffer constraint fails
        occupied: set[tuple[int, int]] = set()
        placements: list[dict] = []
        for name, size in self.ships:
            placed = False
            while not placed:
                orientation = random.choice(["horizontal", "vertical"])
                if orientation == "horizontal":
                    row = random.randint(0, self.board_size - 1)
                    col = random.randint(0, self.board_size - size)
                    coords = [(row, col + i) for i in range(size)]
                else:
                    row = random.randint(0, self.board_size - size)
                    col = random.randint(0, self.board_size - 1)
                    coords = [(row + i, col) for i in range(size)]

                if not any(c in occupied for c in coords):
                    for c in coords:
                        occupied.add(c)
                    placements.append({
                        "name": name,
                        "size": size,
                        "coordinates": coords,
                    })
                    placed = True

        return placements

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _all_cells(self) -> list[tuple[int, int]]:
        """Return every cell on the board as a list of (row, col) tuples."""
        return [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
        ]

    def _available_cells(self) -> list[tuple[int, int]]:
        """Return cells that have NOT been attacked yet."""
        return [cell for cell in self._all_cells() if cell not in self.shots_taken]

    def _is_valid(self, row: int, col: int) -> bool:
        """Check whether (row, col) is inside the board."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _record_shot(self, row: int, col: int, result: str,
                     sunk_ship_name: Optional[str] = None,
                     sunk_ship_size: Optional[int] = None) -> None:
        """Common bookkeeping after a shot — call from subclass process_result."""
        self.shots_taken.add((row, col))
        if result in ("hit", "sunk"):
            self.hits.add((row, col))
        else:
            self.misses.add((row, col))

        if result == "sunk" and sunk_ship_size is not None:
            if sunk_ship_size in self.remaining_ship_sizes:
                self.remaining_ship_sizes.remove(sunk_ship_size)
            if sunk_ship_name:
                self.sunk_ships.append(sunk_ship_name)

    @property
    def name(self) -> str:
        """Human-readable strategy name (override in subclass if desired)."""
        return self.__class__.__name__


def axis_extension_cells(
    hits: set[tuple[int, int]],
    shots_taken: set[tuple[int, int]],
    board_size: int,
) -> set[tuple[int, int]]:
    """Cells just beyond straight runs of >=2 collinear un-sunk hits.

    Once two adjacent hits reveal a ship's orientation (same row -> horizontal,
    same column -> vertical), the only cells that can continue that ship are the
    ones directly beyond the ends of the run along its axis — firing anywhere
    else (e.g. perpendicular) wastes a turn. Returns those cells that are still
    inside the board and have not been shot at yet. When no run has >=2 hits
    (only isolated single hits), the orientation is unknown and the set is empty.
    """
    exts: set[tuple[int, int]] = set()

    # Horizontal runs: group hit columns per row, split on gaps, and for every
    # run of >=2 consecutive cells keep the cells beyond both ends.
    by_row: dict[int, list[int]] = {}
    for r, c in hits:
        by_row.setdefault(r, []).append(c)
    for r, cols in by_row.items():
        cols = sorted(cols)
        run = [cols[0]]
        for c in cols[1:]:
            if c == run[-1] + 1:
                run.append(c)
            else:
                if len(run) >= 2:
                    exts.add((r, run[0] - 1))
                    exts.add((r, run[-1] + 1))
                run = [c]
        if len(run) >= 2:
            exts.add((r, run[0] - 1))
            exts.add((r, run[-1] + 1))

    # Vertical runs: same logic per column.
    by_col: dict[int, list[int]] = {}
    for r, c in hits:
        by_col.setdefault(c, []).append(r)
    for c, rows in by_col.items():
        rows = sorted(rows)
        run = [rows[0]]
        for r in rows[1:]:
            if r == run[-1] + 1:
                run.append(r)
            else:
                if len(run) >= 2:
                    exts.add((run[0] - 1, c))
                    exts.add((run[-1] + 1, c))
                run = [r]
        if len(run) >= 2:
            exts.add((run[0] - 1, c))
            exts.add((run[-1] + 1, c))

    return {
        (r, c)
        for r, c in exts
        if 0 <= r < board_size and 0 <= c < board_size and (r, c) not in shots_taken
    }
