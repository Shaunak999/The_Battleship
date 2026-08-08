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
        self.remaining_ship_sizes: list[int] = [s for _, s in self.ships]
        self.sunk_ships: list[str] = []

    # ------------------------------------------------------------------
    # Shared helper: random valid ship placement
    # ------------------------------------------------------------------

    def place_ships_randomly(self) -> list[dict]:
        """Generate a complete random valid placement for the AI's own ships.

        Returns
        -------
        list[dict]
            Each dict has keys:
                ``"name"``  — ship name (str)
                ``"size"``  — ship size (int)
                ``"coordinates"`` — list of (row, col) tuples
        """
        occupied: set[tuple[int, int]] = set()
        placements: list[dict] = []

        for name, size in self.ships:
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
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

            if not placed:
                # Extremely unlikely with a 10×10 board — restart entirely
                return self.place_ships_randomly()

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
