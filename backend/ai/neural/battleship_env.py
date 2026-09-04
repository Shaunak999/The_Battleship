"""
Gymnasium-compatible Battleship environment with rich multi-channel features.

Models the **firing** side — the opponent places ships randomly
each episode. The agent observes a 6×10×10 feature tensor combining raw
shot history with combinatorial probability priors and target boundaries.

Observation
-----------
6 × 10 × 10 float32 tensor (channels-first):
    0 – un-attacked mask (1 = not yet shot, 0 = already shot)
    1 – hit mask (1 = hit)
    2 – miss mask (1 = miss)
    3 – sunk mask (1 = belongs to a sunk ship)
    4 – normalized probability density heat-map
    5 – target boundary & parity layer

Action space
------------
Discrete(100) → cell index 0-99 maps to (row, col) = (a // 10, a % 10)

Reward
------
    step penalty : -1.0  (strictly encourages shortest-path to victory)
    hit          : +3.0
    target bonus : +2.0  (hit adjacent to an unsunk hit)
    sunk         : +4.0 × ship_size
    win          : +10.0
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ── Ship definitions (same order / sizes as the rest of the codebase) ──────
SHIP_DEFINITIONS: List[Tuple[str, int]] = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]

BOARD_SIZE = 10
TOTAL_SHIP_CELLS = sum(s for _, s in SHIP_DEFINITIONS)  # 17


# ── Helper: place ships randomly ───────────────────────────────────────────

def _touches(
    coords: List[Tuple[int, int]],
    occupied: Dict[Tuple[int, int], Tuple[str, int]],
    board_size: int,
) -> bool:
    """True if any coordinate is adjacent (incl. diagonally) to an occupied cell."""
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and (nr, nc) in occupied:
                    return True
    return False


def _place_ships_randomly(
    board_size: int = BOARD_SIZE,
    ships: List[Tuple[str, int]] = SHIP_DEFINITIONS,
    touch_probability: float = 0.5,
) -> Dict[Tuple[int, int], Tuple[str, int]]:
    """Return a mapping of (row, col) → (ship_name, ship_index).

    Mixes the two real-world placement styles so training matches what an
    agent actually faces: AI defenders / "Randomize" keep a 1-cell gap,
    while manual human placement only forbids overlap (ships may touch).
    With ``touch_probability`` (default 0.5) the fleet follows the human
    style; otherwise the gapped style.
    """
    allow_touch = random.random() < touch_probability
    occupied: Dict[Tuple[int, int], Tuple[str, int]] = {}
    for idx, (name, size) in enumerate(ships):
        placed = False
        for _ in range(10000):
            orient = random.choice(["h", "v"])
            if orient == "h":
                r = random.randint(0, board_size - 1)
                c = random.randint(0, board_size - size)
                coords = [(r, c + i) for i in range(size)]
            else:
                r = random.randint(0, board_size - size)
                c = random.randint(0, board_size - 1)
                coords = [(r + i, c) for i in range(size)]
            if not any(co in occupied for co in coords) and (
                allow_touch or not _touches(coords, occupied, board_size)
            ):
                for co in coords:
                    occupied[co] = (name, idx)
                placed = True
                break
        if not placed:
            return _place_ships_randomly(board_size, ships, touch_probability)
    return occupied


# ── Gymnasium Environment ─────────────────────────────────────────────────

class BattleshipEnv(gym.Env):
    """A Battleship firing environment with 6-channel state and step-penalized rewards."""

    metadata = {"render_modes": []}

    def __init__(self, board_size: int = BOARD_SIZE, touch_probability: float = 0.5):
        super().__init__()
        self.board_size = board_size
        self.n_cells = board_size * board_size
        self._touch_probability = touch_probability

        # Observation: 6 channels × 10 × 10
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, board_size, board_size),
            dtype=np.float32,
        )

        # Action: cell index 0-99
        self.action_space = spaces.Discrete(self.n_cells)

        # Internal state (set in reset)
        self._board: Dict[Tuple[int, int], Tuple[str, int]] = {}
        self._shots: set = set()
        self._hits: set = set()
        self._misses: set = set()
        self._sunk_cells: set = set()
        self._sunk_ships: set = set()
        self._hits_scored: int = 0
        self._ship_hp: Dict[int, set] = {}
        self._total_ship_cells: int = TOTAL_SHIP_CELLS

    # ── Gym interface ──────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._board = _place_ships_randomly(self.board_size, touch_probability=self._touch_probability)
        self._shots = set()
        self._hits = set()
        self._misses = set()
        self._sunk_cells = set()
        self._sunk_ships = set()
        self._hits_scored = 0

        # Track remaining HP per ship index
        self._ship_hp = {}
        for _, (_, idx) in self._board.items():
            if idx not in self._ship_hp:
                self._ship_hp[idx] = set()
        for co, (_, idx) in self._board.items():
            self._ship_hp[idx].add(co)
        self._total_ship_cells = TOTAL_SHIP_CELLS

        return self._get_obs(), {"board": self._board}

    def step(self, action: int):
        row, col = divmod(action, self.board_size)
        cell = (row, col)

        # If cell was already shot, heavy penalty
        if cell in self._shots:
            return self._get_obs(), -5.0, False, False, {"result": "repeat"}

        # Base step penalty: -1.0 for every turn taken to minimize total shots
        reward = -1.0
        self._shots.add(cell)

        # Check if cell was adjacent to an existing unsunk hit before this shot
        unsunk_hits = self._hits - self._sunk_cells
        was_adjacent_to_hit = any(
            abs(row - hr) + abs(col - hc) == 1 for hr, hc in unsunk_hits
        )

        if cell in self._board:
            ship_name, ship_idx = self._board[cell]
            self._hits.add(cell)
            self._hits_scored += 1
            self._ship_hp[ship_idx].discard(cell)

            reward += 3.0
            if was_adjacent_to_hit:
                reward += 2.0  # Targeted hit bonus

            # Check if ship is sunk
            if len(self._ship_hp[ship_idx]) == 0:
                self._sunk_ships.add(ship_idx)
                for co, (_, si) in self._board.items():
                    if si == ship_idx:
                        self._sunk_cells.add(co)

                ship_size = next(s for n, s in SHIP_DEFINITIONS if n == ship_name)
                reward += 4.0 * ship_size

                # Check if game is won
                if self._hits_scored >= self._total_ship_cells:
                    reward += 10.0
                    return self._get_obs(), reward, True, False, {
                        "result": "win",
                        "ship": ship_name,
                    }

                return self._get_obs(), reward, False, False, {
                    "result": "sunk",
                    "ship": ship_name,
                }
            else:
                return self._get_obs(), reward, False, False, {"result": "hit"}
        else:
            self._misses.add(cell)
            return self._get_obs(), reward, False, False, {"result": "miss"}

    # ── Action masking (for SB3 DQN) ──────────────────────────────────

    def action_masks(self) -> List[bool]:
        """Return a boolean mask: True for valid (un-shot) cells."""
        mask = [True] * self.n_cells
        for r, c in self._shots:
            mask[r * self.board_size + c] = False
        return mask

    # ── Observation builder ────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        board = self.board_size
        obs = np.zeros((6, board, board), dtype=np.float32)

        unsunk_hits = self._hits - self._sunk_cells

        # Surviving ship sizes
        remaining_sizes = [
            size for idx, (_, size) in enumerate(SHIP_DEFINITIONS)
            if idx not in self._sunk_ships
        ]

        min_size = min(remaining_sizes) if remaining_sizes else 2

        # 0: Unattacked, 1: Hits, 2: Misses, 3: Sunk
        for r in range(board):
            for c in range(board):
                cell = (r, c)
                if cell not in self._shots:
                    obs[0, r, c] = 1.0
                if cell in self._hits:
                    obs[1, r, c] = 1.0
                if cell in self._misses:
                    obs[2, r, c] = 1.0
                if cell in self._sunk_cells:
                    obs[3, r, c] = 1.0

        # 4: Probability Heatmap Layer
        heatmap = np.zeros((board, board), dtype=np.float32)
        invalid_mask = self._misses | self._sunk_cells

        for size in remaining_sizes:
            # Horizontal placements
            for r in range(board):
                for c in range(board - size + 1):
                    coords = [(r, c + i) for i in range(size)]
                    if not any(co in invalid_mask for co in coords):
                        for cr, cc in coords:
                            heatmap[cr, cc] += 1.0

            # Vertical placements
            for r in range(board - size + 1):
                for c in range(board):
                    coords = [(r + i, c) for i in range(size)]
                    if not any(co in invalid_mask for co in coords):
                        for cr, cc in coords:
                            heatmap[cr, cc] += 1.0

        # Zero out shot cells in heatmap and normalize
        for r, c in self._shots:
            heatmap[r, c] = 0.0
        max_h = heatmap.max()
        if max_h > 0:
            obs[4] = heatmap / max_h

        # 5: Target boundary & parity mask
        if unsunk_hits:
            for r, c in unsunk_hits:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board and 0 <= nc < board and (nr, nc) not in self._shots:
                        obs[5, nr, nc] = 1.0
        else:
            # Parity hunting pattern for smallest remaining ship
            for r in range(board):
                for c in range(board):
                    if (r + c) % min_size == 0 and (r, c) not in self._shots:
                        obs[5, r, c] = 1.0

        return obs

    # ── Utility ────────────────────────────────────────────────────────

    def get_ship_map(self) -> Dict[Tuple[int, int], str]:
        """Return the hidden board for evaluation / debugging."""
        return {co: name for co, (name, _) in self._board.items()}

