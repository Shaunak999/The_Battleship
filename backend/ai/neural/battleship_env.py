"""
Gymnasium-compatible Battleship environment.

Models only the **firing** side — the opponent places ships randomly
each episode.  The agent observes its own shot history on a 10×10
grid and picks one of 100 cells per turn.

Observation
-----------
4 × 10 × 10 float32 tensor (channels-first):
    0 – un-attacked mask  (1 = not yet shot, 0 = already shot)
    1 – hit mask          (1 = hit)
    2 – miss mask         (1 = miss)
    3 – sunk mask         (1 = belongs to a sunk ship)

Action space
------------
Discrete(100)  →  cell index 0-99 maps to (row, col) = (a // 10, a % 10)

Reward
------
    miss  :  -0.1
    hit   :  +1.0
    sunk  :  +2.0 × ship_size
    win   :  +5.0  (bonus, added on the final sinking shot)
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

def _place_ships_randomly(
    board_size: int = BOARD_SIZE,
    ships: List[Tuple[str, int]] = SHIP_DEFINITIONS,
) -> Dict[Tuple[int, int], Tuple[str, int]]:
    """Return a mapping of (row, col) → (ship_name, ship_index)."""
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
            if not any(co in occupied for co in coords):
                for co in coords:
                    occupied[co] = (name, idx)
                placed = True
                break
        if not placed:
            return _place_ships_randomly(board_size, ships)
    return occupied


# ── Gymnasium Environment ─────────────────────────────────────────────────

class BattleshipEnv(gym.Env):
    """A Battleship firing environment for Stable-Baselines3 DQN."""

    metadata = {"render_modes": []}

    def __init__(self, board_size: int = BOARD_SIZE):
        super().__init__()
        self.board_size = board_size
        self.n_cells = board_size * board_size

        # Observation: 4 channels × 10 × 10
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(4, board_size, board_size),
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

        self._board = _place_ships_randomly(self.board_size)
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

        # If cell was already shot, penalize and don't count as a turn
        if cell in self._shots:
            return self._get_obs(), -1.0, False, False, {"result": "repeat"}

        self._shots.add(cell)

        if cell in self._board:
            ship_name, ship_idx = self._board[cell]
            self._hits.add(cell)
            self._hits_scored += 1
            self._ship_hp[ship_idx].discard(cell)

            # Check if ship is sunk
            if len(self._ship_hp[ship_idx]) == 0:
                self._sunk_ships.add(ship_idx)
                # Mark all cells of this ship as sunk
                for co, (_, si) in self._board.items():
                    if si == ship_idx:
                        self._sunk_cells.add(co)

                ship_size = next(s for n, s in SHIP_DEFINITIONS if n == ship_name)
                reward = 2.0 * ship_size

                # Check if game is won
                if self._hits_scored >= self._total_ship_cells:
                    reward += 5.0
                    return self._get_obs(), reward, True, False, {
                        "result": "win",
                        "ship": ship_name,
                    }

                return self._get_obs(), reward, False, False, {
                    "result": "sunk",
                    "ship": ship_name,
                }
            else:
                return self._get_obs(), 1.0, False, False, {"result": "hit"}
        else:
            self._misses.add(cell)
            return self._get_obs(), -0.1, False, False, {"result": "miss"}

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
        obs = np.zeros((4, board, board), dtype=np.float32)

        for r in range(board):
            for c in range(board):
                cell = (r, c)
                if cell not in self._shots:
                    obs[0, r, c] = 1.0  # unattacked
                if cell in self._hits:
                    obs[1, r, c] = 1.0
                if cell in self._misses:
                    obs[2, r, c] = 1.0
                if cell in self._sunk_cells:
                    obs[3, r, c] = 1.0

        return obs

    # ── Utility ────────────────────────────────────────────────────────

    def get_ship_map(self) -> Dict[Tuple[int, int], str]:
        """Return the hidden board for evaluation / debugging."""
        return {co: name for co, (name, _) in self._board.items()}
