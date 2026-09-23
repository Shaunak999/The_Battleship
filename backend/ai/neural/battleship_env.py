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

Reward (gapped-board optimized)
------
    step penalty      : -1.0   (encourages shortest-path to victory)
    repeat shot       : -10.0  (strong deterrent against re-shooting)
    hit (base)        : +3.0   (net +2.0 after step penalty)
    target adjacency  : +2.0   (hit orthogonally adjacent to unsunk hit)
    axis extension    : +5.0   (hit continuing a collinear line of ≥2)
    sunk              : +2.0 × ship_size  (scaled: carrier=+10, destroyer=+4)
    win               : +15.0
    sunk-adjacent miss: -2.0   (guaranteed waste on gapped boards)
    perpendicular miss: -1.5   (off-axis when ship orientation is known)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from ai.base_ai import axis_extension_cells
except ImportError:
    from backend.ai.base_ai import axis_extension_cells

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
    touch_probability: float = 0.0,
) -> Dict[Tuple[int, int], Tuple[str, int]]:
    """Return a mapping of (row, col) -> (ship_name, ship_index).

    ``touch_probability`` controls the placement style for the whole board:

    * 0.0 (default) - ships never touch, not even diagonally. This is the
      gapped style the Randomize button / AI defenders produce and the style
      the agent is trained against.
    * 1.0 - only overlap is forbidden, so ships may sit side by side. This is
      what manual human placement in the UI can produce.

    Values in between pick one style per board.
    """
    occupied: Dict[Tuple[int, int], Tuple[str, int]] = {}
    allow_touch = random.random() < touch_probability

    for idx, (name, size) in enumerate(ships):
        placed = False
        for _ in range(5000):
            orient = random.choice(["h", "v"])
            if orient == "h":
                r = random.randint(0, board_size - 1)
                c = random.randint(0, board_size - size)
                coords = [(r, c + i) for i in range(size)]
            else:
                r = random.randint(0, board_size - size)
                c = random.randint(0, board_size - 1)
                coords = [(r + i, c) for i in range(size)]
            if any(co in occupied for co in coords):
                continue
            if not allow_touch and _touches(coords, occupied, board_size):
                continue
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

    def __init__(self, board_size: int = BOARD_SIZE, touch_probability: float = 0.0):
        """``touch_probability`` selects the ship-placement style (see
        :func:`_place_ships_randomly`). The default keeps ships apart, matching
        the trainer's evaluation and the board style the agent is measured on.
        """
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

        # Ship placement draws from the stdlib RNG, so propagate an explicit
        # seed to keep episodes reproducible when debugging.
        if seed is not None:
            random.seed(seed)

        self._board = _place_ships_randomly(self.board_size, touch_probability=self._touch_probability)
        self._shots = set()
        self._hits = set()
        self._misses = set()
        self._sunk_cells = set()
        self._sunk_ships = set()
        self._hits_scored = 0
        self._parity_phase = random.randint(0, 1)

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
            return self._get_obs(), -10.0, False, False, {"result": "repeat"}

        # Base step penalty (strictly encourages shortest-path to victory)
        reward = -1.0
        self._shots.add(cell)

        unsunk_hits = self._hits - self._sunk_cells

        # Check if shot was on the collinear axis extension of unsunk hits
        is_axis_extension = False
        if len(unsunk_hits) >= 2:
            r_coords = [r for r, c in unsunk_hits]
            c_coords = [c for r, c in unsunk_hits]
            if len(set(r_coords)) == 1:  # horizontal line
                r_fixed = r_coords[0]
                min_c, max_c = min(c_coords), max(c_coords)
                is_axis_extension = (row == r_fixed and (col == min_c - 1 or col == max_c + 1))
            elif len(set(c_coords)) == 1:  # vertical line
                c_fixed = c_coords[0]
                min_r, max_r = min(r_coords), max(r_coords)
                is_axis_extension = (col == c_fixed and (row == min_r - 1 or row == max_r + 1))

        # Check if shot is orthogonally adjacent to an unsunk hit
        is_adjacent_to_hit = any(
            (row + dr, col + dc) in unsunk_hits
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )

        if cell in self._board:
            # ── HIT ────────────────────────────────────────────────────
            ship_name, ship_idx = self._board[cell]
            self._hits.add(cell)
            self._hits_scored += 1
            self._ship_hp[ship_idx].discard(cell)

            # Base hit reward
            reward += 3.0  # net +2.0

            # Target adjacency: probing next to a wounded ship is smart
            if is_adjacent_to_hit:
                reward += 2.0

            # Axis extension: continuing a known line is optimal play
            if is_axis_extension:
                reward += 5.0

            # Check if ship is sunk
            if len(self._ship_hp[ship_idx]) == 0:
                self._sunk_ships.add(ship_idx)
                for co, (_, si) in self._board.items():
                    if si == ship_idx:
                        self._sunk_cells.add(co)

                ship_size = next(s for n, s in SHIP_DEFINITIONS if n == ship_name)
                # Scaled sunk reward: larger ships → higher payoff
                reward += 2.0 * ship_size

                # Check if game is won
                if self._hits_scored >= self._total_ship_cells:
                    reward += 15.0
                    return self._get_obs(), reward, True, False, {
                        "result": "win",
                        "ship": ship_name,
                        "ship_size": ship_size,
                    }

                return self._get_obs(), reward, False, False, {
                    "result": "sunk",
                    "ship": ship_name,
                    "ship_size": ship_size,
                }
            else:
                return self._get_obs(), reward, False, False, {"result": "hit"}
        else:
            # ── MISS ───────────────────────────────────────────────────
            self._misses.add(cell)

            # Gapped-board penalty: shooting 8-connected to a sunk ship
            # is a guaranteed waste (ships never touch on gapped boards)
            is_sunk_adjacent = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if (nr, nc) in self._sunk_cells:
                        is_sunk_adjacent = True
                        break
                if is_sunk_adjacent:
                    break
            if is_sunk_adjacent:
                reward -= 2.0  # Wasted shot near sunk ship

            # Perpendicular miss: when >=2 collinear hits reveal axis,
            # shooting off-axis is suboptimal
            if len(unsunk_hits) >= 2 and not is_axis_extension:
                reward -= 1.5

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

        # 4: Probability Heatmap Layer (Target-Conditioned)
        heatmap = np.zeros((board, board), dtype=np.float32)
        invalid_mask = self._misses | self._sunk_cells

        # Sunk-ship adjacency exclusion: cells adjacent (8-connected) to
        # sunk ships are virtually always empty on gapped boards. Strong
        # impossible, so the agent never wastes a shot there.
        sunk_adjacent = set()
        for sr, sc in self._sunk_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < board and 0 <= nc < board:
                        if (nr, nc) not in self._shots and (nr, nc) not in self._sunk_cells:
                            sunk_adjacent.add((nr, nc))

        for size in remaining_sizes:
            # Horizontal placements
            for r in range(board):
                for c in range(board - size + 1):
                    coords = [(r, c + i) for i in range(size)]
                    if not any(co in invalid_mask for co in coords):
                        overlap = sum(1 for co in coords if co in unsunk_hits)
                        weight = 1.0 + (overlap * 5.0 if unsunk_hits else 0.0)
                        # Discount placements that touch sunk ships (near-zero on gapped boards)
                        if any(co in sunk_adjacent for co in coords):
                            # Ships never touch on gapped boards, so a placement
                            # that touches a sunk ship is impossible, not unlikely.
                            continue
                        for cr, cc in coords:
                            heatmap[cr, cc] += weight

            # Vertical placements
            for r in range(board - size + 1):
                for c in range(board):
                    coords = [(r + i, c) for i in range(size)]
                    if not any(co in invalid_mask for co in coords):
                        overlap = sum(1 for co in coords if co in unsunk_hits)
                        weight = 1.0 + (overlap * 5.0 if unsunk_hits else 0.0)
                        if any(co in sunk_adjacent for co in coords):
                            # Ships never touch on gapped boards, so a placement
                            # that touches a sunk ship is impossible, not unlikely.
                            continue
                        for cr, cc in coords:
                            heatmap[cr, cc] += weight

        # Zero out shot cells in heatmap and normalize
        for r, c in self._shots:
            heatmap[r, c] = 0.0
        max_h = heatmap.max()
        if max_h > 0:
            obs[4] = heatmap / max_h

        # 5: Target boundary & dynamic parity mask
        if unsunk_hits:
            exts = axis_extension_cells(unsunk_hits, self._shots, board)
            if exts:
                for nr, nc in exts:
                    obs[5, nr, nc] = 1.0
            else:
                # Single hit: probe orthogonal neighbors, but prune directions where
                # remaining open corridor is shorter than the smallest surviving ship.
                blocking = self._misses | self._sunk_cells
                for r, c in unsunk_hits:
                    c_left = c
                    while c_left > 0 and (r, c_left - 1) not in blocking:
                        c_left -= 1
                    c_right = c
                    while c_right < board - 1 and (r, c_right + 1) not in blocking:
                        c_right += 1
                    h_span = c_right - c_left + 1

                    r_up = r
                    while r_up > 0 and (r_up - 1, c) not in blocking:
                        r_up -= 1
                    r_down = r
                    while r_down < board - 1 and (r_down + 1, c) not in blocking:
                        r_down += 1
                    v_span = r_down - r_up + 1

                    if h_span >= min_size:
                        for dc in (-1, 1):
                            nc = c + dc
                            if 0 <= nc < board and (r, nc) not in self._shots:
                                obs[5, r, nc] = 1.0 + (0.2 if h_span > v_span else 0.0)
                    if v_span >= min_size:
                        for dr in (-1, 1):
                            nr = r + dr
                            if 0 <= nr < board and (nr, c) not in self._shots:
                                obs[5, nr, c] = 1.0 + (0.2 if v_span > h_span else 0.0)
        else:
            # Dynamic parity modulus based on smallest surviving ship
            # Also prune dead cells (where no remaining ship can fit)
            phase = getattr(self, "_parity_phase", 0)
            for r in range(board):
                for c in range(board):
                    if (r, c) in self._shots:
                        continue
                    if (r + c + phase) % min_size != 0:
                        continue
                    # Dead-cell check: both h-span and v-span < min_size
                    blocking = self._misses | self._sunk_cells
                    # Horizontal span through (r, c)
                    hs = c
                    while hs > 0 and (r, hs - 1) not in blocking:
                        hs -= 1
                    he = c
                    while he < board - 1 and (r, he + 1) not in blocking:
                        he += 1
                    if (he - hs + 1) >= min_size:
                        obs[5, r, c] = 1.0
                        continue
                    # Vertical span through (r, c)
                    vs = r
                    while vs > 0 and (vs - 1, c) not in blocking:
                        vs -= 1
                    ve = r
                    while ve < board - 1 and (ve + 1, c) not in blocking:
                        ve += 1
                    if (ve - vs + 1) >= min_size:
                        obs[5, r, c] = 1.0

        return obs

    # ── Utility ────────────────────────────────────────────────────────

    def get_ship_map(self) -> Dict[Tuple[int, int], str]:
        """Return the hidden board for evaluation / debugging."""
        return {co: name for co, (name, _) in self._board.items()}

