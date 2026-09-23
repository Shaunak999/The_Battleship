
"""
Neural AI strategy for Battleship.

Runs an optimized PyTorch Q-Network / Dueling Q-Network with rich 6-channel
spatial inputs (raw shots + probability heatmap prior + target boundary & parity),
providing fast inference with strict action masking.

Usage
-----
    ai = NeuralAgent()                        # loads default model path
    ai = NeuralAgent(model_path="path/to/model.zip")
    move = ai.choose_move()
    ai.process_result(row, col, "hit")
"""

from __future__ import annotations

import io
import os
import random
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.dqn.policies import QNetwork

from .base_ai import BaseAI, BOARD_SIZE, SHIP_DEFINITIONS, axis_extension_cells
from .neural.features import BattleshipCNN, SpatialBattleshipCNN


# Default path relative to the backend/ directory
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "neural", "battleship_dqn"
)

# Global model cache to avoid re-reading and re-allocating weights per instance
_MODEL_CACHE: Dict[str, nn.Module] = {}


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network: separates state Value V(s) and action Advantage A(s, a)."""

    def __init__(self, in_channels: int = 6, board_size: int = BOARD_SIZE):
        super().__init__()
        self.board_size = board_size
        self.n_actions = board_size * board_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        flatten_dim = 128 * board_size * board_size

        self.value_stream = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        feat = self.features(observations)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        return val + (adv - adv.mean(dim=-1, keepdim=True))


def load_dqn_qnetwork(model_path: str, board_size: int = BOARD_SIZE) -> Optional[nn.Module]:
    """Load and cache the PyTorch QNetwork / DuelingQNetwork weights from disk."""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    best_candidate = base + "_best.zip"
    normal_candidate = base + ".zip"

    if os.path.exists(best_candidate):
        zip_path = best_candidate
    elif os.path.exists(normal_candidate):
        zip_path = normal_candidate
    else:
        zip_path = normal_candidate

    if zip_path in _MODEL_CACHE:
        return _MODEL_CACHE[zip_path]

    if not os.path.exists(zip_path):
        _MODEL_CACHE[zip_path] = None
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            pth = io.BytesIO(z.read("policy.pth"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(pth, map_location=device)

        # Check if saved model uses SpatialBattleshipCNN or legacy BattleshipCNN
        is_spatial = any("conv.0.weight" in k for k in state_dict.keys())

        # Determine in_channels
        in_channels = 6
        for k, v in state_dict.items():
            if "cnn.0.weight" in k or "conv.0.weight" in k or "features.0.weight" in k:
                in_channels = v.shape[1]
                break

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(in_channels, board_size, board_size), dtype=np.float32
        )
        act_space = spaces.Discrete(board_size * board_size)

        if is_spatial:
            feat_extractor = SpatialBattleshipCNN(obs_space, features_dim=100)
            feat_dim = 100
        else:
            feat_extractor = BattleshipCNN(obs_space, features_dim=256)
            feat_dim = 256

        # Infer net_arch from state dict keys
        if "q_net.q_net.4.weight" in state_dict:
            net_arch = [state_dict["q_net.q_net.0.weight"].shape[0], state_dict["q_net.q_net.2.weight"].shape[0]]
        elif "q_net.q_net.2.weight" in state_dict:
            net_arch = [state_dict["q_net.q_net.0.weight"].shape[0]]
        else:
            net_arch = []

        q_net = QNetwork(
            observation_space=obs_space,
            action_space=act_space,
            features_extractor=feat_extractor,
            features_dim=feat_dim,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            normalize_images=False,
        )

        q_net_state = {}
        for k, v in state_dict.items():
            if k.startswith("q_net."):
                q_net_state[k[len("q_net."):]] = v

        q_net.load_state_dict(q_net_state, strict=False)
        q_net.eval()
        q_net = q_net.to(device)

        _MODEL_CACHE[zip_path] = q_net
        return q_net
    except Exception:
        _MODEL_CACHE[zip_path] = None
        return None


class NeuralAgent(BaseAI):
    """AI strategy powered by a trained DQN model with multi-channel priors.

    Combines deep feature evaluation with probability heatmap & target boundary
    priors for state-of-the-art shot efficiency.
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        model_path: Optional[str] = None,
        phase_random: bool = True,
    ):
        """NeuralAgent that plays the trained DQN whenever one can be loaded.

        The checkpoint (``battleship_dqn_best.zip`` when present, otherwise
        ``battleship_dqn.zip``) is read lazily on the first move and its
        action-masked greedy output drives play. When no checkpoint can be
        loaded the agent falls back to the built-in heat-map + parity
        heuristic, so the strategy still works in deployments that ship
        without the model file. Two knobs keep the agent from being trivially
        predictable across games:

        - ``phase_random``  flips the checkerboard colour of the parity hint
          per game (either colour finds a size-2 ship equally well),
        - a small per-game noise term is added to hunt-mode Q-values only, so
          the opening varies while targeting decisions stay deterministic.
        """
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._q_net: Optional[nn.Module] = None
        self._model_attempted = False
        self._unsunk_hits: set[Tuple[int, int]] = set()
        self._phase_random = phase_random
        self._parity_shift = 0
        self._fallback_active = False
        # Per-game hunt ordering noise: shuffles the priority of near-tied
        # hunt cells so the AI never traces the same sequence twice.
        self._hunt_noise: Optional[np.ndarray] = None
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Neural"

    def reset(self) -> None:
        super().reset()
        self._unsunk_hits = set()
        if self._phase_random:
            self._parity_shift = random.randint(0, 1)
        # Roll a fresh random hunt-order each game: this noise is added to
        # Q-values of valid parity cells so the AI starts in a different
        # area of the board every game, not always the same diagonal.
        n = self.board_size * self.board_size
        self._hunt_noise = np.random.uniform(0.0, 2.0, size=n).astype(np.float32)
        self._observation = self._build_obs()

    def choose_move(self) -> Tuple[int, int]:
        """Return the best un-shot cell.

        When a trained DQN is available its action-masked greedy move is
        used; otherwise the heat-map + parity heuristic below takes over.
        """
        if self._ensure_model():
            return self._model_move()

        mask = np.ones(self.board_size * self.board_size, dtype=bool)
        for r, c in self.shots_taken:
            mask[r * self.board_size + c] = False

        # 1. Target mode: finish wounded ships immediately with strict axis extension
        if self._unsunk_hits:
            exts = axis_extension_cells(self._unsunk_hits, self.shots_taken, self.board_size)
            if exts:
                heat = self._observation[4]
                best_ext = max(exts, key=lambda rc: heat[rc[0], rc[1]])
                return best_ext

            # Single hit: pick orthogonal neighbor with the longest open corridor
            heat = self._observation[4]
            best_neighbor = None
            best_score = -1e9
            blocking = self.misses | (self.hits - self._unsunk_hits)
            min_size = min(self.remaining_ship_sizes) if self.remaining_ship_sizes else 2

            for r, c in self._unsunk_hits:
                c_left = c
                while c_left > 0 and (r, c_left - 1) not in blocking:
                    c_left -= 1
                c_right = c
                while c_right < self.board_size - 1 and (r, c_right + 1) not in blocking:
                    c_right += 1
                h_span = c_right - c_left + 1

                r_up = r
                while r_up > 0 and (r_up - 1, c) not in blocking:
                    r_up -= 1
                r_down = r
                while r_down < self.board_size - 1 and (r_down + 1, c) not in blocking:
                    r_down += 1
                v_span = r_down - r_up + 1

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and (nr, nc) not in self.shots_taken:
                        span = h_span if dr == 0 else v_span
                        if span >= min_size:
                            sc = heat[nr, nc] * 10.0 + span * 2.0
                            if sc > best_score:
                                best_score = sc
                                best_neighbor = (nr, nc)

            if best_neighbor is not None:
                return best_neighbor

        # 2. Hunt mode (Pure Optimal - No Wasted Shots):
        heat = self._observation[4].copy()
        parity = self._observation[5].copy()

        # Soft discount perimeter of sunk ships
        for r, c in self.sunk_perimeter:
            if (r, c) not in self.shots_taken:
                heat[r, c] *= 0.02

        # Dead-cell elimination
        valid_heat = mask.reshape((10, 10)) & (heat > 0)
        valid_mask = valid_heat.flatten()
        if not np.any(valid_mask):
            valid_mask = mask

        score = heat.flatten() * 10.0 + parity.flatten() * 6.0
        score[~valid_mask] = -1e9

        # Micro-jitter only for exact ties (preserves maximum mathematical efficiency)
        score = score + np.random.uniform(0.0, 0.01, size=score.shape)
        score[~mask] = -1e9

        action = int(np.argmax(score))
        return divmod(action, self.board_size)

    # ── Model-backed move selection ──────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Load the DQN lazily (at most once).

        Returns True when a network is available for play. A network injected
        by the caller (see ``GreedyEvalCallback``) is reused instead of
        re-reading it from disk.
        """
        if self._q_net is not None:
            return True
        if self._model_attempted:
            return False
        self._model_attempted = True
        self._q_net = load_dqn_qnetwork(self._model_path, self.board_size)
        self._fallback_active = self._q_net is None
        return self._q_net is not None

    def _model_move(self) -> Tuple[int, int]:
        """Action-masked greedy move from the loaded Q-network."""
        mask = np.ones(self.board_size * self.board_size, dtype=bool)
        for r, c in self.shots_taken:
            mask[r * self.board_size + c] = False

        device = next(self._q_net.parameters()).device
        obs_t = torch.as_tensor(
            self._observation[np.newaxis], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            q = self._q_net(obs_t).cpu().numpy().flatten()

        # Hunt mode only: a fresh per-game noise term keeps openings varied
        # without disturbing the targeting decisions the net has learned.
        if not self._unsunk_hits and self._hunt_noise is not None:
            q = q + self._hunt_noise

        q[~mask] = -np.inf
        return divmod(int(np.argmax(q)), self.board_size)

    def process_result(
        self,
        row: int,
        col: int,
        result: str,
        sunk_ship_name: Optional[str] = None,
        sunk_ship_size: Optional[int] = None,
    ) -> None:
        """Record the shot and rebuild the observation tensor."""
        self._record_shot(row, col, result, sunk_ship_name, sunk_ship_size)

        if result == "hit":
            self._unsunk_hits.add((row, col))
        elif result == "sunk":
            self._unsunk_hits.add((row, col))
            sunk_cells = [(row, col)]
            if sunk_ship_size and sunk_ship_size > 1:
                # Find horizontal segment
                h_hits = [(row, col)]
                c = col - 1
                while (row, c) in self._unsunk_hits:
                    h_hits.append((row, c))
                    c -= 1
                c = col + 1
                while (row, c) in self._unsunk_hits:
                    h_hits.append((row, c))
                    c += 1

                # Find vertical segment
                v_hits = [(row, col)]
                r = row - 1
                while (r, col) in self._unsunk_hits:
                    v_hits.append((r, col))
                    r -= 1
                r = row + 1
                while (r, col) in self._unsunk_hits:
                    v_hits.append((r, col))
                    r += 1

                if len(h_hits) == sunk_ship_size:
                    sunk_cells = h_hits
                elif len(v_hits) == sunk_ship_size:
                    sunk_cells = v_hits
                elif len(h_hits) > len(v_hits):
                    # Sort horizontal by col and take slice of length sunk_ship_size containing (row, col)
                    h_sorted = sorted(h_hits, key=lambda x: x[1])
                    idx = h_sorted.index((row, col))
                    start = max(0, min(idx - sunk_ship_size + 1, len(h_sorted) - sunk_ship_size))
                    sunk_cells = h_sorted[start:start + sunk_ship_size]
                elif len(v_hits) > 1:
                    v_sorted = sorted(v_hits, key=lambda x: x[0])
                    idx = v_sorted.index((row, col))
                    start = max(0, min(idx - sunk_ship_size + 1, len(v_sorted) - sunk_ship_size))
                    sunk_cells = v_sorted[start:start + sunk_ship_size]

            for sc in sunk_cells:
                self._unsunk_hits.discard(sc)
            self._mark_sunk_perimeter(sunk_cells)

        self._observation = self._build_obs()

    # ── Internal helpers ───────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Build the 6-channel observation tensor.

        Channels:
            0 – un-attacked mask
            1 – hit mask
            2 – miss mask
            3 – sunk mask
            4 – probability density heat-map
            5 – target boundary & parity mask
        """
        board = self.board_size
        obs = np.zeros((6, board, board), dtype=np.float32)

        min_size = min(self.remaining_ship_sizes) if self.remaining_ship_sizes else 2

        # 0: Unattacked, 1: Hits, 2: Misses, 3: Sunk
        for r in range(board):
            for c in range(board):
                cell = (r, c)
                if cell not in self.shots_taken:
                    obs[0, r, c] = 1.0
                if cell in self.hits:
                    obs[1, r, c] = 1.0
                if cell in self.misses:
                    obs[2, r, c] = 1.0
                if cell in self.hits and cell not in self._unsunk_hits:
                    obs[3, r, c] = 1.0

        # 4: Probability Heatmap Layer (Target-Conditioned)
        heatmap = np.zeros((board, board), dtype=np.float32)
        invalid_mask = self.misses | (self.hits - self._unsunk_hits)

        # Sunk-ship adjacency exclusion: discount placements that touch sunk ships
        sunk_cells = self.hits - self._unsunk_hits
        sunk_adjacent = set()
        for sr, sc in sunk_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < board and 0 <= nc < board:
                        if (nr, nc) not in self.shots_taken and (nr, nc) not in sunk_cells:
                            sunk_adjacent.add((nr, nc))

        for size in self.remaining_ship_sizes:
            # Horizontal placements
            for r in range(board):
                for c in range(board - size + 1):
                    coords = [(r, c + i) for i in range(size)]
                    if not any(co in invalid_mask for co in coords):
                        overlap = sum(1 for co in coords if co in self._unsunk_hits)
                        weight = 1.0 + (overlap * 5.0 if self._unsunk_hits else 0.0)
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
                        overlap = sum(1 for co in coords if co in self._unsunk_hits)
                        weight = 1.0 + (overlap * 5.0 if self._unsunk_hits else 0.0)
                        if any(co in sunk_adjacent for co in coords):
                            # Ships never touch on gapped boards, so a placement
                            # that touches a sunk ship is impossible, not unlikely.
                            continue
                        for cr, cc in coords:
                            heatmap[cr, cc] += weight

        # Zero out shot cells in heatmap and normalize
        for r, c in self.shots_taken:
            heatmap[r, c] = 0.0
        max_h = heatmap.max()
        if max_h > 0:
            obs[4] = heatmap / max_h

        # 5: Target boundary & dynamic parity mask
        if self._unsunk_hits:
            exts = axis_extension_cells(self._unsunk_hits, self.shots_taken, board)
            if exts:
                for nr, nc in exts:
                    obs[5, nr, nc] = 1.0
            else:
                # Single hit: probe orthogonal neighbors, but prune directions where
                # remaining open corridor is shorter than the smallest surviving ship.
                blocking = self.misses | sunk_cells
                for r, c in self._unsunk_hits:
                    # Check horizontal open corridor through (r, c)
                    c_left = c
                    while c_left > 0 and (r, c_left - 1) not in blocking:
                        c_left -= 1
                    c_right = c
                    while c_right < board - 1 and (r, c_right + 1) not in blocking:
                        c_right += 1
                    h_span = c_right - c_left + 1

                    # Check vertical open corridor through (r, c)
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
                            if 0 <= nc < board and (r, nc) not in self.shots_taken:
                                obs[5, r, nc] = 1.0 + (0.2 if h_span > v_span else 0.0)
                    if v_span >= min_size:
                        for dr in (-1, 1):
                            nr = r + dr
                            if 0 <= nr < board and (nr, c) not in self.shots_taken:
                                obs[5, nr, c] = 1.0 + (0.2 if v_span > h_span else 0.0)
        else:
            # Dynamic parity modulus for smallest remaining ship
            # Also prune dead cells (where no remaining ship can fit)
            phase = self._parity_shift if self._phase_random else 0
            blocking = self.misses | sunk_cells
            for r in range(board):
                for c in range(board):
                    if (r, c) in self.shots_taken:
                        continue
                    if (r + c + phase) % min_size != 0:
                        continue
                    # Dead-cell check: both h-span and v-span < min_size
                    hs = c
                    while hs > 0 and (r, hs - 1) not in blocking:
                        hs -= 1
                    he = c
                    while he < board - 1 and (r, he + 1) not in blocking:
                        he += 1
                    if (he - hs + 1) >= min_size:
                        obs[5, r, c] = 1.0
                        continue
                    vs = r
                    while vs > 0 and (vs - 1, c) not in blocking:
                        vs -= 1
                    ve = r
                    while ve < board - 1 and (ve + 1, c) not in blocking:
                        ve += 1
                    if (ve - vs + 1) >= min_size:
                        obs[5, r, c] = 1.0

        return obs


