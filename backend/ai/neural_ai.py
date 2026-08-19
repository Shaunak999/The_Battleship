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
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork

from .base_ai import BaseAI, BOARD_SIZE, SHIP_DEFINITIONS


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


class BattleshipCNN(BaseFeaturesExtractor):
    """CNN feature extractor matching the Battleship architecture."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[np.newaxis], dtype=torch.float32
            )
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def load_dqn_qnetwork(model_path: str, board_size: int = BOARD_SIZE) -> Optional[nn.Module]:
    """Load and cache the PyTorch QNetwork / DuelingQNetwork weights from disk."""
    if not model_path.endswith(".zip"):
        zip_path = model_path + ".zip"
    else:
        zip_path = model_path

    if zip_path in _MODEL_CACHE:
        return _MODEL_CACHE[zip_path]

    if not os.path.exists(zip_path):
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            pth = io.BytesIO(z.read("policy.pth"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(pth, map_location=device)

        # Check if saved model has 4 or 6 channels
        in_channels = 4
        for k, v in state_dict.items():
            if "cnn.0.weight" in k or "features.0.weight" in k:
                in_channels = v.shape[1]
                break

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(in_channels, board_size, board_size), dtype=np.float32
        )
        act_space = spaces.Discrete(board_size * board_size)
        feat_extractor = BattleshipCNN(obs_space, features_dim=256)

        q_net = QNetwork(
            observation_space=obs_space,
            action_space=act_space,
            features_extractor=feat_extractor,
            features_dim=256,
            net_arch=[64, 64],
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
    ):
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._q_net: Optional[nn.Module] = None
        self._unsunk_hits: set[Tuple[int, int]] = set()
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Neural"

    def reset(self) -> None:
        super().reset()
        self._unsunk_hits = set()
        self._observation = self._build_obs()

    def choose_move(self) -> Tuple[int, int]:
        """Pick the best un-shot cell using masked model predictions or heatmap prior."""
        if self._q_net is None:
            self._q_net = load_dqn_qnetwork(self._model_path, self.board_size)

        # Build mask of valid (un-attacked) cells
        mask = np.ones(self.board_size * self.board_size, dtype=bool)
        for r, c in self.shots_taken:
            mask[r * self.board_size + c] = False

        if self._q_net is not None:
            # Check model channel count
            in_c = 4
            for p in self._q_net.parameters():
                if p.dim() == 4:
                    in_c = p.shape[1]
                    break

            obs_slice = self._observation[:in_c]
            obs_tensor = torch.from_numpy(obs_slice).unsqueeze(0)  # (1, C, 10, 10)
            
            # Move to same device as model
            device = next(self._q_net.parameters()).device
            obs_tensor = obs_tensor.to(device)
            
            with torch.no_grad():
                q_values = self._q_net(obs_tensor).cpu().numpy().flatten()

            q_values[~mask] = -np.inf
            action = int(np.argmax(q_values))
            return divmod(action, self.board_size)

        # High-performance hybrid fallback: use heatmap + target mask directly
        scores = self._observation[4] * 100.0 + self._observation[5] * 500.0
        scores_flat = scores.flatten()
        scores_flat[~mask] = -np.inf
        action = int(np.argmax(scores_flat))
        return divmod(action, self.board_size)

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
            # Clear hits belonging to sunk ship
            if sunk_ship_size:
                # Find connected hits in line
                to_remove = set()
                to_remove.add((row, col))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    cr, cc = row + dr, col + dc
                    while (cr, cc) in self._unsunk_hits and len(to_remove) < sunk_ship_size:
                        to_remove.add((cr, cc))
                        cr += dr
                        cc += dc
                self._unsunk_hits -= to_remove

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

        # 4: Probability Heatmap Layer
        heatmap = np.zeros((board, board), dtype=np.float32)
        invalid_mask = self.misses | (self.hits - self._unsunk_hits)

        for size in self.remaining_ship_sizes:
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
        for r, c in self.shots_taken:
            heatmap[r, c] = 0.0
        max_h = heatmap.max()
        if max_h > 0:
            obs[4] = heatmap / max_h

        # 5: Target boundary & parity mask
        if self._unsunk_hits:
            for r, c in self._unsunk_hits:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board and 0 <= nc < board and (nr, nc) not in self.shots_taken:
                        obs[5, nr, nc] = 1.0
        else:
            # Parity hunting pattern for smallest remaining ship
            for r in range(board):
                for c in range(board):
                    if (r + c) % min_size == 0 and (r, c) not in self.shots_taken:
                        obs[5, r, c] = 1.0

        return obs


