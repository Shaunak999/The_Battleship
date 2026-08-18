"""
Neural AI strategy for Battleship.

Wraps a Stable-Baselines3 DQN model trained on the BattleshipEnv
into the BaseAI interface so it can be used interchangeably with
RandomAI, HuntTargetAI, and ProbabilityAI.

Usage
-----
    ai = NeuralAgent()                        # loads default model path
    ai = NeuralAgent(model_path="path/to/model.zip")
    move = ai.choose_move()
    ai.process_result(row, col, "hit")
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from stable_baselines3 import DQN

from .base_ai import BaseAI, BOARD_SIZE


# Default path relative to the backend/ directory
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "neural", "battleship_dqn"
)


class NeuralAgent(BaseAI):
    """AI strategy powered by a trained DQN model.

    After each ``process_result()`` call the agent rebuilds its internal
    observation tensor and uses the model to pick the next cell.
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        model_path: Optional[str] = None,
    ):
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._model: Optional[DQN] = None
        super().__init__(board_size)

    @property
    def name(self) -> str:
        return "Neural (DQN)"

    def reset(self) -> None:
        super().reset()
        self._observation = self._build_obs()
        # Don't load model here — defer to choose_move() so __init__
        # doesn't crash when no trained model exists yet.

    def choose_move(self) -> tuple[int, int]:
        """Use the DQN model to pick the best un-shot cell."""
        if self._model is None:
            self._load_model()

        obs = self._observation[np.newaxis, :]  # add batch dim → (1, 4, 10, 10)

        # Use the model's predict — SB3 DQN applies action masking
        # by setting masked Q-values to -inf during greedy actions
        action, _ = self._model.predict(obs, deterministic=True)

        # Manual fallback: if model picks an invalid cell, find best valid one
        action = int(action)
        row, col = divmod(action, self.board_size)
        if (row, col) in self.shots_taken:
            # Fallback: pick the valid action with highest Q-value
            action = self._greedy_fallback(obs)
            row, col = divmod(action, self.board_size)

        return row, col

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
        self._observation = self._build_obs()

    # ── Internal helpers ───────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the trained DQN from disk."""
        path = self._model_path
        if not path.endswith(".zip"):
            zip_path = path + ".zip"
        else:
            zip_path = path

        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"Trained model not found at {zip_path}. "
                f"Run `python -m ai.neural.train` first."
            )

        self._model = DQN.load(path)

    def _build_obs(self) -> np.ndarray:
        """Build the 4-channel observation tensor from current state.

        Channels:
            0 – un-attacked mask (1 = not yet shot)
            1 – hit mask
            2 – miss mask
            3 – sunk mask (approximation: hits with no unattacked neighbours)
        """
        obs = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)

        for r in range(self.board_size):
            for c in range(self.board_size):
                cell = (r, c)
                if cell not in self.shots_taken:
                    obs[0, r, c] = 1.0
                if cell in self.hits:
                    obs[1, r, c] = 1.0
                if cell in self.misses:
                    obs[2, r, c] = 1.0
                # Sunk cells: hits with no un-attacked neighbours
                if cell in self.hits:
                    has_unattacked_neighbour = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self.board_size
                            and 0 <= nc < self.board_size
                            and (nr, nc) not in self.shots_taken
                        ):
                            has_unattacked_neighbour = True
                            break
                    if not has_unattacked_neighbour:
                        obs[3, r, c] = 1.0

        return obs

    def _greedy_fallback(self, obs: np.ndarray) -> int:
        """If the model's greedy pick is invalid, scan all valid actions
        and return the one with the highest Q-value."""
        mask = np.ones(self.board_size * self.board_size, dtype=bool)
        for r, c in self.shots_taken:
            mask[r * self.board_size + c] = False

        q_values = self._model.policy.q_net(
            self._model.policy.obs_to_tensor(obs)[0]
        ).detach().cpu().numpy().flatten()

        q_values[~mask] = -np.inf
        return int(np.argmax(q_values))
