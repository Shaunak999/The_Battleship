"""
Feature extractors shared by the trainer and the runtime agent.

Both :mod:`ai.neural.train` (which builds the network) and
:mod:`ai.neural_ai` (which loads a checkpoint back into a network) must agree
on the architecture *exactly* — any drift silently changes how the saved
weights are interpreted. Keeping the classes here is what guarantees that.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialBattleshipCNN(BaseFeaturesExtractor):
    """Spatially-preserving grid architecture for 10x10 Battleship.

    The output is ``features_dim`` (100) values, one per board cell, so the
    downstream 1:1 head can turn them straight into per-cell Q-values.

    Q is built as a *strong fixed prior plus a bounded local correction*::

        Q(a) = w_heat * heat(a) + w_target * target(a) + scale * tanh(conv(s)(a))

    ``heat`` (channel 4) and ``target`` (channel 5) are the probability and
    target/parity priors, so an untrained network already plays at roughly the
    heuristic's strength. The CNN may only nudge each cell by +/- ``scale``,
    which is small compared with the prior, so it can break near-ties but can
    never re-rank cells the prior is confident about.

    That bound matters: letting the CNN (or a trainable head) act on the full
    prior scale lets the value-regression loss rotate the whole ranking away
    from the prior. Measured on gapped boards, an unbounded residual degrades
    the greedy policy from ~39 to ~49 average shots as training proceeds,
    i.e. training makes the agent worse than not training at all.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 100,
        residual_scale: float = 2.0,
    ):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 6

        self.residual_scale = float(residual_scale)

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Prior multipliers. Buffers (not Parameters) so the tuned heuristic
        # scale cannot be rescaled by gradients — a shrunken prior would hand
        # the ordering back to the bounded residual.
        self.register_buffer("w_heat", torch.tensor(10.0))
        self.register_buffer("w_target", torch.tensor(50.0))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.conv(observations)).squeeze(1) * self.residual_scale
        prior = self.w_heat * observations[:, 4] + self.w_target * observations[:, 5]
        return (residual + prior).flatten(start_dim=1)


class BattleshipCNN(BaseFeaturesExtractor):
    """Legacy CNN feature extractor matching the older Battleship architecture.

    Kept purely so checkpoints written before the spatial architecture can
    still be loaded; new training never uses it.
    """

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
