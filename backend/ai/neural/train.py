"""
Train a DQN agent on the Battleship environment using Stable-Baselines3.

Usage:
    cd backend
    python -m ai.neural.train [--timesteps N] [--save-path PATH]

The trained model is saved to ``neural/battleship_dqn.zip`` by default.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .battleship_env import BattleshipEnv


# ── Custom CNN features extractor (handles 6×10×10 input) ─────────────────

class BattleshipCNN(BaseFeaturesExtractor):
    """CNN that works with the 6×10×10 Battleship observation tensor.

    Architecture:
        Conv2d(6→32, 3×3, pad 1) → ReLU → Conv2d(32→64, 3×3, pad 1) → ReLU
        → Conv2d(64→128, 3×3, pad 1) → ReLU → Flatten → Linear(→256) → ReLU
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 6

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


# ── Masked DQN ────────────────────────────────────────────────────────────
#
# SB3 v2.9.0's DQN does NOT call env.action_masks() anywhere.
# This subclass overrides _sample_action to mask out already-shot cells
# during both exploration and exploitation.

class MaskedDQN(DQN):
    """DQN that respects action masks from the environment.

    During exploration (epsilon-greedy):
      Samples only from valid (un-shot) actions.
    During exploitation (greedy):
      Sets Q-values of invalid actions to -inf before argmax.
    """

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ) -> tuple:
        """Apply action masking during action selection."""
        masks = self._get_action_masks(n_envs)

        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup: random valid actions only
            actions = np.array([
                np.random.choice(np.where(masks[i])[0])
                for i in range(n_envs)
            ])
            return actions, actions

        # Epsilon-greedy
        if np.random.random() < self.exploration_rate:
            # Explore: random valid action
            actions = np.array([
                np.random.choice(np.where(masks[i])[0])
                for i in range(n_envs)
            ])
        else:
            # Exploit: greedy with masking
            obs = self._last_obs
            obs_tensor = self.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                q_values = self.policy.q_net(obs_tensor).cpu().numpy()

            # Mask invalid actions
            for i in range(q_values.shape[0]):
                q_values[i][~masks[i]] = -np.inf

            actions = np.argmax(q_values, axis=1)

        return actions, actions

    def _get_action_masks(self, n_envs: int) -> np.ndarray:
        """Extract action masks from the env.

        DummyVecEnv stores raw envs in self.env.envs[].
        VecMonitor wraps DummyVecEnv, accessible via self.env.venv.
        """
        n_actions = self.action_space.n

        # Navigate: VecMonitor → DummyVecEnv → raw envs
        vec_env = self.env
        if hasattr(vec_env, "venv"):
            vec_env = vec_env.venv

        if hasattr(vec_env, "envs"):
            try:
                masks = [e.action_masks() for e in vec_env.envs[:n_envs]]
                return np.array(masks, dtype=bool)
            except (AttributeError, IndexError):
                pass

        # Fallback: all actions valid
        return np.ones((n_envs, n_actions), dtype=bool)


# ── Reward logging callback ────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Log per-episode reward and shot count during training."""

    def __init__(self):
        super().__init__()
        self._ep_rewards: list[float] = []
        self._ep_shots: list[int] = []
        self._current_reward: float = 0.0
        self._current_shots: int = 0

    def _on_step(self) -> bool:
        self._current_reward += float(self.locals["rewards"][0])
        self._current_shots += 1

        if self.locals["dones"][0]:
            self._ep_rewards.append(self._current_reward)
            self._ep_shots.append(self._current_shots)
            self._current_reward = 0.0
            self._current_shots = 0

            if len(self._ep_rewards) % 100 == 0:
                avg_r = np.mean(self._ep_rewards[-100:])
                avg_s = np.mean(self._ep_shots[-100:])
                print(
                    f"  [Ep {len(self._ep_rewards):6d}] "
                    f"avg_reward={avg_r:+7.2f}  avg_shots={avg_s:5.1f}"
                )
        return True


# ── Training ───────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 500_000,
    save_path: str = "ai/neural/battleship_dqn",
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    exploration_fraction: float = 0.3,
    learning_starts: int = 1000,
    target_update_interval: int = 500,
    verbose: int = 1,
    **kwargs,
) -> MaskedDQN:
    """Train a DQN agent and save it to *save_path*."""

    print("=" * 60)
    print("  BATTLESHIP DQN TRAINING")
    print("=" * 60)

    # Plain DummyVecEnv — MaskedDQN reaches through it to get masks
    env = DummyVecEnv([lambda: BattleshipEnv()])
    env = VecMonitor(env)

    # Check if we should resume from existing model
    resume = kwargs.pop("resume", False)
    model_path = save_path + ".zip"
    if resume and os.path.exists(model_path):
        print(f"\nResuming from {model_path}")
        model = MaskedDQN.load(save_path)
        model.set_env(env)
    else:
        model = MaskedDQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=0.05,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            train_freq=4,
            gradient_steps=1,
            verbose=verbose,
            device="auto",
            policy_kwargs=dict(
                features_extractor_class=BattleshipCNN,
                features_extractor_kwargs=dict(features_dim=256),
                normalize_images=False,
            ),
        )

    print(f"\nTraining for {total_timesteps:,} timesteps ...")
    print(f"  Policy:        CnnPolicy (BattleshipCNN)")
    print(f"  Action mask:   ENABLED (MaskedDQN)")
    print(f"  Buffer size:   {buffer_size:,}")
    print(f"  Batch size:    {batch_size}")
    print(f"  LR:            {learning_rate}")
    print(f"  Explore frac:  {exploration_fraction}")
    print()

    callback = RewardLogger()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    if callback._ep_rewards:
        last_rewards = callback._ep_rewards[-100:]
        last_shots = callback._ep_shots[-100:]
        print(f"\nFinal 100 episodes:")
        print(f"  Avg reward: {np.mean(last_rewards):+.2f}")
        print(f"  Avg shots:  {np.mean(last_shots):.1f}")
        print(f"  Win rate:   {sum(1 for r in last_rewards if r > 5) / len(last_rewards) * 100:.0f}%")

    return model


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DQN on Battleship")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--save-path", type=str, default="ai/neural/battleship_dqn",
                        help="Path to save trained model")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing saved model")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
