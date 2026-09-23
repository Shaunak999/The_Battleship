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

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .battleship_env import BattleshipEnv
from .features import SpatialBattleshipCNN

try:
    from ai.neural_ai import NeuralAgent
except ImportError:
    from backend.ai.neural_ai import NeuralAgent


# ── Masked DQN ────────────────────────────────────────────────────────────
#
# SB3 v2.9.0's DQN does NOT call env.action_masks() anywhere.
# This subclass overrides _sample_action to mask out already-shot cells
# during both exploration and exploitation, AND overrides train() so that
# the Bellman target calculation max_a' Q(s', a') strictly masks invalid
# actions in s' instead of overestimating Q-values on illegal repeat shots.

class MaskedDQN(DQN):
    """DQN that respects action masks from the environment.

    During exploration (epsilon-greedy):
      Samples only from valid (un-shot) actions.
    During exploitation (greedy):
      Sets Q-values of invalid actions to -inf before argmax.
    During learning (train):
      Masks invalid actions in next_observations before target max.
    """

    def _setup_model(self) -> None:
        super()._setup_model()
        # Identity-align AND freeze the final linear head.
        #
        # Q is then a per-cell score: Q(s, a) = prior(a) + conv(s)(a), so the
        # ordering that makes the hand-crafted prior good is preserved and the
        # CNN can only add a (spatially local) additive correction. Leaving the
        # head trainable makes Q(s, a) = W @ features(s): a 100x100 matrix that
        # mixes every cell's features into every action, letting SGD rotate the
        # whole ranking away from the prior (measured: 39 -> 49 avg shots).
        for net in (getattr(self, "q_net", None), getattr(self, "q_net_target", None)):
            if net is None or not hasattr(net, "q_net"):
                continue
            head = net.q_net
            if (
                len(head) > 0
                and getattr(head[0], "weight", None) is not None
                and head[0].weight.shape == (100, 100)
            ):
                with torch.no_grad():
                    head[0].weight.copy_(torch.eye(100))
                    if head[0].bias is not None:
                        head[0].bias.zero_()
                # Frozen AFTER the optimizer was built: parameters without a
                # gradient are skipped by torch's optimizers, so this is enough.
                head[0].weight.requires_grad_(False)
                if head[0].bias is not None:
                    head[0].bias.requires_grad_(False)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Override train to apply action masks to next_observations."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Next Q-values from target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                
                # Channel 0 is un-attacked mask (1.0 = valid, 0.0 = already shot)
                next_unattacked = replay_data.next_observations[:, 0].flatten(start_dim=1)
                next_mask = next_unattacked > 0.5
                
                # Set invalid actions to large negative value before max
                next_q_values[~next_mask] = -1e9
                next_q_values, _ = torch.max(next_q_values, dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Current Q-values for taken actions
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

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

        # Epsilon-greedy (Guided Exploration)
        if np.random.random() < self.exploration_rate:
            obs = self._last_obs
            actions = []
            for i in range(n_envs):
                # Use Probability/Target priors from Channel 4 & 5 with exploration jitter
                prior = obs[i, 4] * 20.0 + obs[i, 5] * 100.0
                prior_flat = prior.flatten() + np.random.uniform(0.0, 5.0, size=100)
                prior_flat[~masks[i]] = -np.inf
                if np.all(np.isneginf(prior_flat)):
                    actions.append(np.random.choice(np.where(masks[i])[0]))
                else:
                    # 80% sample top prior, 20% random valid for novel discoveries
                    if np.random.random() < 0.8:
                        actions.append(int(np.argmax(prior_flat)))
                    else:
                        actions.append(int(np.random.choice(np.where(masks[i])[0])))
            actions = np.array(actions)
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

        # Navigate: VecMonitor -> DummyVecEnv -> raw envs
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
                eps = getattr(self.model, "exploration_rate", 0.0)
                print(
                    f"  [Train Ep {len(self._ep_rewards):6d}] "
                    f"avg_shots={avg_s:5.1f} (noise eps={eps:.2f})  avg_reward={avg_r:+7.2f}"
                )
        return True


# ── Greedy evaluation callback ────────────────────────────────────────────
#
# The RewardLogger above reports shots measured UNDER epsilon-greedy noise,
# which stays in the high 80s-90s because random exploratory moves are taken.
# This callback instead drives the *deployed* NeuralAgent with the live
# network, so the score describes the policy that actually ships.

class GreedyEvalCallback(BaseCallback):
    """Every ``eval_freq`` steps, play greedy games and save the best model.

    Evaluation reuses :class:`ai.neural_ai.NeuralAgent` - the same class the
    backend serves - with the training network injected and the training board
    distribution. That keeps the reported score, the saved "best" checkpoint
    and early stopping all tied to the shipped policy rather than to a
    separate heuristic.

    Early-stopping: if greedy eval doesn't improve for ``patience``
    consecutive evaluations, training is halted.
    """

    def __init__(
        self,
        eval_freq: int = 25_000,
        n_games: int = 50,
        best_path: str = "ai/neural/battleship_dqn_best",
        patience: int = 8,
        touch_probability: float = 0.0,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.best_path = best_path
        self.patience = patience
        self.best_avg_shots: float = float("inf")
        self._last_eval: int = 0
        self._no_improve_count: int = 0
        # Same board distribution the env is trained on, so the score is a
        # like-for-like measurement and not a distribution shift.
        self._env = BattleshipEnv(touch_probability=touch_probability)
        self._agent = NeuralAgent()

    def _play_greedy_game(self) -> int:
        """Play one fully-greedy game with the live network; return shots used."""
        # Inject the policy currently being trained so we measure what ships.
        self._agent._q_net = self.model.policy.q_net
        self._agent.reset()
        self._env.reset()

        shots = 0
        while shots < self._env.n_cells:
            row, col = self._agent.choose_move()
            _, _, terminated, truncated, info = self._env.step(
                row * self._env.board_size + col
            )
            shots += 1

            result = info.get("result", "miss")
            if result in ("sunk", "win"):
                self._agent.process_result(
                    row, col, "sunk", info.get("ship"), info.get("ship_size")
                )
            else:
                self._agent.process_result(row, col, result)

            if terminated or truncated:
                break

        return shots

    def _evaluate(self, step: int):
        shots_list = [self._play_greedy_game() for _ in range(self.n_games)]

        avg_shots = float(np.mean(shots_list))
        win_rate = (
            100.0 * sum(1 for s in shots_list if s < self._env.n_cells) / len(shots_list)
        )

        improved = avg_shots < self.best_avg_shots
        if improved:
            self.best_avg_shots = avg_shots
            self._no_improve_count = 0
            # Never checkpoint the untrained step-0 snapshot: only a policy
            # that actually beat the baseline is worth keeping as "best".
            if step > 0:
                os.makedirs(os.path.dirname(self.best_path) or ".", exist_ok=True)
                self.model.save(self.best_path)
        else:
            self._no_improve_count += 1

        tag = "saved (new best)" if improved else f"best={self.best_avg_shots:.1f}"
        patience_str = f"  patience={self._no_improve_count}/{self.patience}"
        print(
            f"  >>> [GREEDY EVAL @ {step:>7,} steps] "
            f"avg_shots={avg_shots:5.1f}  win={win_rate:3.0f}%  -> {tag}{patience_str}"
        )

    def _on_training_start(self) -> None:
        self._evaluate(0)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        self._evaluate(self.num_timesteps)
        # Early stopping
        if self._no_improve_count >= self.patience:
            print(f"  >>> EARLY STOP: no improvement for {self.patience} evals")
            return False
        return True


# ── Training ───────────────────────────────────────────────────────────────

def _cosine_lr_schedule(initial_lr: float, min_lr: float = 1e-6):
    """Return a callable LR schedule that cosine-decays from initial_lr to min_lr."""
    import math
    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 -> 0.0 during training
        cosine = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return min_lr + (initial_lr - min_lr) * cosine
    return schedule


def train(
    total_timesteps: int = 500_000,
    save_path: str = "ai/neural/battleship_dqn",
    learning_rate: float = 1e-4,
    buffer_size: int = 150_000,
    batch_size: int = 64,
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.02,
    learning_starts: int = 500,
    target_update_interval: int = 1000,
    train_freq: int = 4,
    gradient_steps: int = 2,
    eval_freq: int = 25_000,
    eval_games: int = 50,
    n_envs: int = 4,
    patience: int = 8,
    touch_probability: float = 0.0,
    verbose: int = 1,
    **kwargs,
) -> MaskedDQN:
    """Train a DQN agent and save it to *save_path*.

    ``touch_probability`` controls the ship-placement style used for both
    training and greedy evaluation (0.0 = ships never touch, 1.0 = only
    overlaps are forbidden). Keeping the two identical is essential: a model
    trained on gapped boards is measured on gapped boards.
    """

    print("=" * 60)
    print("  BATTLESHIP DQN TRAINING (Spatial CNN + Action Masking)")
    print("=" * 60)

    # Vectorized environments for parallel batch data collection
    env = DummyVecEnv(
        [lambda: BattleshipEnv(touch_probability=touch_probability) for _ in range(n_envs)]
    )
    env = VecMonitor(env)

    # Check if we should resume from existing model
    resume = kwargs.pop("resume", False)
    model_path = save_path + ".zip"
    if resume and os.path.exists(model_path):
        print(f"\nResuming from {model_path} with fine-tuning exploration (eps: 0.08 -> 0.02)")
        model = MaskedDQN.load(save_path)
        model.set_env(env)
        # Construct explicit fine-tuning exploration schedule
        model.exploration_initial_eps = 0.08
        model.exploration_final_eps = 0.02
        model.exploration_rate = 0.08
        model.exploration_schedule = lambda progress: 0.02 + (0.08 - 0.02) * progress
    else:
        lr_schedule = _cosine_lr_schedule(learning_rate, min_lr=1e-6)
        model = MaskedDQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=lr_schedule,
            buffer_size=buffer_size,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            verbose=verbose,
            device="auto",
            policy_kwargs=dict(
                features_extractor_class=SpatialBattleshipCNN,
                features_extractor_kwargs=dict(features_dim=100),
                net_arch=[],
                normalize_images=False,
            ),
        )

    print(f"\nTraining for {total_timesteps:,} timesteps ...")
    print(f"  Parallel Envs: {n_envs}")
    print(f"  Policy:        CnnPolicy (SpatialBattleshipCNN)")
    print(f"  Action mask:   ENABLED (MaskedDQN)")
    print(f"  Train cadence: {train_freq} step(s), {gradient_steps} grad step(s)")
    print(f"  Buffer size:   {buffer_size:,}")
    print(f"  Batch size:    {batch_size}")
    print(f"  LR:            {learning_rate}")
    print(f"  Explore:       {exploration_fraction} of run -> eps {exploration_final_eps}")
    print(f"  LR schedule:   cosine decay {learning_rate} -> 1e-6")
    print(f"  Board style:   touch_probability={touch_probability}")
    print(f"  Greedy eval:   every {eval_freq:,} steps x {eval_games} games")
    print(f"  Early stop:    patience={patience} evals")
    print()

    callbacks = [
        RewardLogger(),
        GreedyEvalCallback(
            eval_freq=eval_freq,
            n_games=eval_games,
            best_path=save_path + "_best",
            patience=patience,
            touch_probability=touch_probability,
        ),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    reward_logger = callbacks[0]
    greedy_eval = callbacks[1]

    if reward_logger._ep_rewards:
        last_rewards = reward_logger._ep_rewards[-100:]
        last_shots = reward_logger._ep_shots[-100:]
        print(f"\nFinal 100 episodes (epsilon-greedy):")
        print(f"  Avg reward: {np.mean(last_rewards):+.2f}")
        print(f"  Avg shots:  {np.mean(last_shots):.1f}")
        print(f"  Win rate:   {sum(1 for r in last_rewards if r > 5) / len(last_rewards) * 100:.0f}%")

    if greedy_eval.best_avg_shots < float("inf"):
        print(f"\nBest GREEDY avg shots: {greedy_eval.best_avg_shots:.1f}")
        if os.path.exists(save_path + "_best.zip"):
            print(f"  Saved to {save_path}_best.zip")
        else:
            print("  (baseline was never beaten - kept the final model only)")

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
    parser.add_argument("--buffer-size", type=int, default=150_000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Greedy self-evaluation every N steps")
    parser.add_argument("--eval-games", type=int, default=50,
                        help="Games per greedy self-evaluation")
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Run a gradient update every N env steps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--exploration-fraction", type=float, default=0.2,
                        help="Fraction of training period for epsilon decay")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing saved model")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stop after N evals with no improvement")
    parser.add_argument("--touch-prob", type=float, default=0.0,
                        help="Ship-placement style used for training AND eval: "
                             "0.0 = ships never touch (default), 1.0 = ships may touch")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games,
        train_freq=args.train_freq,
        n_envs=args.n_envs,
        exploration_fraction=args.exploration_fraction,
        patience=args.patience,
        touch_probability=args.touch_prob,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
