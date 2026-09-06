"""PPO Training Script for SelfRisingRobot (robo1).

Trains a policy to right the robot from fallen poses using Proximal Policy
Optimization (PPO) via Stable-Baselines3.

Usage:
    python train.py --timesteps 200000 --n-envs 4
    python train.py --model-in robo1_getup_ppo.zip --timesteps 100000
    python train.py --pretrain --timesteps 200000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robo1_env import Robo1GetupEnv


def make_env(xml_path: str, randomize: bool = False, rank: int = 0, seed: int = 0):
    def _init():
        env = Robo1GetupEnv(xml_path=xml_path, randomize_pose_offset=randomize)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def pretrain_policy_from_scripted(model: PPO, xml_path: str, epochs: int = 2000) -> PPO:
    """Jumpstart the policy network using behavioral cloning from reference trajectories."""
    from getup_reference import getup_sequence_for_pose
    from robo1_env import FALLEN_POSES, roll_pitch_to_quat

    print(f"\n[Pretraining] Collecting expert demonstration data across 4 fallen poses...")
    observations = []
    actions = []

    for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
        env = Robo1GetupEnv(xml_path=xml_path, fallen_poses=(pose,))
        obs, _ = env.reset(options={"pose": pose})
        sequence = getup_sequence_for_pose(pose)

        # Build trajectory schedule
        targets = []
        prev = np.zeros(2, dtype=np.float64)
        for waypoint in sequence:
            for i in range(350):
                t = (i + 1) / 350.0
                ctrl = (1.0 - t) * prev + t * waypoint
                if i % env.frame_skip == env.frame_skip - 1:
                    targets.append(ctrl.copy())
            prev = waypoint

        for waypoint in targets:
            error = waypoint - env.target
            action = np.clip(error / env.target_delta, -1.0, 1.0).astype(np.float32)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()

    obs_tensor = torch.as_tensor(np.array(observations), dtype=torch.float32, device=model.device)
    act_tensor = torch.as_tensor(np.array(actions), dtype=torch.float32, device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)

    print(f"[Pretraining] Fitting policy network on {len(observations)} state-action pairs for {epochs} epochs...")
    for epoch in range(epochs):
        dist = model.policy.get_distribution(obs_tensor)
        pred = dist.distribution.mean
        loss = torch.mean((pred - act_tensor) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:4d}/{epochs} | MSE Loss: {loss.item():.6f}")

    print("[Pretraining] Complete! Transitioning to PPO RL exploration.\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy for robo1")
    parser.add_argument(
        "--xml",
        type=str,
        default=str(Path(__file__).parent / "robo1.xml"),
        help="Path to robo1.xml model",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200,000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Steps per environment per update (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size (default: 256)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Run behavioral cloning pretraining before PPO",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Add random orientation noise during training",
    )
    parser.add_argument(
        "--model-in",
        type=str,
        default=None,
        help="Existing model zip to resume training from",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(Path(__file__).parent / "robo1_getup_ppo.zip"),
        help="Output path for trained model zip",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).parent / "logs"),
        help="Tensorboard log directory",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in timesteps (default: 10,000)",
    )
    args = parser.parse_args()

    xml_path = str(Path(args.xml).resolve())
    out_path = Path(args.model_out).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SelfRisingRobot (robo1) PPO Training Pipeline")
    print("=" * 65)
    print(f"Model XML:         {xml_path}")
    print(f"Total Timesteps:   {args.timesteps:,}")
    print(f"Parallel Envs:     {args.n_envs}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Randomized Poses:  {args.randomize}")
    print(f"Output Path:       {out_path}")
    print(f"Logs Directory:    {log_dir}")
    print("=" * 65)

    # Build vectorized environments
    if args.n_envs > 1:
        vec_env = SubprocVecEnv(
            [make_env(xml_path, args.randomize, i, args.seed) for i in range(args.n_envs)]
        )
    else:
        vec_env = DummyVecEnv([make_env(xml_path, args.randomize, 0, args.seed)])

    # Separate evaluation environment
    eval_env = DummyVecEnv([make_env(xml_path, randomize=False, rank=99, seed=args.seed)])

    # Check if tensorboard is available
    tb_log = None
    try:
        import tensorboard  # noqa: F401
        tb_log = str(log_dir)
    except ImportError:
        pass

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_path.parent / "best_model"),
        log_path=str(log_dir) if tb_log else None,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(out_path.parent / "checkpoints"),
        name_prefix="robo1_ppo",
    )

    # Initialize or load PPO model
    if args.model_in:
        print(f"Loading existing model from: {args.model_in}")
        model = PPO.load(args.model_in, env=vec_env, device="cpu")
    else:
        # 64x64 Tanh MLP policy matches standard deployment network architecture
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
            device="cpu",
        )

    # Optional behavioral cloning pretraining
    if args.pretrain and not args.model_in:
        model = pretrain_policy_from_scripted(model, xml_path=xml_path, epochs=1500)

    has_progress_bar = False
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
        has_progress_bar = True
    except ImportError:
        pass

    # Run RL training
    print("Starting PPO training loop...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=has_progress_bar,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")

    # Save final model
    model.save(str(out_path))
    print(f"\nFinal model saved successfully to: {out_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
