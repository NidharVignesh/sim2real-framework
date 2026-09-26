"""Train a PPO get-up policy for robo1 with Stable-Baselines3.

Every episode starts from a randomly drawn fall (4 canonical poses, any random
orientation including upside down, or already upright) — see robo1_env.py.

Outputs (all standard SB3 .zip files):
    robo1_getup_ppo.zip          final model
    best_model/best_model.zip    best model by EvalCallback success
    checkpoints/robo1_ppo_*.zip  periodic checkpoints

Usage:
    python train.py                                  # 2M steps, 8 envs, domain randomization on
    python train.py --timesteps 500000 --no-domain-rand
    python train.py --resume robo1_getup_ppo.zip --timesteps 1000000
    tensorboard --logdir logs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robo1_env import Robo1GetupEnv

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain-rand", action=argparse.BooleanOptionalAction, default=True,
                        help="Randomize mass, damping, friction, servo speed and IMU noise")
    parser.add_argument("--resume", type=str, default=None, help="Model .zip to continue training")
    parser.add_argument("--out", type=str, default=str(HERE / "robo1_getup_ppo"))
    parser.add_argument("--log-dir", type=str, default=str(HERE / "logs"))
    args = parser.parse_args()

    env = make_vec_env(
        Robo1GetupEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        env_kwargs={"domain_randomization": args.domain_rand},
    )
    # Evaluation on nominal physics, no sensor noise, all start cases.
    eval_env = make_vec_env(Robo1GetupEnv, n_envs=1, seed=args.seed + 1000)

    try:
        import tensorboard  # noqa: F401
        tb_log = args.log_dir
    except ImportError:
        tb_log = None

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=tb_log)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,        # per env -> 8192 samples per rollout with 8 envs
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # Small tanh MLP so the policy fits on the ESP32.
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh),
            tensorboard_log=tb_log,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    callbacks = [
        EvalCallback(
            eval_env,
            n_eval_episodes=30,
            eval_freq=max(50_000 // args.n_envs, 1),
            best_model_save_path=str(HERE / "best_model"),
            log_path=args.log_dir,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(250_000 // args.n_envs, 1),
            save_path=str(HERE / "checkpoints"),
            name_prefix="robo1_ppo",
        ),
    ]

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted, saving current model.")
    model.save(args.out)
    print(f"Saved {args.out}.zip")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
