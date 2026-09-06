"""Evaluation and visualization script for trained robo1 policies.

Evaluates trained PPO models across fallen poses (roll_pos, roll_neg, pitch_pos, pitch_neg),
logging time-to-upright, success rate, and displaying live physics simulation in MuJoCo viewer.

Usage:
    # Run evaluation across all poses with visual viewer
    python eval_policy.py --model robo1_getup_ppo.zip --render

    # Headless benchmark across 20 trials per pose
    python eval_policy.py --model robo1_getup_ppo.zip --episodes 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from robo1_env import FALLEN_POSES, Robo1GetupEnv


def evaluate_pose(
    model: PPO,
    xml_path: str,
    pose: str,
    n_episodes: int = 5,
    render: bool = False,
) -> dict:
    """Evaluate policy performance from a specific fallen pose."""
    env = Robo1GetupEnv(xml_path=xml_path, fallen_poses=(pose,), render_mode="human" if render else None)

    successes = 0
    upright_times = []
    final_uprights = []

    for ep in range(n_episodes):
        obs, _ = env.reset(options={"pose": pose})
        ep_upright_step = None

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                time.sleep(1.0 / 50.0)

            if info["upright"] > 0.90 and ep_upright_step is None:
                ep_upright_step = step

            if terminated:
                successes += 1
                break
            if truncated:
                break

        final_uprights.append(info["upright"])
        if ep_upright_step is not None:
            # step at 50 Hz -> time in seconds
            upright_times.append(ep_upright_step * 0.02)

    env.close()

    success_rate = (successes / n_episodes) * 100.0
    avg_upright_time = float(np.mean(upright_times)) if upright_times else float("nan")
    avg_upright = float(np.mean(final_uprights))

    return {
        "pose": pose,
        "success_rate": success_rate,
        "avg_time_to_upright": avg_upright_time,
        "avg_final_upright": avg_upright,
        "episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate robo1 trained policy")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).parent / "robo1_getup_ppo.zip"),
        help="Path to trained PPO model .zip",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=str(Path(__file__).parent / "robo1.xml"),
        help="Path to robo1.xml model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation trials per pose (default: 5)",
    )
    parser.add_argument(
        "--pose",
        choices=["all", "roll_pos", "roll_neg", "pitch_pos", "pitch_neg"],
        default="all",
        help="Fallen pose to test (default: all)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Launch MuJoCo interactive 3D viewer",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        # Fallback to library pretrained model if available
        fallback = Path(__file__).resolve().parents[2] / "library" / "robo1_getup_ppo.zip"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading policy from: {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    poses = list(FALLEN_POSES.keys()) if args.pose == "all" else [args.pose]

    print("\n" + "=" * 70)
    print(f"{'Pose':<12} | {'Success Rate':<14} | {'Avg Time to Upright':<20} | {'Final Upright':<12}")
    print("=" * 70)

    for p in poses:
        res = evaluate_pose(model, args.xml, p, n_episodes=args.episodes, render=args.render)
        time_str = f"{res['avg_time_to_upright']:.2f} s" if not np.isnan(res['avg_time_to_upright']) else "N/A"
        print(f"{res['pose']:<12} | {res['success_rate']:>6.1f} %        | {time_str:<20} | {res['avg_final_upright']:>8.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
