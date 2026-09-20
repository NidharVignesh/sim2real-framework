"""Evaluation and visualization script for trained robo1 policies.

Evaluates trained PPO models across fallen poses (roll_pos, roll_neg, pitch_pos, pitch_neg)
and tests robot resilience against external perturbation forces in MuJoCo.

Features:
    - Multi-pose get-up benchmark (success rate, time-to-upright, final posture)
    - External force knockdown & recovery benchmark (--test-push)
    - Periodic automated perturbation testing (--push-force, --push-interval)
    - Interactive 3D viewer with live keyboard force controls (--render):
        [F] / [Up Arrow]    : Push Forward (+X)
        [B] / [Down Arrow]  : Push Backward (-X)
        [L] / [Left Arrow]  : Push Left (+Y)
        [R] / [Right Arrow] : Push Right (-Y)
        [Space] / [K]       : Hard Knockdown impulse
        [P]                 : Random Direction Push
        [+] / [-]           : Increase / Decrease Force magnitude
        [Mouse Drag]        : Right-click & drag on robot in MuJoCo viewer

Usage:
    # Evaluate across all fallen poses
    python eval_policy.py --model robo1_getup_ppo.zip

    # Interactive 3D evaluation with live keyboard push controls
    python eval_policy.py --model robo1_getup_ppo.zip --render

    # Knockdown recovery test: apply external forces to knock robot over and verify recovery
    python eval_policy.py --model robo1_getup_ppo.zip --test-push --render

    # Headless push recovery benchmark
    python eval_policy.py --model robo1_getup_ppo.zip --test-push --push-force 2.0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
from stable_baselines3 import PPO

from robo1_env import FALLEN_POSES, Robo1GetupEnv


def evaluate_pose(
    model: PPO,
    xml_path: str,
    pose: str,
    n_episodes: int = 5,
    render: bool = False,
    push_interval: int = 0,
    push_force: float = 2.0,
    push_dir: str = "random",
) -> dict:
    """Evaluate policy performance from a specific fallen pose, optionally applying periodic pushes."""
    env = Robo1GetupEnv(
        xml_path=xml_path,
        fallen_poses=(pose,),
        render_mode="human" if render else None,
        randomize_pose_offset=True,
    )

    successes = 0
    upright_times = []
    final_uprights = []

    for ep in range(n_episodes):
        obs, _ = env.reset(options={"pose": pose})
        ep_upright_step = None
        stable_steps = 0

        for step in range(env.max_steps):
            # Optional automated periodic pushes
            if push_interval > 0 and step > 0 and step % push_interval == 0:
                fx, fy = _get_push_vector(push_dir, push_force)
                env.apply_external_force([fx, fy, 0.0], duration_env_steps=2)
                if render:
                    print(f"  [Step {step}] Periodic push applied: ({fx:+.1f} N, {fy:+.1f} N)")

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                time.sleep(1.0 / 50.0)

            upright = info["upright"]
            if upright > 0.90:
                if ep_upright_step is None:
                    ep_upright_step = step
                stable_steps += 1
            else:
                stable_steps = 0

            # Success condition: terminated by env OR maintained upright for >= 25 steps
            if terminated or stable_steps >= 25:
                successes += 1
                break
            if truncated:
                break

        final_uprights.append(info["upright"])
        if ep_upright_step is not None:
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


def _get_push_vector(direction: str, force: float) -> Tuple[float, float]:
    """Calculate (Fx, Fy) force components for a given direction."""
    if direction == "forward":
        return force, 0.0
    elif direction == "backward":
        return -force, 0.0
    elif direction == "left":
        return 0.0, force
    elif direction == "right":
        return 0.0, -force
    else:  # random
        theta = np.random.uniform(0, 2 * math.pi)
        return force * math.cos(theta), force * math.sin(theta)


def test_push_knockdown_recovery(
    model: PPO,
    xml_path: str,
    push_force: float = 2.0,
    render: bool = False,
) -> List[dict]:
    """Test policy recovery after being knocked down by external forces from upright standing."""
    directions = [
        ("Forward (+X)", (push_force, 0.0, 0.0)),
        ("Backward (-X)", (-push_force, 0.0, 0.0)),
        ("Left (+Y)", (0.0, push_force, 0.0)),
        ("Right (-Y)", (0.0, -push_force, 0.0)),
        ("Oblique (Random)", None),
    ]

    results = []
    print("\n" + "=" * 78)
    print(f"External Force Knockdown & Recovery Benchmark (Force: {push_force:.1f} N)")
    print("=" * 78)
    print(f"{'Direction':<18} | {'Toppled?':<10} | {'Recovered?':<12} | {'Time to Recover':<18} | {'Final Upright':<12}")
    print("-" * 78)

    for name, force_vec in directions:
        if force_vec is None:
            theta = np.random.uniform(0, 2 * math.pi)
            force_vec = (push_force * math.cos(theta), push_force * math.sin(theta), 0.0)

        env = Robo1GetupEnv(xml_path=xml_path, render_mode="human" if render else None)

        # 1. Initialize robot in upright standing pose
        env.data.qpos[:] = 0.0
        env.data.qvel[:] = 0.0
        env.data.qpos[0:3] = [0.0, 0.0, 0.08]
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        env.target[:] = 0.0
        env.data.ctrl[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(100):
            mujoco.mj_step(env.model, env.data)

        # Let robot settle standing for a few steps
        obs = env._get_obs()
        for _ in range(25):
            obs, _, _, _, _ = env.step([0.0, 0.0])
            if render:
                time.sleep(1.0 / 50.0)

        # 2. Apply external knockdown force perturbation to the upper body
        env.apply_external_force(force_vec, duration_env_steps=3, body_name="arm2")
        obs, _, _, _, info = env.step([0.0, 0.0])
        if render:
            time.sleep(1.0 / 50.0)

        # Allow physics to carry the fall for 15 steps (0.3s)
        toppled = False
        for _ in range(15):
            obs, _, _, _, info = env.step([0.0, 0.0])
            if info["upright"] < 0.4:
                toppled = True
            if render:
                time.sleep(1.0 / 50.0)

        fall_upright = info["upright"]
        if fall_upright < 0.5:
            toppled = True

        # 3. Give policy control to recover from the fall
        recovered = False
        rec_step = None
        stable_count = 0

        for step in range(350):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                time.sleep(1.0 / 50.0)

            if info["upright"] > 0.90:
                if rec_step is None:
                    rec_step = step
                stable_count += 1
            else:
                stable_count = 0

            if terminated or stable_count >= 25:
                recovered = True
                break

        time_to_rec = (rec_step * 0.02) if rec_step is not None else float("nan")
        final_upright = info["upright"]
        env.close()

        res = {
            "direction": name,
            "toppled": toppled,
            "recovered": recovered,
            "time_to_recover": time_to_rec,
            "final_upright": final_upright,
        }
        results.append(res)

        time_str = f"{time_to_rec:.2f} s" if not np.isnan(time_to_rec) else "N/A"
        print(
            f"{name:<18} | {'YES' if toppled else 'RESISTED':<10} | "
            f"{'YES' if recovered else 'NO':<12} | {time_str:<18} | {final_upright:>8.4f}"
        )

    print("=" * 78 + "\n")
    return results


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
        help="Launch MuJoCo interactive 3D viewer (with live keyboard push controls)",
    )
    parser.add_argument(
        "--test-push",
        action="store_true",
        help="Run dedicated external knockdown force and recovery test suite",
    )
    parser.add_argument(
        "--push-force",
        type=float,
        default=2.0,
        help="External perturbation force magnitude in Newtons (default: 2.0 N)",
    )
    parser.add_argument(
        "--push-interval",
        type=int,
        default=0,
        help="Steps between automated periodic pushes during standard eval (0 = disabled)",
    )
    parser.add_argument(
        "--push-dir",
        choices=["random", "forward", "backward", "left", "right"],
        default="random",
        help="Direction for automated periodic pushes (default: random)",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        fallback = Path(__file__).resolve().parents[2] / "library" / "robo1_getup_ppo.zip"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading policy from: {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    if args.render:
        print("\nInteractive Viewer Mode Enabled:")
        print("Use your keyboard inside the MuJoCo viewer window to push the robot:")
        print("  [F] / [Up Arrow]    : Push Forward (+X)")
        print("  [B] / [Down Arrow]  : Push Backward (-X)")
        print("  [L] / [Left Arrow]  : Push Left (+Y)")
        print("  [R] / [Right Arrow] : Push Right (-Y)")
        print("  [Space] / [K]       : Hard Knockdown impulse (topple robot)")
        print("  [P]                 : Random Direction Push")
        print("  [+] / [-]           : Increase / Decrease Force magnitude")
        print("  [Mouse Drag]        : Right-click & drag on robot anywhere\n")

    # Run dedicated push knockdown recovery test if requested
    if args.test_push:
        test_push_knockdown_recovery(model, args.xml, push_force=args.push_force, render=args.render)
        return

    # Otherwise run multi-pose getup evaluation
    poses = list(FALLEN_POSES.keys()) if args.pose == "all" else [args.pose]

    print("\n" + "=" * 70)
    print(f"{'Pose':<12} | {'Success Rate':<14} | {'Avg Time to Upright':<20} | {'Final Upright':<12}")
    print("=" * 70)

    for p in poses:
        res = evaluate_pose(
            model,
            args.xml,
            p,
            n_episodes=args.episodes,
            render=args.render,
            push_interval=args.push_interval,
            push_force=args.push_force,
            push_dir=args.push_dir,
        )
        time_str = f"{res['avg_time_to_upright']:.2f} s" if not np.isnan(res['avg_time_to_upright']) else "N/A"
        print(f"{res['pose']:<12} | {res['success_rate']:>6.1f} %        | {time_str:<20} | {res['avg_final_upright']:>8.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
