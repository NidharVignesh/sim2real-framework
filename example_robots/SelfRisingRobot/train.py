"""PPO Training Pipeline for SelfRisingRobot (robo1) with Sim-to-Real Architecture.

Implements the end-to-end Reinforcement Learning for Robotics pipeline referenced from:
https://meta-quantum.today/?p=8881 (MuJoCo + Gymnasium Sim-to-Real Deployment).

Pipeline Phases:
    Phase 1: Modeling
        - MuJoCo digital twin (robo1.xml) with verified inertia, geometry, and motor parameters.
        - Yaw-invariant tilt sensing matching real hardware MPU-6050 complementary filter.
    Phase 2: Training
        - Gymnasium 1.x environment (Robo1GetupEnv) with multi-env parallelization.
        - Behavioral Cloning (BC) pretraining from reference righting trajectories.
        - Sim-to-Real Domain Randomization: link mass, damping, friction, motor strength,
          MPU-6050 sensor noise, and disturbance pushes.
        - Reward shaping diagnostics tracked via TensorBoard.
    Phase 3: Deployment
        - Deterministic ONNX export with numerical verification (PyTorch vs ONNXRuntime).
        - Embedded C header generation (policy_network.h) for ESP32 / Arduino.
        - MicroPython policy generation (policy_network.py).
        - Post-training multi-pose validation benchmark across all 4 initial fallen poses.

Usage:
    # Standard PPO training with domain randomization and automatic ONNX/C export:
    python train.py --timesteps 200000 --n-envs 4 --domain-rand

    # Jumpstart with Behavioral Cloning pretraining + RL fine-tuning:
    python train.py --pretrain --pretrain-epochs 1500 --timesteps 200000 --domain-rand

    # Pretrain only and immediately export to ONNX and ESP32 C header:
    python train.py --pretrain-only --export-onnx --export-c

    # Resume training from existing model checkpoint:
    python train.py --model-in robo1_getup_ppo.zip --timesteps 100000 --domain-rand
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robo1_env import FALLEN_POSES, Robo1GetupEnv


# -----------------------------------------------------------------------------
# Reward Diagnostics Callback (Phase 2: Training Observation & Reward Shaping)
# -----------------------------------------------------------------------------
class RewardDiagnosticsCallback(BaseCallback):
    """Logs individual reward components to TensorBoard for diagnosing reward shaping.

    Referencing https://meta-quantum.today/?p=8881:
    'Key debugging heuristic: if reward graphs look good but real/inferred behavior
    is poor, the problem is almost always in reward architecture, not the training
    algorithm — iterate on reward shaping before touching hyperparameters.'
    """

    def __init__(self, check_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.reward_sums: Dict[str, float] = {}
        self.step_counter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_breakdown" in info:
                for k, v in info["reward_breakdown"].items():
                    self.reward_sums[k] = self.reward_sums.get(k, 0.0) + float(v)
            if "upright" in info:
                self.reward_sums["upright_raw"] = self.reward_sums.get("upright_raw", 0.0) + float(info["upright"])
            self.step_counter += 1

        if self.n_calls % self.check_freq == 0 and self.step_counter > 0:
            for k, total in self.reward_sums.items():
                mean_val = total / self.step_counter
                if k.startswith("reward_") or k.startswith("cost_"):
                    self.logger.record(f"rewards/{k}", mean_val)
                elif k == "total_reward":
                    self.logger.record("rewards/step_total", mean_val)
                elif k == "upright_raw":
                    self.logger.record("metrics/upright_alignment", mean_val)
            self.reward_sums.clear()
            self.step_counter = 0

        return True


# -----------------------------------------------------------------------------
# Environment Factory with Sim-to-Real Domain Randomization
# -----------------------------------------------------------------------------
def make_env(
    xml_path: str,
    domain_randomization: bool = False,
    randomize_pose_offset: bool = False,
    rand_mass_range: Tuple[float, float] = (0.85, 1.15),
    rand_damping_range: Tuple[float, float] = (0.80, 1.20),
    rand_friction_range: Tuple[float, float] = (0.80, 1.20),
    rand_actuator_range: Tuple[float, float] = (0.85, 1.15),
    sensor_noise_std: float = 0.015,
    random_pushes: bool = False,
    rank: int = 0,
    seed: int = 0,
):
    """Build a monitored Gymnasium environment instance with domain randomization settings."""
    def _init():
        env = Robo1GetupEnv(
            xml_path=xml_path,
            randomize_pose_offset=randomize_pose_offset,
            domain_randomization=domain_randomization,
            rand_mass_range=rand_mass_range,
            rand_damping_range=rand_damping_range,
            rand_friction_range=rand_friction_range,
            rand_actuator_range=rand_actuator_range,
            sensor_noise_std=sensor_noise_std,
            random_pushes=random_pushes,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# -----------------------------------------------------------------------------
# Phase 2: Behavioral Cloning Jumpstart Pretraining
# -----------------------------------------------------------------------------
def pretrain_policy_from_scripted(model: PPO, xml_path: str, epochs: int = 1500) -> PPO:
    """Jumpstart policy network weights using behavioral cloning from reference trajectories."""
    from getup_reference import getup_sequence_for_pose

    print(f"\n[Pretraining] Collecting expert demonstration data across 4 fallen poses + holding...")
    observations = []
    actions = []

    # 1. Collect get-up trajectories with diverse initial offsets
    offsets = [(0.0, 0.0), (5.0, -3.0), (-5.0, 4.0)]
    for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
        for offset in offsets:
            env = Robo1GetupEnv(xml_path=xml_path, fallen_poses=(pose,))
            obs, _ = env.reset(options={"pose": pose, "offset": offset})
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

            # Crucial: hold upright posture at [0, 0] for 80 steps
            for _ in range(80):
                targets.append(np.array([0.0, 0.0]))

            for waypoint in targets:
                error = waypoint - env.target
                action = np.clip(error / env.target_delta, -1.0, 1.0).astype(np.float32)
                observations.append(obs.copy())
                actions.append(action.copy())
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()

    # 2. Collect standing-still stability demonstrations (upright pose, actions = [0, 0])
    env = Robo1GetupEnv(xml_path=xml_path)
    for _ in range(3):
        env.data.qpos[:] = 0.0
        env.data.qvel[:] = 0.0
        env.data.qpos[0:3] = [0.0, 0.0, 0.08]
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        env.target[:] = 0.0
        env.data.ctrl[:] = 0.0
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        for _ in range(100):
            mujoco.mj_step(env.model, env.data)
        for _ in range(60):
            obs = env._get_obs()
            action = np.zeros(2, dtype=np.float32)
            observations.append(obs.copy())
            actions.append(action.copy())
            env.step(action)
    env.close()

    # 3. Dynamic Push Knockdown Demonstrations across all compass directions:
    # Teaches the network to sense arbitrary dynamic falls and execute the recovery motion
    push_recovery_directions = [
        ((2.0, 0.0, 0.0), "pitch_pos"),     # Forward push -> recovery
        ((-2.0, 0.0, 0.0), "roll_neg"),     # Backward push -> recovery
        ((0.0, 2.0, 0.0), "roll_neg"),      # Left push -> recovery
        ((0.0, -2.0, 0.0), "pitch_pos"),    # Right push -> recovery
        ((1.4, 1.4, 0.0), "roll_neg"),      # Forward-left push
        ((1.4, -1.4, 0.0), "roll_neg"),     # Forward-right push
        ((-1.4, 1.4, 0.0), "roll_pos"),     # Backward-left push
        ((-1.4, -1.4, 0.0), "pitch_pos"),   # Backward-right push
    ]

    for push_vec, rec_seq_name in push_recovery_directions:
        env = Robo1GetupEnv(xml_path=xml_path)
        env.data.qpos[:] = 0.0
        env.data.qvel[:] = 0.0
        env.data.qpos[0:3] = [0.0, 0.0, 0.08]
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        env.target[:] = 0.0
        env.data.ctrl[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(100):
            mujoco.mj_step(env.model, env.data)
        for _ in range(15):
            env.step([0.0, 0.0])

        env.apply_external_force(push_vec, duration_env_steps=3, body_name="arm2")
        for _ in range(15):
            obs = env._get_obs()
            action = np.zeros(2, dtype=np.float32)
            observations.append(obs.copy())
            actions.append(action.copy())
            env.step(action)

        seq = getup_sequence_for_pose(rec_seq_name)
        targets = []
        prev = env.target.copy()
        for waypoint in seq:
            for i in range(350):
                t = (i + 1) / 350.0
                ctrl = (1.0 - t) * prev + t * waypoint
                if i % env.frame_skip == env.frame_skip - 1:
                    targets.append(ctrl.copy())
            prev = waypoint

        for _ in range(80):
            targets.append(np.array([0.0, 0.0]))

        for wp in targets:
            err = wp - env.target
            action = np.clip(err / env.target_delta, -1.0, 1.0).astype(np.float32)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
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

        if (epoch + 1) % 300 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:4d}/{epochs} | MSE Loss: {loss.item():.6f}")

    print("[Pretraining] Complete!\n")
    return model


# -----------------------------------------------------------------------------
# Phase 3: Deployment — ONNX & Embedded Hardware Export
# -----------------------------------------------------------------------------
class OnnxablePolicy(torch.nn.Module):
    """Wrapper that exposes deterministic action prediction for ONNX export."""

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy.to("cpu")

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Deterministic action mean from actor network
        return self.policy.get_distribution(observation).distribution.mean


def export_policy_to_onnx(
    model: PPO,
    output_path: str,
    obs_dim: int = 4,
    verify: bool = True,
) -> str:
    """Export PPO policy network to ONNX format with numerical verification.

    Directly implements the Phase 3 deployment specification from
    https://meta-quantum.today/?p=8881:
    'torch.onnx.export(policy, dummy_input, "policy.onnx") with careful
    verification that scaling behaves identically post-conversion.'
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    onnxable = OnnxablePolicy(model.policy)
    onnxable.eval()

    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)

    torch.onnx.export(
        onnxable,
        dummy_input,
        str(out_file),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )
    print(f"\n[Sim-to-Real Deployment] Exported ONNX policy to: {out_file} ({out_file.stat().st_size} bytes)")

    if verify:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(out_file))
            test_inputs = np.random.uniform(-math.pi, math.pi, size=(32, obs_dim)).astype(np.float32)
            with torch.no_grad():
                pt_out = onnxable(torch.as_tensor(test_inputs)).numpy()
            onnx_out = sess.run(None, {"observation": test_inputs})[0]
            max_err = float(np.max(np.abs(pt_out - onnx_out)))
            if max_err < 1e-4:
                print(f"[Sim-to-Real Deployment] ONNX Numerical Verification: PASSED (max delta: {max_err:.2e} < 1e-4)")
            else:
                print(f"[Sim-to-Real Deployment] WARNING: ONNX difference ({max_err:.2e}) exceeded threshold!")
        except Exception as e:
            print(f"[Sim-to-Real Deployment] ONNX verification check skipped ({e})")

    return str(out_file)


def export_policy_to_c_header(model_path: str, output_path: str) -> str:
    """Export trained policy as C header (policy_network.h) for ESP32 / Arduino."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from library.simtoreal.converter import convert

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    convert(str(model_path), output_path=str(out_file), lang="c")
    print(f"[Sim-to-Real Deployment] Exported C header for ESP32 to: {out_file}")
    return str(out_file)


def export_policy_to_micropython(model_path: str, output_path: str) -> str:
    """Export trained policy as MicroPython module (policy_network.py)."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from library.simtoreal.converter import convert

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    convert(str(model_path), output_path=str(out_file), lang="python")
    print(f"[Sim-to-Real Deployment] Exported MicroPython module to: {out_file}")
    return str(out_file)


# -----------------------------------------------------------------------------
# Post-Training Quantitative Validation Benchmark
# -----------------------------------------------------------------------------
def run_post_training_evaluation(
    model: PPO,
    xml_path: str,
    n_episodes_per_pose: int = 5,
    render: bool = False,
) -> dict:
    """Run quantitative validation benchmark across all 4 initial fallen poses.

    Directly addresses https://meta-quantum.today/?p=8881:
    'visual and quantitative validation is non-negotiable — a promising graph
    does not guarantee good real-world recovery behavior.'
    """
    print("\n" + "=" * 70)
    print("Post-Training Validation Benchmark across All Fallen Poses")
    print("=" * 70)

    results = {}
    for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
        env = Robo1GetupEnv(
            xml_path=xml_path,
            fallen_poses=(pose,),
            render_mode="human" if render else None,
            randomize_pose_offset=True,
        )
        successes = 0
        getup_times = []
        final_uprights = []

        for _ in range(n_episodes_per_pose):
            obs, _ = env.reset(options={"pose": pose})
            stable_count = 0
            ep_getup_step = None
            upright = 0.0

            for step in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                upright = float(info["upright"])

                if upright > 0.90:
                    if ep_getup_step is None:
                        ep_getup_step = step
                    stable_count += 1
                else:
                    stable_count = 0

                if terminated or stable_count >= 25:
                    successes += 1
                    break

            final_uprights.append(upright)
            if ep_getup_step is not None:
                getup_times.append(ep_getup_step * 0.02)  # 50 Hz control loop -> 0.02s per step

        env.close()

        success_rate = (successes / n_episodes_per_pose) * 100.0
        avg_time = float(np.mean(getup_times)) if getup_times else float("nan")
        mean_upright = float(np.mean(final_uprights))

        results[pose] = {
            "success_rate": success_rate,
            "avg_time_sec": avg_time,
            "mean_upright": mean_upright,
        }

        time_str = f"{avg_time:4.2f}s" if not math.isnan(avg_time) else " N/A "
        print(
            f"  Pose: {pose:10s} | Success Rate: {success_rate:5.1f}% ({successes}/{n_episodes_per_pose}) "
            f"| Avg Stand Time: {time_str} | Final Upright: {mean_upright:.3f}"
        )

    overall_success = float(np.mean([r["success_rate"] for r in results.values()]))
    print("-" * 70)
    print(f"  OVERALL BENCHMARK SUCCESS RATE: {overall_success:5.1f}%")
    print("=" * 70 + "\n")
    return results


# -----------------------------------------------------------------------------
# Main Training Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train PPO policy for SelfRisingRobot (robo1) adhering to Sim-to-Real pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Simulation & Environment options
    parser.add_argument(
        "--xml",
        type=str,
        default=str(Path(__file__).parent / "robo1.xml"),
        help="Path to robo1.xml MuJoCo model",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (SubprocVecEnv)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless rendering backend (MUJOCO_GL=egl or osmesa)",
    )

    # RL & PPO Hyperparameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Steps per environment per PPO rollout update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size for PPO surrogate updates",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Behavioral Cloning Pretraining
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Run behavioral cloning pretraining from reference trajectories before PPO",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=1500,
        help="Number of supervised BC epochs for pretraining",
    )
    parser.add_argument(
        "--pretrain-only",
        action="store_true",
        help="Run behavioral cloning pretraining, export models, and exit without running PPO",
    )

    # Sim-to-Real Domain Randomization
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="Enable full Sim-to-Real Domain Randomization (mass, damping, friction, motor strength, sensor noise)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Alias to enable domain randomization & initial pose angular noise",
    )
    parser.add_argument(
        "--sensor-noise",
        type=float,
        default=0.015,
        help="MPU-6050 sensor noise standard deviation in radians (~0.85 deg)",
    )
    parser.add_argument(
        "--rand-pushes",
        action="store_true",
        help="Apply occasional random disturbance push forces during training",
    )
    parser.add_argument(
        "--rand-mass-range",
        type=float,
        nargs=2,
        default=[0.85, 1.15],
        help="Min and max multiplier for link mass randomization",
    )
    parser.add_argument(
        "--rand-damping-range",
        type=float,
        nargs=2,
        default=[0.80, 1.20],
        help="Min and max multiplier for joint damping & frictionloss",
    )
    parser.add_argument(
        "--rand-actuator-range",
        type=float,
        nargs=2,
        default=[0.85, 1.15],
        help="Min and max multiplier for servo actuator strength (battery sag)",
    )

    # Sim-to-Real Deployment & Export options
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export policy network to ONNX format with numerical verification",
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        default=str(Path(__file__).parent / "robo1_policy.onnx"),
        help="Path for exported ONNX model",
    )
    parser.add_argument(
        "--export-c",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export policy network as C header (policy_network.h) for ESP32/Arduino",
    )
    parser.add_argument(
        "--c-out",
        type=str,
        default=str(Path(__file__).parent / "policy_network.h"),
        help="Path for exported C header file",
    )
    parser.add_argument(
        "--export-mpy",
        action="store_true",
        help="Export policy network as MicroPython module (policy_network.py)",
    )
    parser.add_argument(
        "--mpy-out",
        type=str,
        default=str(Path(__file__).parent / "policy_network.py"),
        help="Path for exported MicroPython file",
    )

    # Post-Training Validation Benchmark
    parser.add_argument(
        "--eval-after-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run post-training multi-pose validation benchmark",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation benchmark episodes per fallen pose",
    )
    parser.add_argument(
        "--eval-render",
        action="store_true",
        help="Open 3D interactive viewer during post-training evaluation",
    )

    # Paths & Logging
    parser.add_argument(
        "--model-in",
        type=str,
        default=None,
        help="Existing model zip checkpoint to resume training from",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(Path(__file__).parent / "robo1_getup_ppo.zip"),
        help="Output path for final trained model zip",
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
        help="Evaluation frequency in timesteps for EvalCallback",
    )

    args = parser.parse_args()

    # Headless rendering configuration (Troubleshooting section of reference site)
    if args.headless or ("DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ):
        os.environ["MUJOCO_GL"] = "egl"
        print("[Rendering] Headless mode enabled (MUJOCO_GL=egl)")

    xml_path = str(Path(args.xml).resolve())
    out_path = Path(args.model_out).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_domain_rand = args.domain_rand or args.randomize

    print("=" * 70)
    print("SelfRisingRobot (robo1) Sim-to-Real PPO Training Pipeline")
    print("Referenced from: https://meta-quantum.today/?p=8881")
    print("=" * 70)
    print(f"Model XML:              {xml_path}")
    print(f"Total Timesteps:        {args.timesteps:,}")
    print(f"Parallel Envs:          {args.n_envs}")
    print(f"Learning Rate:          {args.lr}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Domain Randomization:   {use_domain_rand}")
    if use_domain_rand:
        print(f"  - Mass Range:         {args.rand_mass_range}")
        print(f"  - Damping Range:      {args.rand_damping_range}")
        print(f"  - Actuator Strength:  {args.rand_actuator_range}")
        print(f"  - Sensor Noise Std:   {args.sensor_noise:.4f} rad")
        print(f"  - Random Pushes:      {args.rand_pushes}")
    print(f"Output Checkpoint:      {out_path}")
    print(f"TensorBoard Logs:       {log_dir}")
    print(f"Export ONNX:            {args.export_onnx}")
    print(f"Export C Header:        {args.export_c}")
    print("=" * 70)

    # Build vectorized training environments
    env_kwargs = dict(
        xml_path=xml_path,
        domain_randomization=use_domain_rand,
        randomize_pose_offset=use_domain_rand,
        rand_mass_range=tuple(args.rand_mass_range),
        rand_damping_range=tuple(args.rand_damping_range),
        rand_friction_range=tuple(args.rand_damping_range),
        rand_actuator_range=tuple(args.rand_actuator_range),
        sensor_noise_std=args.sensor_noise if use_domain_rand else 0.0,
        random_pushes=args.rand_pushes if use_domain_rand else False,
    )

    if args.n_envs > 1:
        vec_env = SubprocVecEnv(
            [make_env(rank=i, seed=args.seed, **env_kwargs) for i in range(args.n_envs)]
        )
    else:
        vec_env = DummyVecEnv([make_env(rank=0, seed=args.seed, **env_kwargs)])

    # Separate unrandomized evaluation environment for consistent benchmarking
    eval_env = DummyVecEnv([
        make_env(
            xml_path=xml_path,
            domain_randomization=False,
            randomize_pose_offset=False,
            rank=99,
            seed=args.seed,
        )
    ])

    # Check if tensorboard is available
    tb_log = None
    try:
        import tensorboard  # noqa: F401
        tb_log = str(log_dir)
    except ImportError:
        pass

    # Callbacks
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

    reward_diag_callback = RewardDiagnosticsCallback(check_freq=50)

    # Initialize or load PPO model
    if args.model_in:
        print(f"Loading existing model from: {args.model_in}")
        model = PPO.load(args.model_in, env=vec_env, device="cpu")
    else:
        # 64x64 Tanh MLP policy matches standard embedded deployment network architecture
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        )
        ent_coef = 0.001 if (args.pretrain or args.pretrain_only) else 0.005
        lr = min(args.lr, 1e-4) if args.pretrain else args.lr
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
            device="cpu",
        )

    # Optional behavioral cloning pretraining
    if (args.pretrain or args.pretrain_only) and not args.model_in:
        model = pretrain_policy_from_scripted(model, xml_path=xml_path, epochs=args.pretrain_epochs)

    # If user specified --pretrain-only, save and export immediately
    if args.pretrain_only:
        model.save(str(out_path))
        best_dir = out_path.parent / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(best_dir / "best_model.zip"))
        print(f"[Pretraining] Model saved directly to: {out_path} and {best_dir / 'best_model.zip'}")

        if args.export_onnx:
            export_policy_to_onnx(model, args.onnx_out, obs_dim=4, verify=True)
        if args.export_c:
            export_policy_to_c_header(str(out_path), args.c_out)
        if args.export_mpy:
            export_policy_to_micropython(str(out_path), args.mpy_out)
        if args.eval_after_train:
            run_post_training_evaluation(
                model, xml_path, n_episodes_per_pose=args.eval_episodes, render=args.eval_render
            )

        vec_env.close()
        eval_env.close()
        return

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
            callback=[eval_callback, checkpoint_callback, reward_diag_callback],
            progress_bar=has_progress_bar,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")

    # Save final model
    model.save(str(out_path))
    print(f"\nFinal model checkpoint saved successfully to: {out_path}")

    # Phase 3 Deployment: ONNX, C Header, and MicroPython Exports
    if args.export_onnx:
        export_policy_to_onnx(model, args.onnx_out, obs_dim=4, verify=True)

    if args.export_c:
        export_policy_to_c_header(str(out_path), args.c_out)

    if args.export_mpy:
        export_policy_to_micropython(str(out_path), args.mpy_out)

    # Post-training quantitative validation benchmark
    if args.eval_after_train:
        run_post_training_evaluation(
            model, xml_path, n_episodes_per_pose=args.eval_episodes, render=args.eval_render
        )

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
