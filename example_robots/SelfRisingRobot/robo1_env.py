"""Gymnasium Environment for SelfRisingRobot (robo1).

Wraps MuJoCo simulation of robo1.xml for training self-righting policies
using Reinforcement Learning (PPO).

Observation Space:
    [0] base_roll:   Roll angle of foot base [-pi, pi]
    [1] base_pitch:  Pitch angle of foot base [-pi, pi]
    [2] servo1_cmd:  Current target angle of servo1 [-1.55, 1.55] rad
    [3] servo2_cmd:  Current target angle of servo2 [-1.55, 1.55] rad

Action Space:
    [0] delta_servo1: Normalized rate of change for servo1 [-1.0, 1.0]
    [1] delta_servo2: Normalized rate of change for servo2 [-1.0, 1.0]
    Mapped to target angle updates: target += action * 0.08 rad (at 50 Hz).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

FALLEN_POSES = {
    "roll_pos": (math.pi / 2.0, 0.0),
    "roll_neg": (-math.pi / 2.0, 0.0),
    "pitch_pos": (0.0, math.pi / 2.0),
    "pitch_neg": (0.0, -math.pi / 2.0),
}
DEFAULT_FALLEN_POSES = tuple(FALLEN_POSES.keys())


def roll_pitch_to_quat(roll: float, pitch: float) -> np.ndarray:
    """Convert Euler roll and pitch (yaw=0) to quaternion [w, x, y, z]."""
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)


def quat_to_roll_pitch(q: np.ndarray) -> Tuple[float, float]:
    """Convert quaternion [w, x, y, z] to roll and pitch angles in radians."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


class Robo1GetupEnv(gym.Env):
    """Gymnasium environment for robo1 self-rising task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        fallen_poses: Optional[Tuple[str, ...]] = None,
        render_mode: Optional[str] = None,
        randomize_pose_offset: bool = False,
        domain_randomization: bool = False,
        rand_mass_range: Tuple[float, float] = (0.85, 1.15),
        rand_damping_range: Tuple[float, float] = (0.80, 1.20),
        rand_friction_range: Tuple[float, float] = (0.80, 1.20),
        rand_actuator_range: Tuple[float, float] = (0.85, 1.15),
        sensor_noise_std: float = 0.015,
        random_pushes: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_pose_offset = randomize_pose_offset
        self.domain_randomization = domain_randomization
        self.rand_mass_range = rand_mass_range
        self.rand_damping_range = rand_damping_range
        self.rand_friction_range = rand_friction_range
        self.rand_actuator_range = rand_actuator_range
        self.sensor_noise_std = sensor_noise_std
        self.random_pushes = random_pushes

        # Resolve XML path relative to this file if not specified
        if xml_path is None:
            xml_path = str(Path(__file__).parent / "robo1.xml")
        self.xml_path = str(Path(xml_path).resolve())

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Simulation timing parameters
        # timestep = 0.001s, frame_skip = 20 -> 50 Hz control loop (matches real SG90 PWM)
        self.frame_skip = 20
        self.max_steps = 700
        self.target_delta = 0.08  # max change per 50Hz step (rad)
        self.target_limit = 1.55   # servo joint limit (~pi/2 rad)
        self.settle_steps = 200    # simulation steps to settle on the floor

        self.fallen_poses = list(fallen_poses or DEFAULT_FALLEN_POSES)
        self.current_pose_name = self.fallen_poses[0]

        # Model IDs for fast state lookups
        self.foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot"
        )
        self.servo1_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo1_joint")
        ]
        self.servo2_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo2_joint")
        ]
        self.servo1_qvel_id = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo1_joint")
        ]
        self.servo2_qvel_id = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo2_joint")
        ]

        self.arm2_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "arm2"
        )
        self.applied_force = np.zeros(3, dtype=np.float64)
        self.applied_force_steps = 0
        self.applied_force_body_id = self.arm2_body_id
        self.interactive_push_magnitude = 2.0  # Newtons

        self.standing_height = self._compute_standing_height()

        # Action space: normalized deltas in [-1, 1] for [servo1, servo2]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: [roll, pitch, servo1_target, servo2_target]
        self.observation_space = spaces.Box(
            low=np.array([-math.pi, -math.pi, -1.55, -1.55], dtype=np.float32),
            high=np.array([math.pi, math.pi, 1.55, 1.55], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_count = 0
        self.target = np.zeros(2, dtype=np.float64)
        self.success_count = 0
        self.prev_upright = 0.0
        self._viewer = None

        # Store nominal physics parameters for domain randomization (sim-to-real)
        self.nominal_body_mass = self.model.body_mass.copy()
        self.nominal_body_inertia = self.model.body_inertia.copy()
        self.nominal_dof_damping = self.model.dof_damping.copy()
        self.nominal_dof_frictionloss = self.model.dof_frictionloss.copy()
        self.nominal_geom_friction = self.model.geom_friction.copy()
        self.actuator_strength_scale = 1.0

    def _compute_standing_height(self) -> float:
        """Measure the resting foot height when the robot is standing upright."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = [0.0, 0.0, 0.08]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[self.servo1_qpos_id] = 0.0
        data.qpos[self.servo2_qpos_id] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, data)

        for _ in range(self.settle_steps):
            data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, data)

        return float(data.xpos[self.foot_body_id, 2])

    def _set_fallen_pose(self, pose_name: str, offset_deg: Tuple[float, float] = (0.0, 0.0)):
        """Initialize robot in a fallen resting pose on the ground plane."""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, 0.08]

        base_roll, base_pitch = FALLEN_POSES[pose_name]
        roll = base_roll + math.radians(offset_deg[0])
        pitch = base_pitch + math.radians(offset_deg[1])

        self.data.qpos[3:7] = roll_pitch_to_quat(roll, pitch)
        self.data.qpos[self.servo1_qpos_id] = 0.0
        self.data.qpos[self.servo2_qpos_id] = 0.0

        self.target[:] = 0.0
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)

        # Allow physics to settle onto ground
        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        self.target[:] = 0.0
        self.data.ctrl[:] = self.target

    def _compute_orientation_tilt(self) -> Tuple[float, float]:
        """Compute continuous, yaw-invariant roll and pitch tilt angles matching MPU6050.
        
        Uses the projection of world upward vertical (opposite of gravity) onto the
        robot base's body frame axes. This eliminates Euler angle gimbal-lock flips
        and accurately models the microcontroller's accelerometer/complementary filter.
        """
        xmat = self.data.xmat[self.foot_body_id].reshape(3, 3)
        ax, ay, az = float(xmat[0, 2]), float(xmat[1, 2]), float(xmat[2, 2])
        # Aligned convention:
        # roll: tilt about body X axis (positive tilts right)
        roll = math.atan2(-ay, math.sqrt(ax * ax + az * az))
        # pitch: tilt about body Y axis (positive tilts forward)
        pitch = math.atan2(ax, math.sqrt(ay * ay + az * az))
        return roll, pitch

    def _get_obs(self) -> np.ndarray:
        """Compute the 4D observation vector [roll, pitch, target1, target2]."""
        roll, pitch = self._compute_orientation_tilt()
        if self.domain_randomization and self.sensor_noise_std > 0.0:
            # Sim-to-Real: Add Gaussian noise matching physical MPU-6050 noise & complementary filter drift
            noise = self.np_random.normal(0.0, self.sensor_noise_std, size=2)
            roll = float(np.clip(roll + noise[0], -math.pi, math.pi))
            pitch = float(np.clip(pitch + noise[1], -math.pi, math.pi))

        return np.array(
            [
                roll,
                pitch,
                self.target[0],
                self.target[1],
            ],
            dtype=np.float32,
        )

    def apply_external_force(
        self,
        force_xyz: Tuple[float, float, float] | np.ndarray,
        duration_env_steps: int = 1,
        body_name: str = "arm2",
    ):
        """Apply external 3D perturbation force (Newtons) to a robot body.
        
        Args:
            force_xyz: Force vector [Fx, Fy, Fz] in Newtons (world frame).
            duration_env_steps: Number of environment steps (at 50 Hz) to apply the force.
            body_name: Name of target body ('arm2' for head/top body, 'foot' for base).
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            body_id = self.arm2_body_id
        self.applied_force_body_id = body_id
        self.applied_force = np.asarray(force_xyz, dtype=np.float64)
        self.applied_force_steps = max(1, duration_env_steps)

    def _upright(self) -> float:
        """Measure uprightness from Z component of base orientation matrix."""
        xmat = self.data.xmat[self.foot_body_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _foot_height(self) -> float:
        return float(self.data.xpos[self.foot_body_id, 2])

    def _servo_angles(self) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self.servo1_qpos_id],
                self.data.qpos[self.servo2_qpos_id],
            ],
            dtype=np.float64,
        )

    def _servo_velocities(self) -> np.ndarray:
        return np.array(
            [
                self.data.qvel[self.servo1_qvel_id],
                self.data.qvel[self.servo2_qvel_id],
            ],
            dtype=np.float64,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.success_count = 0
        self.applied_force_steps = 0
        self.applied_force[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        pose_name = None
        offset_deg = (0.0, 0.0)
        if options is not None:
            pose_name = options.get("pose")
            offset_deg = options.get("offset", (0.0, 0.0))

        if pose_name is None:
            pose_name = self.np_random.choice(self.fallen_poses)

        # Sim-to-Real Domain Randomization (Meta-Quantum / Kevin Wood RL Robotics pipeline)
        if self.domain_randomization and options is None:
            # 1. Randomize link mass and inertia
            mass_scale = self.np_random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1], size=self.model.nbody
            )
            self.model.body_mass[:] = self.nominal_body_mass * mass_scale
            self.model.body_inertia[:] = self.nominal_body_inertia * mass_scale[:, None]

            # 2. Randomize joint damping and friction loss
            damp_scale = self.np_random.uniform(
                self.rand_damping_range[0], self.rand_damping_range[1], size=self.model.nv
            )
            self.model.dof_damping[:] = self.nominal_dof_damping * damp_scale

            fric_scale = self.np_random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1], size=self.model.nv
            )
            self.model.dof_frictionloss[:] = self.nominal_dof_frictionloss * fric_scale

            # 3. Randomize ground contact friction
            fric_contact_scale = float(
                self.np_random.uniform(self.rand_friction_range[0], self.rand_friction_range[1])
            )
            self.model.geom_friction[:, 0] = self.nominal_geom_friction[:, 0] * fric_contact_scale

            # 4. Randomize servo actuator strength scale (simulating battery voltage drop)
            self.actuator_strength_scale = float(
                self.np_random.uniform(self.rand_actuator_range[0], self.rand_actuator_range[1])
            )
        else:
            # Restore nominal parameters
            self.model.body_mass[:] = self.nominal_body_mass
            self.model.body_inertia[:] = self.nominal_body_inertia
            self.model.dof_damping[:] = self.nominal_dof_damping
            self.model.dof_frictionloss[:] = self.nominal_dof_frictionloss
            self.model.geom_friction[:] = self.nominal_geom_friction
            self.actuator_strength_scale = 1.0

        if (self.randomize_pose_offset or self.domain_randomization) and options is None:
            offset_deg = (
                float(self.np_random.uniform(-10.0, 10.0)),
                float(self.np_random.uniform(-10.0, 10.0)),
            )

        self.current_pose_name = str(pose_name)
        self._set_fallen_pose(self.current_pose_name, offset_deg)
        self.prev_upright = self._upright()

        return self._get_obs(), {"pose": self.current_pose_name}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        old_target = self.target.copy()
        effective_delta = self.target_delta * self.actuator_strength_scale
        self.target += action * effective_delta
        self.target = np.clip(self.target, -self.target_limit, self.target_limit)

        # Disturbance push perturbations during training (sim-to-real transfer)
        if self.random_pushes and self.np_random.random() < 0.02 and self.applied_force_steps == 0:
            push_angle = self.np_random.uniform(0, 2 * math.pi)
            push_mag = self.np_random.uniform(0.5, 1.8)
            fx = push_mag * math.cos(push_angle)
            fy = push_mag * math.sin(push_angle)
            self.apply_external_force([fx, fy, 0.0], duration_env_steps=2)

        for _ in range(self.frame_skip):
            if self.applied_force_steps > 0:
                self.data.xfrc_applied[self.applied_force_body_id, :3] = self.applied_force
            else:
                self.data.xfrc_applied[self.applied_force_body_id, :3] = 0.0

            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        if self.applied_force_steps > 0:
            self.applied_force_steps -= 1
            if self.applied_force_steps == 0:
                self.data.xfrc_applied[self.applied_force_body_id, :3] = 0.0

        self.step_count += 1
        obs = self._get_obs()

        upright = self._upright()
        upright_progress = upright - self.prev_upright
        self.prev_upright = upright
        roll, pitch = float(obs[0]), float(obs[1])
        servo_angles = self._servo_angles()
        servo_velocities = self._servo_velocities()
        foot_height_error = self._foot_height() - self.standing_height

        # Reward formulation
        # stand_gate smoothly transitions from 0 (fallen) to 1 (standing upright)
        stand_gate = float(np.clip((upright - 0.65) / 0.35, 0.0, 1.0))
        tilt_cost = 0.2 * (roll * roll + pitch * pitch)
        servo_zero_cost = stand_gate * 0.5 * float(np.sum(servo_angles * servo_angles))
        target_zero_cost = stand_gate * 0.2 * float(np.sum(self.target * self.target))
        height_cost = stand_gate * 5.0 * (foot_height_error * foot_height_error)
        body_vel_cost = stand_gate * 0.01 * float(np.sum(self.data.qvel[0:6] ** 2))
        servo_vel_cost = stand_gate * 0.005 * float(np.sum(servo_velocities ** 2))
        action_cost = 0.001 * float(np.sum(action ** 2))
        servo_motion_cost = stand_gate * 0.02 * float(np.sum((self.target - old_target) ** 2))

        upright_reward = 3.0 * upright
        progress_reward = 8.0 * upright_progress

        reward = (
            upright_reward
            + progress_reward
            - tilt_cost
            - servo_zero_cost
            - target_zero_cost
            - height_cost
            - body_vel_cost
            - servo_vel_cost
            - action_cost
            - servo_motion_cost
        )

        goal_pose = (
            upright > 0.92
            and abs(roll) < 0.35
            and abs(pitch) < 0.35
            and np.max(np.abs(servo_angles)) < 0.25
            and abs(foot_height_error) < 0.04
        )

        goal_bonus = 3.0 if goal_pose else 0.0
        if goal_pose:
            self.success_count += 1
            reward += goal_bonus
        else:
            self.success_count = 0

        terminated = self.success_count >= 40
        truncated = self.step_count >= self.max_steps

        reward_breakdown = {
            "reward_upright": float(upright_reward),
            "reward_progress": float(progress_reward),
            "reward_goal": float(goal_bonus),
            "cost_tilt": float(tilt_cost),
            "cost_servo_zero": float(servo_zero_cost),
            "cost_target_zero": float(target_zero_cost),
            "cost_height": float(height_cost),
            "cost_body_vel": float(body_vel_cost),
            "cost_servo_vel": float(servo_vel_cost),
            "cost_action": float(action_cost),
            "cost_servo_motion": float(servo_motion_cost),
            "total_reward": float(reward),
        }

        info = {
            "pose": self.current_pose_name,
            "upright": upright,
            "foot_height_error": foot_height_error,
            "servo_angles": servo_angles.copy(),
            "goal_pose": goal_pose,
            "target": self.target.copy(),
            "success_count": self.success_count,
            "reward_breakdown": reward_breakdown,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _print_push_help(self):
        print("\n" + "=" * 60)
        print("  MuJoCo Viewer Interactive Force Perturbation Controls:")
        print("    [F] / [Up Arrow]    : Push Forward (+X)")
        print("    [B] / [Down Arrow]  : Push Backward (-X)")
        print("    [L] / [Left Arrow]  : Push Left (+Y)")
        print("    [R] / [Right Arrow] : Push Right (-Y)")
        print("    [Space] / [K]       : Hard Knockdown (topple the robot)")
        print("    [P]                 : Random Direction Push")
        print(f"    [+] / [-]           : Adjust Force (Current: {self.interactive_push_magnitude:.1f} N)")
        print("    [H]                 : Show this help menu")
        print("    [Mouse Drag]        : Right-click & drag on robot in viewer")
        print("=" * 60 + "\n")

    def _on_key(self, keycode: int):
        """Handle keyboard push perturbations inside the MuJoCo viewer window."""
        mag = self.interactive_push_magnitude
        if keycode in (ord('f'), ord('F'), 265):
            self.apply_external_force([mag, 0.0, 0.0], duration_env_steps=2)
            print(f"\n>>> [Force Perturbation] Pushed FORWARD (+X) with {mag:.1f} N!")
        elif keycode in (ord('b'), ord('B'), 264):
            self.apply_external_force([-mag, 0.0, 0.0], duration_env_steps=2)
            print(f"\n>>> [Force Perturbation] Pushed BACKWARD (-X) with {mag:.1f} N!")
        elif keycode in (ord('l'), ord('L'), 263):
            self.apply_external_force([0.0, mag, 0.0], duration_env_steps=2)
            print(f"\n>>> [Force Perturbation] Pushed LEFT (+Y) with {mag:.1f} N!")
        elif keycode in (ord('r'), ord('R'), 262):
            self.apply_external_force([0.0, -mag, 0.0], duration_env_steps=2)
            print(f"\n>>> [Force Perturbation] Pushed RIGHT (-Y) with {mag:.1f} N!")
        elif keycode in (ord('k'), ord('K'), 32):
            knock_force = max(2.5, mag * 1.5)
            theta = np.random.uniform(0, 2 * math.pi)
            fx = knock_force * math.cos(theta)
            fy = knock_force * math.sin(theta)
            self.apply_external_force([fx, fy, 0.0], duration_env_steps=3)
            print(f"\n>>> [Force Perturbation] HARD KNOCKDOWN ({knock_force:.1f} N) applied to topple robot!")
        elif keycode in (ord('p'), ord('P')):
            theta = np.random.uniform(0, 2 * math.pi)
            fx = mag * math.cos(theta)
            fy = mag * math.sin(theta)
            self.apply_external_force([fx, fy, 0.0], duration_env_steps=2)
            print(f"\n>>> [Force Perturbation] Random push ({mag:.1f} N) applied!")
        elif keycode in (ord('+'), ord('=')):
            self.interactive_push_magnitude = min(10.0, self.interactive_push_magnitude + 0.5)
            print(f"\n>>> [Config] Push force increased to: {self.interactive_push_magnitude:.1f} N")
        elif keycode in (ord('-'), ord('_')):
            self.interactive_push_magnitude = max(0.5, self.interactive_push_magnitude - 0.5)
            print(f"\n>>> [Config] Push force decreased to: {self.interactive_push_magnitude:.1f} N")
        elif keycode in (ord('h'), ord('H')):
            self._print_push_help()

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, key_callback=self._on_key
                )
                self._print_push_help()
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data, camera="front")
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
