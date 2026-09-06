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
    ):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_pose_offset = randomize_pose_offset

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

    def _get_obs(self) -> np.ndarray:
        """Compute the 4D observation vector [roll, pitch, target1, target2]."""
        roll, pitch = quat_to_roll_pitch(self.data.qpos[3:7])
        return np.array(
            [
                roll,
                pitch,
                self.target[0],
                self.target[1],
            ],
            dtype=np.float32,
        )

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

        pose_name = None
        offset_deg = (0.0, 0.0)
        if options is not None:
            pose_name = options.get("pose")
            offset_deg = options.get("offset", (0.0, 0.0))

        if pose_name is None:
            pose_name = self.np_random.choice(self.fallen_poses)

        if self.randomize_pose_offset and options is None:
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
        self.target += action * self.target_delta
        self.target = np.clip(self.target, -self.target_limit, self.target_limit)

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

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
        tilt_cost = 0.2 * (roll * roll + pitch * pitch)
        stand_gate = float(np.clip((upright - 0.65) / 0.35, 0.0, 1.0))
        servo_zero_cost = stand_gate * 0.5 * float(np.sum(servo_angles * servo_angles))
        target_zero_cost = stand_gate * 0.2 * float(np.sum(self.target * self.target))
        height_cost = stand_gate * 5.0 * (foot_height_error * foot_height_error)
        body_vel_cost = 0.01 * float(np.sum(self.data.qvel[0:6] ** 2))
        servo_vel_cost = 0.005 * float(np.sum(servo_velocities ** 2))
        action_cost = 0.005 * float(np.sum(action ** 2))
        servo_motion_cost = 0.02 * float(np.sum((self.target - old_target) ** 2))

        reward = (
            2.0 * upright
            + 8.0 * upright_progress
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
            upright > 0.95
            and abs(roll) < 0.25
            and abs(pitch) < 0.25
            and np.max(np.abs(servo_angles)) < 0.12
            and abs(foot_height_error) < 0.02
            and np.linalg.norm(self.data.qvel[0:6]) < 0.25
        )

        if goal_pose:
            self.success_count += 1
            reward += 5.0
        else:
            self.success_count = 0

        terminated = self.success_count >= 50
        truncated = self.step_count >= self.max_steps

        info = {
            "pose": self.current_pose_name,
            "upright": upright,
            "foot_height_error": foot_height_error,
            "servo_angles": servo_angles.copy(),
            "goal_pose": goal_pose,
            "target": self.target.copy(),
            "success_count": self.success_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data, camera="front")
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
