"""Gymnasium environment for the SelfRisingRobot (robo1) get-up task.

Observation (5,) float32 — only what the real ESP32 can measure:
    [0] roll     MPU-6050 tilt about body X, atan2(ay, sqrt(ax^2+az^2))   rad
    [1] pitch    MPU-6050 tilt about body Y, atan2(-ax, sqrt(ay^2+az^2))  rad
    [2] az       MPU-6050 Z acceleration in g (+1 upright, -1 upside down)
    [3] target1  current servo1 set-point                      [-1.55, 1.55] rad
    [4] target2  current servo2 set-point                      [-1.55, 1.55] rad
    roll/pitch alone read ~0 both upright and upside down; az disambiguates.

Action (2,) float32 in [-1, 1]:
    Servo set-point increment, target += action * 0.08 rad, at 50 Hz.

Start states (every reset draws one):
    "roll_pos", "roll_neg", "pitch_pos", "pitch_neg"  the four canonical falls
    "random"   any orientation (incl. upside down), any yaw, random servo angles
    "upright"  already standing, so the policy also learns to hold still

An episode succeeds (terminated=True) once the goal pose is held for
HOLD_STEPS consecutive control steps. It is truncated after max_steps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
START_CASES = (*FALLEN_POSES, "random", "upright")
# Sampling weights for START_CASES during training (must sum to 1).
DEFAULT_CASE_WEIGHTS = (0.15, 0.15, 0.15, 0.15, 0.30, 0.10)

HOLD_STEPS = 40  # 0.8 s at 50 Hz


def euler_to_quat(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """ZYX Euler angles -> quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


class Robo1GetupEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        cases: Tuple[str, ...] = START_CASES,
        case_weights: Optional[Tuple[float, ...]] = None,
        domain_randomization: bool = False,
        sensor_noise_std: float = 0.015,
    ):
        super().__init__()
        unknown = set(cases) - set(START_CASES)
        if unknown:
            raise ValueError(f"Unknown start cases {unknown}; choose from {START_CASES}")
        self.render_mode = render_mode
        self.cases = tuple(cases)
        if case_weights is None:
            if self.cases == START_CASES:
                case_weights = DEFAULT_CASE_WEIGHTS
            else:
                case_weights = (1.0,) * len(self.cases)
        w = np.asarray(case_weights, dtype=np.float64)
        self.case_probs = w / w.sum()
        self.domain_randomization = domain_randomization
        self.sensor_noise_std = sensor_noise_std

        if xml_path is None:
            xml_path = str(Path(__file__).parent / "robo1.xml")
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        self.data = mujoco.MjData(self.model)

        # 1 ms physics x 20 = 50 Hz control loop (matches the ESP32 loop).
        self.frame_skip = 20
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = 700  # 14 s
        self.target_delta = 0.08
        self.target_limit = 1.55
        self.settle_steps = 500

        self.foot_id = self.model.body("foot").id
        j1, j2 = self.model.joint("servo1_joint"), self.model.joint("servo2_joint")
        self.servo_qpos = np.array([j1.qposadr[0], j2.qposadr[0]])
        self.servo_qvel = np.array([j1.dofadr[0], j2.dofadr[0]])
        # Robot geoms that collide with the floor (used to place the robot on the ground).
        self._robot_geoms = np.where(
            (self.model.geom_bodyid != 0) & (self.model.geom_contype != 0)
        )[0]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        lim = self.target_limit
        self.observation_space = spaces.Box(
            low=np.array([-math.pi, -math.pi, -1.5, -lim, -lim], dtype=np.float32),
            high=np.array([math.pi, math.pi, 1.5, lim, lim], dtype=np.float32),
            dtype=np.float32,
        )

        self._nominal = {
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "dof_frictionloss": self.model.dof_frictionloss.copy(),
            "geom_friction": self.model.geom_friction.copy(),
        }
        self.actuator_scale = 1.0

        self.target = np.zeros(2)
        self.step_count = 0
        self.hold_count = 0
        self.prev_upright = 0.0
        self.case = self.cases[0]
        self._viewer = None

        self.standing_height = self._measure_standing_height()

    # ------------------------------------------------------------------ state
    def _place_on_ground(self, quat: np.ndarray, servo: np.ndarray):
        """Set orientation and servo angles, lift the robot so it touches the floor, let it settle."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[3:7] = quat
        self.data.qpos[self.servo_qpos] = servo
        self.target[:] = servo
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] += 0.005 - self._lowest_point()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

    def _lowest_point(self) -> float:
        """Lowest world z over the corners of every robot geom's bounding box."""
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        lowest = np.inf
        for g in self._robot_geoms:
            center, half = self.model.geom_aabb[g, :3], self.model.geom_aabb[g, 3:]
            corners = center + signs * half
            world = self.data.geom_xpos[g] + corners @ self.data.geom_xmat[g].reshape(3, 3).T
            lowest = min(lowest, world[:, 2].min())
        return float(lowest)

    def _measure_standing_height(self) -> float:
        self._place_on_ground(np.array([1.0, 0, 0, 0]), np.zeros(2))
        return float(self.data.xpos[self.foot_id, 2])

    def _tilt(self) -> Tuple[float, float]:
        """Yaw-invariant roll/pitch from the gravity vector in the body frame (as the IMU sees it)."""
        xmat = self.data.xmat[self.foot_id].reshape(3, 3)
        # Row 2 of R = world Z in the body frame. Signs keep the original roll/pitch convention.
        ax, ay, az = -xmat[2, 0], -xmat[2, 1], xmat[2, 2]
        roll = math.atan2(-ay, math.hypot(ax, az))
        pitch = math.atan2(ax, math.hypot(ay, az))
        return roll, pitch

    def _upright(self) -> float:
        """cos(tilt): 1 upright, 0 on its side, -1 upside down."""
        return float(self.data.xmat[self.foot_id, 8])

    def _get_obs(self) -> np.ndarray:
        roll, pitch = self._tilt()
        az = self._upright()
        if self.domain_randomization and self.sensor_noise_std > 0:
            roll, pitch, az = np.array([roll, pitch, az]) + self.np_random.normal(
                0, self.sensor_noise_std, size=3
            )
        return np.array([roll, pitch, az, *self.target], dtype=np.float32)

    def _randomize_physics(self):
        n = self._nominal
        m = self.model
        u = self.np_random.uniform
        mass_scale = u(0.85, 1.15, size=m.nbody)
        m.body_mass[:] = n["body_mass"] * mass_scale
        m.body_inertia[:] = n["body_inertia"] * mass_scale[:, None]
        m.dof_damping[:] = n["dof_damping"] * u(0.8, 1.2, size=m.nv)
        m.dof_frictionloss[:] = n["dof_frictionloss"] * u(0.8, 1.2, size=m.nv)
        m.geom_friction[:] = n["geom_friction"]
        m.geom_friction[:, 0] *= u(0.8, 1.2)
        self.actuator_scale = float(u(0.85, 1.15))  # battery sag -> slower servos

    def _restore_physics(self):
        for name, value in self._nominal.items():
            getattr(self.model, name)[:] = value
        self.actuator_scale = 1.0

    # ------------------------------------------------------------ gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """options: {"case": one of START_CASES} forces the start case."""
        super().reset(seed=seed)
        options = options or {}
        case = options.get("case") or str(self.np_random.choice(self.cases, p=self.case_probs))
        if case not in START_CASES:
            raise ValueError(f"Unknown case {case!r}")
        self.case = case

        if self.domain_randomization:
            self._randomize_physics()
        else:
            self._restore_physics()

        rng = self.np_random
        yaw = rng.uniform(-math.pi, math.pi)
        if case in FALLEN_POSES:
            roll, pitch = FALLEN_POSES[case]
            roll += math.radians(rng.uniform(-10, 10))
            pitch += math.radians(rng.uniform(-10, 10))
            quat, servo = euler_to_quat(roll, pitch, yaw), np.zeros(2)
        elif case == "random":
            # Uniformly random rotation, so every fall direction (and upside down) is covered.
            q = rng.normal(size=4)
            quat = q / np.linalg.norm(q)
            servo = rng.uniform(-self.target_limit, self.target_limit, size=2)
        else:  # upright
            quat, servo = euler_to_quat(0.0, 0.0, yaw), np.zeros(2)

        self._place_on_ground(quat, servo)
        self.step_count = 0
        self.hold_count = 0
        self.prev_upright = self._upright()
        return self._get_obs(), {"case": case}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        old_target = self.target.copy()
        self.target = np.clip(
            self.target + action * self.target_delta * self.actuator_scale,
            -self.target_limit,
            self.target_limit,
        )
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        roll, pitch = self._tilt()
        upright = self._upright()
        progress = upright - self.prev_upright
        self.prev_upright = upright
        servo = self.data.qpos[self.servo_qpos]
        servo_vel = self.data.qvel[self.servo_qvel]
        height_err = float(self.data.xpos[self.foot_id, 2]) - self.standing_height

        # stand_gate: 0 while fallen, 1 when upright -> "stand still" costs only apply once up.
        gate = float(np.clip((upright - 0.65) / 0.35, 0.0, 1.0))
        reward = (
            3.0 * upright
            + 8.0 * progress
            - 0.2 * (roll**2 + pitch**2)
            - gate * 0.5 * float(np.sum(servo**2))
            - gate * 0.2 * float(np.sum(self.target**2))
            - gate * 5.0 * height_err**2
            - gate * 0.01 * float(np.sum(self.data.qvel[0:6] ** 2))
            - gate * 0.005 * float(np.sum(servo_vel**2))
            - gate * 0.02 * float(np.sum((self.target - old_target) ** 2))
            - 0.001 * float(np.sum(action**2))
        )

        goal = (
            upright > 0.92
            and abs(roll) < 0.35
            and abs(pitch) < 0.35
            and float(np.max(np.abs(servo))) < 0.25
            and abs(height_err) < 0.04
        )
        if goal:
            self.hold_count += 1
            reward += 3.0
        else:
            self.hold_count = 0

        terminated = self.hold_count >= HOLD_STEPS
        truncated = self.step_count >= self.max_steps
        info = {"case": self.case, "upright": upright, "goal_pose": goal, "is_success": terminated}

        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
