# SelfRisingRobot (`robo1`) — PPO Training & Sim-to-Real Workflow

MuJoCo simulation, PPO training (Stable-Baselines3), benchmark and viewer for **Robo1**, a 2-DOF self-righting robot.

---

## Table of Contents
- [Hardware & Model Specifications](#hardware--model-specifications)
- [RL Environment Design](#rl-environment-design)
- [Directory Structure](#directory-structure)
- [Workflow & Commands](#workflow--commands)
  - [1. Setup](#1-setup)
  - [2. Training](#2-training)
  - [3. Benchmark](#3-benchmark)
  - [4. Play in the MuJoCo Viewer](#4-play-in-the-mujoco-viewer)
  - [5. Export for the ESP32](#5-export-for-the-esp32)
- [Physical Deployment on ESP32](#physical-deployment-on-esp32)
- [Troubleshooting](#troubleshooting)

---

## Hardware & Model Specifications

The MuJoCo model (`robo1.xml`) accurately replicates the physical robot components:

| Component | Part / Spec | Mass | Placement & Notes |
|---|---|---|---|
| **Base Foot** | 3D Printed Link (`foot.stl`) | 37.5 g | Base contact link on ground |
| **Microcontroller** | ESP32 NodeMCU (`esp32.STL`) | 10.0 g | Mounted **flat/horizontal** on foot (`euler="1.5707963 0 0"`) |
| **IMU Sensor** | MPU-6050 breakout (`MPU_6050.stl`) | 2.1 g | Centered on foot with accelerometer + gyro sites |
| **Batteries** | 2× 3.7V LiPo cells (`battery.stl`) | 14.0 g (2× 7g) | Symmetrically mounted on the foot |
| **Joint 1 Servo** | SG90 Micro Servo (`servo1.stl`) | 9.0 g | Roll axis actuator mounted on foot ($\pm 90^\circ$) |
| **Arm 1** | 3D Printed Link (`arm1.stl`) | 10.6 g | Carries Joint 2 servo |
| **Joint 2 Servo** | SG90 Micro Servo (`servo2.stl`) | 9.0 g | Pitch axis actuator on Arm 1 ($\pm 90^\circ$) |
| **Arm 2** | 3D Printed Arm Tip (`arm2.stl`) | 21.6 g | Terminal lever arm for pushing against ground |
| **Total Mass** | — | **~94.8 g** | Verified center of mass & inertial matrix |

> [!NOTE]
> `foot.stl`, `arm1.stl`, `arm2.stl`, and `servo*.stl` are modeled in **meters**. `MPU_6050.stl` and `esp32.STL` are scaled by `0.001` to match SI units in MuJoCo.

---

## RL Environment Design

Implemented in [`robo1_env.py`](robo1_env.py) (Gymnasium 1.x).

**Observation (5 values)**: only what the ESP32 can measure.

| # | Name | Meaning | Unit |
|---|---|---|---|
| 0 | `roll` | `atan2(ay, sqrt(ax² + az²))` from the MPU-6050 | rad |
| 1 | `pitch` | `atan2(-ax, sqrt(ay² + az²))` from the MPU-6050 | rad |
| 2 | `az` | MPU-6050 Z acceleration (+1 upright, −1 upside down) | g |
| 3 | `target1` | current servo1 set-point, [−1.55, 1.55] | rad |
| 4 | `target2` | current servo2 set-point, [−1.55, 1.55] | rad |

Roll and pitch alone read ≈0 both when upright and when upside down; `az` tells them apart.

**Action (2 values in [−1, 1])**: servo set-point increment, `target += action × 0.08 rad`, at 50 Hz (20 ms per step).

**Start cases**: every reset draws one of:

| Case | Share | Description |
|---|---|---|
| `roll_pos`, `roll_neg`, `pitch_pos`, `pitch_neg` | 15% each | The 4 canonical falls (side, side, front, back) with ±10° noise and random yaw |
| `random` | 30% | Any orientation, including upside down, with random servo angles |
| `upright` | 10% | Already standing, so the policy also learns to hold still |

The robot is placed so its lowest point touches the floor, then settles for 0.5 s before the episode starts.

**Reward**: uprightness + progress toward upright − tilt. Once the robot is nearly upright, it is also penalized for servo angles away from 0, height error, body/servo velocity and servo motion. It gets a +3 bonus per step in the goal pose.

**Success**: goal pose (upright > 0.92, |roll|, |pitch| < 0.35 rad, servos within 0.25 rad of 0, correct height) held for 40 steps (0.8 s). Episodes are cut off after 700 steps (14 s).

**Domain randomization** (on by default in training): body mass/inertia ±15%, joint damping and friction ±20%, ground friction ±20%, servo speed ±15% (battery sag), IMU noise σ = 0.015.

---

## Directory Structure

```
example_robots/SelfRisingRobot/
├── assets/                  # STL meshes
├── robo1.xml                # MuJoCo model
├── robo1_env.py             # Gymnasium environment (Robo1GetupEnv)
├── train.py                 # PPO training (Stable-Baselines3)
├── eval_policy.py           # Benchmark: success rate & time-to-upright per case
├── play.py                  # Trained policy in the MuJoCo viewer
└── README.md

# Created by training:
├── robo1_getup_ppo.zip      # Final model
├── best_model/best_model.zip  # Best model by eval success
├── checkpoints/             # Every 250k steps
└── logs/                    # TensorBoard + evaluations.npz
```

---

## Workflow & Commands

### 1. Setup

```bash
cd "/home/nidharshan/Documents/5th_sem/Embedded system project/sim2real-framework/example_robots/SelfRisingRobot"
source ../../venv/bin/activate

# Optional: TensorBoard for training curves
pip install tensorboard
```

### 2. Training

```bash
# Full training: 2M steps, 8 parallel envs, domain randomization on
python train.py 2>&1 | tee train.log

# Or run in the background (keeps going if the terminal closes)
nohup python train.py > train.log 2>&1 &
tail -f train.log

# Watch curves (second terminal): eval/success_rate and rollout/success_rate should climb toward 1.0
tensorboard --logdir logs

# Continue training an existing model
python train.py --resume robo1_getup_ppo.zip --timesteps 1000000
```

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | `2000000` | Total environment steps |
| `--n-envs` | `8` | Parallel environments (`SubprocVecEnv`); lower it on machines with fewer cores |
| `--seed` | `0` | Random seed |
| `--domain-rand` / `--no-domain-rand` | on | Physics + IMU noise randomization |
| `--resume` | — | Model `.zip` to continue training |
| `--out` | `robo1_getup_ppo` | Output path (`.zip` is added) |
| `--log-dir` | `logs` | TensorBoard / eval log directory |

PPO settings (in `train.py`): `lr=3e-4`, `n_steps=1024`, `batch_size=256`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, 64×64 tanh MLP. The network is kept small so it fits on the ESP32.

Press Ctrl+C to stop early; the current model is still saved.

### 3. Benchmark

```bash
python eval_policy.py                                     # robo1_getup_ppo.zip, 50 episodes per case
python eval_policy.py --domain-rand                       # robustness on randomized physics
python eval_policy.py --model best_model/best_model.zip   # best model saved during training
python eval_policy.py --cases random pitch_neg --episodes 100
python eval_policy.py --render --episodes 3               # watch the benchmark
python eval_policy.py --json results.json                 # save results
```

For each case it prints: success rate, time-to-upright (mean / median / std / max, successful episodes only) and mean final uprightness. The last row, "all falls", is the total over every case except `upright`.

### 4. Play in the MuJoCo Viewer

```bash
python play.py                                   # random start case
python play.py --case upright                    # start standing
python play.py --model best_model/best_model.zip
```

The policy runs continuously. Knock the robot over with **Ctrl + right-drag** on a body and it gets back up.

### 5. Export for the ESP32

Uses the framework converter in `library/`:

```bash
# C header (Arduino / ESP-IDF)
python -c "import sys; sys.path.insert(0, '../../'); from library.simtoreal.converter import convert; convert('robo1_getup_ppo.zip', output_path='policy_network.h', lang='c')"

# MicroPython
python -c "import sys; sys.path.insert(0, '../../'); from library.simtoreal.converter import convert; convert('robo1_getup_ppo.zip', output_path='policy_network.py', lang='python')"
```

---

## Physical Deployment on ESP32

Every 20 ms (50 Hz):

1. **Read the MPU-6050**: compute roll and pitch with the same formulas as in the observation table, **converted to radians** (`library/simtoreal/sensors/imu.py` returns degrees). Read `az` in g.
2. **Build the observation**: `[roll, pitch, az, target1, target2]`.
3. **Run the policy**: `action = policy(obs)`.
4. **Update the servos**: `target_i = clamp(target_i + action_i × 0.08, −1.55, 1.55)`, then convert radians to SG90 PWM.

---

## Troubleshooting

- **Fewer CPU cores**: use `--n-envs 4` (or 2).
- **Headless machine / SSH**: training and `eval_policy.py` without `--render` need no display.
- **Old models won't load / shape error**: models trained before the 5-value observation (4 inputs) are incompatible. Retrain.
- **Low success on one case**: run `eval_policy.py --cases <case> --render` to watch it, then continue training with `--resume`.
