# SelfRisingRobot (`robo1`) — PPO Training & Sim-to-Real Workflow

This directory contains the complete simulation, reinforcement learning training pipeline, evaluation suite, and sim-to-real deployment workflow for **Robo1**, a 2-DOF self-righting robot trained using MuJoCo and Stable-Baselines3 (PPO).

---

## Table of Contents
- [Hardware & Model Specifications](#hardware--model-specifications)
- [RL Environment Design](#rl-environment-design)
- [Directory Structure](#directory-structure)
- [Step-by-Step Workflow & Bash Commands](#step-by-step-workflow--bash-commands)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Training with PPO](#2-training-with-ppo)
  - [3. Evaluating & Visualizing the Policy](#3-evaluating--visualizing-the-policy)
  - [4. Sim-to-Real Export (ESP32 Deployment)](#4-sim-to-real-export-esp32-deployment)
- [Physical Deployment on ESP32](#physical-deployment-on-esp32)
- [Troubleshooting & Tips](#troubleshooting--tips)

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

The environment is implemented in [`robo1_env.py`](robo1_env.py) complying with **Gymnasium 1.x**:

- **Observation Space (4D Continuous)**:
  $$\mathbf{o}_t = \left[ \phi_{\text{roll}},\, \theta_{\text{pitch}},\, q_{\text{target}, 1},\, q_{\text{target}, 2} \right]$$
  - $\phi_{\text{roll}}, \theta_{\text{pitch}} \in [-\pi, \pi]$: Roll and pitch of the base foot derived from the MPU-6050 IMU.
  - $q_{\text{target}, 1}, q_{\text{target}, 2} \in [-1.57, 1.57]\,\text{rad}$: Current servo angle setpoints.
  - *Why this matches real hardware*: On the ESP32, the complementary filter computes roll & pitch directly from the MPU-6050, and the servo angles are stored in local variables. No external motion-capture or privileged data is required.

- **Action Space (2D Continuous)**:
  $$\mathbf{a}_t \in [-1, 1]^2 \implies \Delta q_i = \mathbf{a}_i \times 0.08\,\text{rad}$$
  - Actions represent incremental position changes $\Delta q$ executed at $50\,\text{Hz}$ ($20\,\text{ms}$ control step).
  - Maximum rate: $0.08\,\text{rad} / 0.02\,\text{s} \approx 4.0\,\text{rad/s}$ ($229^\circ/\text{s}$), strictly matching the physical SG90 servo speed limit ($0.12\,\text{s} / 60^\circ \approx 500^\circ/\text{s}$ under no load, safely de-rated).

- **Initial Fallen Configurations**:
  On every episode reset, the robot is spawned randomly into one of 4 challenging initial poses:
  1. `roll_pos`: Tilted $+90^\circ$ onto its right flank.
  2. `roll_neg`: Tilted $-90^\circ$ onto its left flank.
  3. `pitch_pos`: Fallen forward face-down.
  4. `pitch_neg`: Fallen backward on its back.

- **Reward Formulation**:
  $$R_t = R_{\text{upright}} + R_{\text{height}} + R_{\text{stability}} - R_{\text{action\_penalty}}$$
  - Upright alignment: $\mathbf{z}_{\text{base}} \cdot \mathbf{z}_{\text{world}} = \cos(\text{tilt})$.
  - Upright height bonus: awards points when base elevation $z > 0.045\,\text{m}$.
  - Stability bonus: granted when upright with low angular velocities ($|\omega| < 0.5\,\text{rad/s}$).
  - Smoothness penalty: penalizes large action jerks to protect real servo gears.

---

## Directory Structure

```
example_robots/SelfRisingRobot/
├── assets/                  # 3D mesh STL files (foot, arm1, arm2, servo, esp32, MPU_6050, battery)
├── robo1.xml                # MuJoCo physics model with sensors, collisions, and masses
├── robo1_env.py             # Gymnasium environment (Robo1GetupEnv)
├── getup_reference.py       # Reference kinematic trajectories for behavioral cloning
├── train.py                 # Multi-core PPO training script (Stable-Baselines3)
├── eval_policy.py           # Evaluation benchmark & 3D MuJoCo visualizer
├── robo1_getup_ppo.zip      # Trained PPO policy checkpoint
├── checkpoints/             # Periodic model checkpoints saved during training
├── best_model/              # Best policy saved by EvalCallback
├── logs/                    # Monitor CSV training logs
└── README.md                # This workflow guide
```

---

## Step-by-Step Workflow & Bash Commands

### 1. Environment Setup

Activate the project's Python virtual environment and navigate to the robot directory:

```bash
# Activate virtual environment
source "/home/nidharshan/Documents/5th_sem/Embedded system project/sim2real-framework/venv/bin/activate"

# Change into the robot workspace
cd "/home/nidharshan/Documents/5th_sem/Embedded system project/sim2real-framework/example_robots/SelfRisingRobot"
```

Verify that the MuJoCo model loads without errors:

```bash
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('robo1.xml'); print(f'Successfully loaded robo1.xml! Total mass: {m.body_mass.sum():.4f} kg')"
```

---

### 2. Training with PPO

#### A. Standard Training from Scratch
Train an agent across 4 parallel environments for 200,000 steps:

```bash
python train.py --timesteps 200000 --n-envs 4
```

#### B. Jumpstart with Behavioral Cloning Pretraining (Recommended)
Pre-trains the neural network weights on reference geometric righting trajectories before running PPO reinforcement learning. This accelerates convergence significantly:

```bash
python train.py --pretrain --pretrain-epochs 2000 --timesteps 200000 --n-envs 4
```

#### C. Resuming / Fine-Tuning an Existing Model
Load an existing `.zip` model checkpoint and continue training for additional steps:

```bash
python train.py --model-in robo1_getup_ppo.zip --timesteps 100000 --n-envs 4
```

#### Available Training CLI Arguments:
| Argument | Default | Description |
|---|---|---|
| `--timesteps` | `200000` | Total environment steps to train |
| `--n-envs` | `4` | Number of parallel worker environments (`SubprocVecEnv`) |
| `--lr` | `3e-4` | Learning rate for Adam optimizer |
| `--batch-size` | `64` | Minibatch size for PPO surrogate updates |
| `--model-in` | `None` | Path to existing `.zip` model to resume training from |
| `--model-out` | `robo1_getup_ppo.zip` | File path to save the final trained model |
| `--pretrain` | `False` | Run behavioral cloning pretraining before RL |
| `--eval-freq` | `10000` | Frequency (in steps) to evaluate policy and save best model |

---

### 3. Evaluating & Visualizing the Policy

#### A. Headless Benchmark across All 4 Fallen Poses
Run quantitative evaluation across 20 trials per pose (`roll_pos`, `roll_neg`, `pitch_pos`, `pitch_neg`):

```bash
python eval_policy.py --model robo1_getup_ppo.zip --episodes 20
```

*Output summary report includes:*
- Success rate percentage per initial pose.
- Average time (in seconds) to achieve upright posture.
- Final pitch/roll settling error.

#### B. Interactive 3D MuJoCo Viewer
Watch the robot execute the policy in real time:

```bash
# Visualize all 4 fallen poses sequentially
python eval_policy.py --model robo1_getup_ppo.zip --render

# Visualize only a specific pose (e.g., negative roll fall)
python eval_policy.py --model robo1_getup_ppo.zip --pose roll_neg --render --episodes 5
```

---

### 4. Sim-to-Real Export (ESP32 Deployment)

The trained policy can be exported directly to embedded formats using the framework converter.

#### Option A: Export to Pure C Header (`policy_network.h`)
For deployment using Arduino IDE or ESP-IDF (no Python runtime required on microcontroller):

```bash
python -c "
import sys
sys.path.insert(0, '../../')
from library.simtoreal.converter import convert

convert('robo1_getup_ppo.zip', output_path='policy_network.h', lang='c')
print('Exported policy_network.h successfully!')
"
```

#### Option B: Export to MicroPython (`policy_network.py`)
For microcontrollers running MicroPython firmware:

```bash
python -c "
import sys
sys.path.insert(0, '../../')
from library.simtoreal.converter import convert

convert('robo1_getup_ppo.zip', output_path='policy_network.py', lang='python')
print('Exported policy_network.py successfully!')
"
```

---

## Physical Deployment on ESP32

When flashing to the ESP32:

1. **Sensor Loop ($50\,\text{Hz} = 20\,\text{ms}$ tick)**:
   - Read MPU-6050 accelerometer ($a_x, a_y, a_z$) and gyroscope ($\omega_x, \omega_y, \omega_z$).
   - Compute roll $\phi$ and pitch $\theta$ using standard complementary filter:
     $$\theta_t = \alpha (\theta_{t-1} + \omega_y \Delta t) + (1 - \alpha) \arctan2(a_x, \sqrt{a_y^2 + a_z^2})$$
2. **Policy Inference**:
   - Construct observation vector: `[roll, pitch, current_servo1_rad, current_servo2_rad]`.
   - Call `policy_forward(obs, action)`.
3. **Actuator Update**:
   - Compute new target: $q_i \leftarrow q_i + \text{action}[i] \times 0.08\,\text{rad}$.
   - Clamp to limits $[-\pi/2, \pi/2]$.
   - Convert radians to PWM microseconds ($1000\,\mu\text{s} - 2000\,\mu\text{s}$) and send to SG90 servos.

---

## Troubleshooting & Tips

- **Headless server / SSH**: If running on a remote machine without a display, do not use `--render`. Evaluation will run headlessly via software stepping.
- **Multiprocessing warning**: If running on a system with fewer than 4 CPU cores, reduce `--n-envs` (e.g. `--n-envs 2`).
- **Policy divergence**: If training from scratch diverges, use `--pretrain` to give the network an initial behavioral prior from the reference trajectories.
