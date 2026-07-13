# Sim-to-Real Deployment Framework for RL-based Robots
 
## Overview
 
This project builds a reusable framework that automatically converts a robot policy trained in simulation (MuJoCo + PPO) into working code deployable on real hardware (ESP32) — replacing the current manual conversion and deployment process.
 
The framework is scoped to: **MuJoCo (simulation) + PPO/Stable-Baselines3 (training) + ESP32-class microcontroller (real hardware)**.
 
The open-source [SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot) project is used as the example case to build, test, and validate the framework.
 
---
 
## Problem
 
A robot is trained in a simulator like MuJoCo, not on the real robot. Once training is done, that trained policy has to be moved onto the real robot. Right now there is no automatic way to do this move.
 
Moving a trained policy to real hardware is currently a manual process. Someone has to hand-convert the trained model into code the microcontroller can run, and hand-write the deployment code for that one robot. This has to be redone from scratch for every new robot.
 
## Current Solution
 
The SelfRisingRobot project shows this manual process. A robot is trained in MuJoCo with PPO, then the trained model is manually converted into a C file and manually written into Arduino code for an ESP32/M5Atom chip. This works, but only for this one robot, and only by doing everything by hand. To the best of our knowledge, no existing framework automates this conversion and deployment process for MuJoCo/PPO-trained policies going to ESP32 hardware.
 
## Feasibility
 
- The framework converts simulation code into working real-world code, as long as the simulation's requirements are met.
- The scope is narrowed to only what SelfRisingRobot uses: MuJoCo, PPO, and ESP32/M5Atom.
- SelfRisingRobot will be built and used as the working test case for the framework.
- Because the framework only targets this one tool combination, not every possible robot or simulator, the workload stays realistic for one semester.
---

#Planned repo structure

```
sim2real-framework/
├── configs/
│   └── robo1.yaml                  # the single source-of-truth config
├── generators/
│   ├── yaml_to_mjcf.py             # builds MJCF from YAML
│   └── yaml_to_arduino.py          # builds Arduino skeleton from YAML
├── templates/
│   ├── mjcf_template.xml           # base MJCF structure with placeholders
│   └── arduino_template.ino        # base Arduino structure with placeholders
├── training/
│   ├── env_builder.py              # builds Gymnasium env from generated MJCF
│   └── train.py                    # PPO training script (generic, config-driven)
├── export/
│   └── export_policy_header.py     # generic policy → C header exporter
├── assets/
│   └── robo1/                      # STL mesh files
├── evaluation/
│   ├── eval_sim.py                 # time-to-upright + failure rate in sim
│   └── eval_real.py                # same metrics logged from real hardware (serial input)
└── README.md
```
 
## Proposed Plan
 
A single YAML config file describes a robot (joints, actuators, sensor, hardware pins). The framework reads this file and generates everything needed for both simulation and real hardware from it, so the robot only has to be defined once.
 
```
                        robot_config.yaml
                              |
              -----------------------------------
              |                                 |
      yaml_to_mjcf.py                   yaml_to_arduino.py
              |                                 |
      MuJoCo model (.xml)              Arduino sketch skeleton
              |
      Gymnasium environment
              |
      PPO training (Stable-Baselines3)
              |
      Trained policy (.zip)
              |
      Generic policy exporter
              |
      C header file (policy weights)
              |
      Generic C++ inference runner  ---->  Flash to ESP32
                                                  |
                                          Run on real robot
                                                  |
                                    Log time-to-upright + failure rate
                                                  |
                                    Compare against simulation results
```
 
The key idea: instead of hand-writing the MJCF model and the Arduino code separately for every robot, both are generated from the same config file, so updating the robot only means updating the YAML.
 
## Current Planned Approach
 
1. Build and run the existing SelfRisingRobot project as-is, to fully understand its MuJoCo model, training script, and Arduino deployment code.
2. Identify every value in that project that is hardcoded but could instead come from a config file (joint ranges, servo pins, sensor type, actuator limits).
3. Design the YAML schema based on those values.
4. Build the generator scripts that turn a YAML file into a working MJCF model and a working Arduino sketch.
5. Build a generic policy exporter and a generic inference runner, so any trained policy (not just this one robot's) can be converted and deployed.
6. Add sim-to-real robustness options (domain randomization, actuator delay modeling) into the training step, as configurable options in the framework.
7. Test the full framework end-to-end using SelfRisingRobot as the example robot, and compare its results (time-to-upright, failure rate) against the original hand-built version.

## Weekly Task Checklist

### Week 1 — Environment Setup & Baseline Understanding
- [x] Install MuJoCo, Gymnasium, Stable-Baselines3, Arduino IDE
- [x] Clone and explore the SelfRisingRobot repository
- [ ] Run `play_robo1_policy.py` and observe the trained policy in the MuJoCo viewer
- [x] Read through `robo1.xml` and identify bodies, joints, actuators, sensors
- [x] Read through `robo1_env.py` and identify observation space, action space, reward function
- [ ] Prepare and deliver Week 1 demo/walkthrough to evaluator

### Week 2 — Hardware Build
- [x] 3D print robot parts (foot, arm1, arm2, arm horn)
- [x] Assemble servos and ESP32
- [ ] Verify hardware powers on and servos respond to basic test commands
- [ ] Document real hardware specs (servo model, sensor model, pin mapping)

### Week 3 — Reproduce Baseline Simulation
- [ ] Confirm MuJoCo model loads and runs correctly
- [ ] Confirm Gymnasium environment wraps the model correctly
- [ ] Run evaluation script (`eval_robo1_policy.py`) and record baseline metrics

### Week 4 — Retrain Baseline Policy
- [ ] Run `train_robo1.py` and retrain PPO policy from scratch
- [ ] Confirm training converges and policy rights the robot in simulation
- [ ] Compare self-trained policy performance to the provided pretrained policy

### Week 5 — Baseline Hardware Deployment
- [ ] Export trained policy to C header (`export_policy_header.py`)
- [ ] Flash exported policy + Arduino sketch to real hardware
- [ ] Test baseline policy on real robot
- [ ] Document manual steps taken (this becomes the basis for what the framework automates)

### Week 6 — Evaluation Protocol Design
- [ ] Define fixed set of starting poses/conditions for testing
- [ ] Build logging for time-to-upright and success/failure rate
- [ ] Run baseline trials in simulation, log results

### Week 7 — Baseline Real-World Evaluation
- [ ] Run matched baseline trials on real hardware, log results
- [ ] Produce baseline sim-vs-real comparison table (time-to-upright, failure rate)

### Week 8 — Framework: Config Schema & MJCF Generator
- [ ] Design YAML config schema (joints, actuators, sensors, world settings)
- [ ] Build `yaml_to_mjcf.py` generator
- [ ] Validate generated MJCF matches original `robo1.xml` behavior

### Week 9 — Framework: Arduino Generator & Generic Inference Runner
- [ ] Build `yaml_to_arduino.py` generator
- [ ] Build generic C++ inference runner (works for any exported network shape)
- [ ] Build generic policy exporter (any SB3 PPO model → C header)

### Week 10 — Framework: Sim2Real Robustness Options
- [ ] Add domain randomization option (mass, friction, servo gain)
- [ ] Add actuator delay modeling option
- [ ] Retrain policy using framework with robustness options enabled

### Week 11 — Framework: Automated Pipeline & Deployment
- [ ] Build single-command pipeline (train → export → flash → test)
- [ ] Deploy framework-generated policy to real hardware
- [ ] Run same evaluation protocol as Weeks 6–7 on the new policy

### Week 12 — Grounding Step & Final Evaluation
- [ ] Collect real robot trajectories, compare to sim rollouts under same actions
- [ ] Manually adjust MuJoCo parameters to reduce mismatch (grounding)
- [ ] Run final full evaluation (sim + real) on the fully improved model

### Week 13 — Report Writing
- [ ] Compile before/after comparison table and plots
- [ ] Write up problem, method, results, discussion, limitations

### Week 14 — Final Presentation
- [ ] Prepare final presentation/demo
- [ ] Buffer time for any outstanding issues

---

## Deliverable

A reusable framework that automatically converts a MuJoCo/PPO-trained policy into deployable code for ESP32-based servo-driven robots, demonstrated and validated on the SelfRisingRobot robot, with a measured before/after reduction in the sim-to-real performance gap (time-to-upright and failure rate).
