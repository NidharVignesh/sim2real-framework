# Sim-to-Real Deployment Framework for RL-based Robots

## Overview

This project builds a reusable framework that automatically converts a robot policy trained in simulation (MuJoCo + PPO) into working code deployable on real hardware (ESP32/M5Atom) — replacing the current manual conversion and deployment process.

The framework is scoped to: **MuJoCo (simulation) + PPO/Stable-Baselines3 (training) + ESP32-class microcontrollers (real hardware)**.

The open-source [SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot) project is used as the example case to build, test, and validate the framework.

---

## Problem

When a robot policy is trained using reinforcement learning, it is trained inside a simulator like MuJoCo, not on the real robot directly. Once training is done, that policy needs to be moved onto the real robot so it can actually control the servos and sensors. But right now, there is no automatic or reusable way to do this move.

Currently, taking a trained simulation policy and getting it to run on real hardware is a manual process. Someone has to hand-convert the trained model into a format the microcontroller (like Arduino/ESP32) can understand, and then hand-write the deployment code specifically for that one robot. This means every time someone builds a new robot or updates the training, they have to repeat this manual conversion and deployment work all over again from scratch.

## Current Solution

An example of this manual process can be seen in the SelfRisingRobot project. A small robot is trained in MuJoCo using PPO, and once training is finished, the trained model is manually converted into a C code file and manually written into Arduino code to run on an ESP32. This works for that one specific robot, but the conversion and deployment steps are all done by hand, with no reusable tool or framework behind it. General tools exist for deploying neural networks to microcontrollers, and individual sim-to-real techniques exist in research literature, but to the best of our knowledge, no existing framework combines these specifically for MuJoCo-trained PPO policies being deployed to ESP32-based robots.

## Feasibility

- Building a framework that converts simulation code into working real-world code, as long as the requirements set in the simulation are fulfilled.
- To make this feasible, the framework is narrowed to only what is used in the SelfRisingRobot project: MuJoCo for simulation, PPO for training, and ESP32/M5Atom as the hardware controller.
- For the example and testing, this exact open-source project will be built, so there is a real, working case to develop and prove the framework on.
- Since the framework only needs to work for this one specific tool combination, and not for every possible robot or simulator, the amount of work stays realistic and achievable within one semester.

---

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
