# simtoreal — 1 minute 55 second video

Faster presentation cut: short animated explanations, male neural narration, and recorded simulation and hardware demonstrations.

## 0:00–0:08 — From simulation to a real robot

> A robot can learn in simulation. But getting that learned controller onto a small board still takes manual coding.

## 0:08–0:16 — Meet simtoreal

> Our simtoreal library bridges that gap. Give it a trained policy and a configuration describing the robot's hardware.

## 0:16–0:24 — Keep the learned controller

> It extracts the decision network from a trained PPO model, keeping the weights needed to turn sensor readings into movements.

## 0:24–0:33 — Three export formats

> Export a C header, Python source, or compiled MicroPython bytecode. The robot runs the learned controller without the training software.

## 0:33–0:43 — Describe the hardware

> The configuration defines observations, actions, driver classes, and pins. The library checks input and output counts and generates the hardware setup.

## 0:43–0:52 — Generate the complete loop

> The generated main dot py reads sensors, runs the policy, and drives the motors. Copy it, the policy, and drivers onto the ESP thirty two.

## 0:52–0:56 — The example robot

> Here's our two-joint self-rising robot.

## 0:56–1:12 — Recorded simulation

> In this recorded simulation, the robot learns to recover from a fallen position. Five inputs describe its orientation and joint targets. Two outputs control the servo movements.

## 1:12–1:24 — Physical trial

> Now watch the physical robot. The ESP thirty two uses its tilt sensor and two servos to move the body toward upright.

## 1:24–1:39 — Longer hardware trial

> This longer trial is shown at one and a half times speed. The robot keeps adjusting after the initial rise. Hardware testing shows what needs to improve.

## 1:39–1:50 — Test and improve

> After testing, retrain and export again. Reuse the hardware configuration when the wiring and policy interface stay the same.

## 1:50–1:55 — simtoreal

> simtoreal. Learn in simulation. Run on real hardware.

## Editing notes

- 1080p, 30 fps; assembled and exported in Drift.
- Keep simulation and the first physical trial at normal playback speed.
- Label the second physical trial as 1.5× playback.
- Narration is synthetic: Kokoro `am_michael`, with quicker delivery and reduced repetition.
- The final exported video and portable project are in `library_video/`.
