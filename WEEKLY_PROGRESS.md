

## 1. This Week Progress Breakdown

### .1 Problem 1: Sensor Value Fluctuation & Solution: Calibration Function
* **Identified Symptom:** During hardware testing with the ESP32, raw MPU-6050 accelerometer and gyroscope data exhibited high-frequency fluctuations, zero-offset bias, and drift over time. Because the RL policy takes tilt angles directly as state observations (`obs0 = roll`, `obs1 = pitch`), raw sensor instability caused servo jitter and erratic control decisions.
* **Implemented Solution:**
  1. **Static Calibration Routine:** Added offset computation to measure and eliminate static sensor zero-bias when the robot is at rest.
  2. **Trigonometric Tilt Estimation:** Implemented pitch and roll calculations using accelerometer projection across planes:
     $$\text{tilt}_x = \text{atan2}(y, \sqrt{x^2 + z^2}) \cdot \frac{180}{\pi}$$
     $$\text{tilt}_y = \text{atan2}(-x, \sqrt{y^2 + z^2}) \cdot \frac{180}{\pi}$$
  3. **Complementary Filter Integration:** Blended high-frequency gyroscope integration with low-frequency accelerometer tilt references:
     $$\theta_{t} = \alpha \cdot (\theta_{t-1} + \omega \cdot \Delta t) + (1 - \alpha) \cdot \theta_{\text{accel}}$$
     Where $\alpha = 0.98$, successfully suppressing fluctuations while preserving fast response times.

### .2 Problem 2: Sim-to-Real Reality Gap & Solution: Exact MuJoCo Digital Twin
* **Identified Symptom:** Pretrained policies struggled when applied to physical hardware because the baseline simulation model assumed idealized geometries and simplified mass distributions, ignoring real physical components like the ESP32 board, battery pack, and MPU-6050 sensor board.
* **Implemented Solution:**
  1. **Component CAD Meshing:** Added exact 3D models for all physical parts into `example_robots/SelfRisingRobot/assets/`:
     - `esp32.STL` (microcontroller body)
     - `MPU_6050.stl` (IMU sensor board)
     - `battery.stl` (power module)
     - `servo1.stl` & `servo2.stl` (actuator bodies)
     - `foot.stl`, `arm1.stl`, `arm2.stl` (structural links)
  2. **Accurate Dynamics & Inertia Definition in `robo1.xml`:**
     - Computed realistic masses and 3D inertia tensors (`fullinertia` matrices) for each link and attached electronic module.
     - Added distinct collision proxy geometries (`geom name="..._collision" type="box"`) to capture ground contact physics accurately without high mesh computation costs.
     - Configured realistic joint hinge limits ($[-\pi/2, \pi/2]$), armature, and damping coefficients ($0.02$).

---

## 2. Technical Comparison Table

| Aspect | Earlier Week Status | This Week Status | Sim-to-Real Impact |
| :--- | :--- | :--- | :--- |
| **Sensor Processing** | Raw register read (`ax, ay, az, gx, gy, gz`) with basic tilt calculations | Static calibration offsets + Complementary filter ($\alpha = 0.98$) | Eliminates sensor jitter and angular drift on real robot |
| **Observation Stability** | Fluctuating readings fed to policy | Smooth, stable, filtered pitch and roll state vectors | Prevents sudden, erratic motor actuations |
| **MuJoCo Simulation Model** | Generic baseline robot model | Exact digital twin replica (`robo1.xml`) with ESP32, MPU6050, battery, servos | Closes dynamics and weight distribution mismatch |
| **Mesh & Assets** | Minimal/placeholder meshes | Comprehensive STL suite for all mechanical and electrical parts | Realistic mass, center of gravity, and contact dynamics |
| **Code Generation Pipeline** | Automated generation of `main.py`, `policy_network.py`, and class drivers | Integrated with calibrated driver classes and custom config support | Seamless transition from trained model to hardware |

---

## 6. Next Steps & Upcoming Milestones

1. **Retrain PPO Policy in MuJoCo:**
   - Train a new self-rising policy using the updated, grounded `robo1.xml` model in Gymnasium.
   - Monitor reward convergence and righting behavior under realistic inertia.
2. **End-to-End Sim-to-Real Deployment:**
   - Run the updated model through the automated `simtoreal` pipeline.
   - Generate the new `policy_network.py` / `.mpy` and deploy along with the calibrated `imu.py` and `servo.py` drivers onto the ESP32.
3. **Physical Hardware Validation & Benchmarking:**
   - Measure real-world time-to-upright and success rate from multiple initial perturbed orientations.
   - Compare performance against simulation rollouts to quantify sim-to-real gap reduction.
