# Board test D3 (MicroPython): run the full control loop on the real hardware for 10 seconds.
# Needs: ESP32 + MPU-6050 (SCL=22, SDA=21) + two SG90 servos (pins 18 and 19).
# WARNING: the servos WILL move. Keep fingers and cables clear.
# Each cycle: read IMU -> build observation -> policy_forward -> update servo targets -> write PWM.
# Measures: time of each stage, total cycle time, and how often the 20 ms (50 Hz) deadline is missed.
import sys
sys.path.insert(0, "/sim2real_tests")
for _m in ("policy_network", "imu", "servo", "results_util"):
    sys.modules.pop(_m, None)

import gc
import math
import time
from policy_network import policy_forward
from imu import mpu6050
from servo import servo
from results_util import save, stats

SCL, SDA = 22, 21
SERVO_PINS = (18, 19)
PERIOD_US = 20000     # 50 Hz, same as training
N_CYCLES = 500        # 10 s
TARGET_DELTA = 0.08   # rad per step per unit action (robo1_env.py)
TARGET_LIMIT = 1.55   # rad (robo1_env.py)
SERVO_CENTER_DEG = 90  # ASSUMPTION: 90 deg on the servo == joint angle 0 in the simulator


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


imu = mpu6050(axis="all", scl=SCL, sda=SDA)
servos = [servo(pin) for pin in SERVO_PINS]
for s in servos:
    s.write(SERVO_CENTER_DEG)
print("Servos centred. Control loop starts in 2 s (servos will move)...")
time.sleep(2)

target = [0.0, 0.0]
read_us, infer_us, act_us, busy_us, period_us = [], [], [], [], []
misses = 0
trace = []
gc.collect()
prev_start = None
t_begin = time.ticks_ms()

for cycle in range(N_CYCLES):
    t_start = time.ticks_us()
    if prev_start is not None:
        period_us.append(time.ticks_diff(t_start, prev_start))
    prev_start = t_start

    d = imu.read()
    obs = [math.radians(d["roll"]), math.radians(d["pitch"]), d["accel"]["z"], target[0], target[1]]
    t1 = time.ticks_us()

    action = policy_forward(obs)
    t2 = time.ticks_us()

    for i in range(2):
        target[i] = clip(target[i] + clip(action[i], -1.0, 1.0) * TARGET_DELTA, -TARGET_LIMIT, TARGET_LIMIT)
        servos[i].write(SERVO_CENTER_DEG + math.degrees(target[i]))
    t3 = time.ticks_us()

    read_us.append(time.ticks_diff(t1, t_start))
    infer_us.append(time.ticks_diff(t2, t1))
    act_us.append(time.ticks_diff(t3, t2))
    busy = time.ticks_diff(t3, t_start)
    busy_us.append(busy)
    if busy > PERIOD_US:
        misses += 1
    if cycle % 10 == 0:
        trace.append([cycle, obs[0], obs[1], obs[2], action[0], action[1], target[0], target[1]])

    spare = PERIOD_US - time.ticks_diff(time.ticks_us(), t_start)
    if spare > 0:
        time.sleep_us(spare)

wall_s = time.ticks_diff(time.ticks_ms(), t_begin) / 1000
for s in servos:
    s.write(SERVO_CENTER_DEG)

# The policy takes ~120 KB of RAM; free it and the raw timing lists before building the JSON,
# otherwise saving fails with MemoryError.
stage_stats = {
    "stage_imu_read_us": stats(read_us),
    "stage_policy_inference_us": stats(infer_us),
    "stage_servo_write_us": stats(act_us),
    "total_busy_time_us": stats(busy_us),
    "loop_period_us": stats(period_us),
}
del read_us, infer_us, act_us, busy_us, period_us, policy_forward
sys.modules.pop("policy_network", None)
gc.collect()

result = {
    "backend": "simtoreal generated MicroPython module",
    "n_cycles": N_CYCLES,
    "target_period_us": PERIOD_US,
    "achieved_rate_hz": N_CYCLES / wall_s,
    "deadline_misses": misses,
    "deadline_miss_percent": 100 * misses / N_CYCLES,
    "servo_pins": list(SERVO_PINS),
    "servo_center_assumption_deg": SERVO_CENTER_DEG,
    "trace_columns": ["cycle", "roll_rad", "pitch_rad", "az_g", "action0", "action1", "target0_rad", "target1_rad"],
    "trace_every_10th_cycle": trace,
}
result.update(stage_stats)
print("Achieved %.1f Hz (target 50 Hz), deadline misses %d/%d"
      % (result["achieved_rate_hz"], misses, N_CYCLES))
print("Mean stage times: IMU %.0f us | policy %.0f us | servos %.0f us | total %.0f us"
      % (result["stage_imu_read_us"]["mean"], result["stage_policy_inference_us"]["mean"],
         result["stage_servo_write_us"]["mean"], result["total_busy_time_us"]["mean"]))
save("device_d3_control_loop", result)
