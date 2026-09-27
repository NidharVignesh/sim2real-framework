# Board test D2 (MicroPython): is the real IMU as noisy as the simulator assumed?
# Needs: ESP32 + MPU-6050 (SCL=22, SDA=21). Keep the sensor FLAT and STILL during the test.
# Measures: noise of roll/pitch/az and I2C read time, compared with the simulator's noise setting.
import sys
sys.path.insert(0, "/sim2real_tests")
for _m in ("imu", "results_util"):
    sys.modules.pop(_m, None)

import math
import time
from imu import mpu6050
from results_util import save, stats

SCL, SDA = 22, 21
N_SAMPLES = 500
PERIOD_US = 20000
SIM_NOISE_STD_RAD = 0.015  # robo1_env.py: sensor_noise_std used during domain randomization

imu = mpu6050(axis="all", scl=SCL, sda=SDA)
print("Keep the MPU-6050 flat and still. Starting in 3 s...")
time.sleep(3)
# The library's complementary filter starts at 0 deg and (alpha=0.98, 50 Hz) takes a few seconds
# to converge to the real tilt, so wait 6 s before measuring or the ramp is counted as noise.
for _ in range(300):
    imu.read()
    time.sleep_ms(20)

raw_roll, raw_pitch, filt_roll, filt_pitch, az, read_us = [], [], [], [], [], []
for _ in range(N_SAMPLES):
    t = time.ticks_us()
    d = imu.read()
    read_us.append(time.ticks_diff(time.ticks_us(), t))
    raw_roll.append(math.radians(d["tilt_x"]))
    raw_pitch.append(math.radians(d["tilt_y"]))
    filt_roll.append(math.radians(d["roll"]))
    filt_pitch.append(math.radians(d["pitch"]))
    az.append(d["accel"]["z"])
    spare = PERIOD_US - time.ticks_diff(time.ticks_us(), t)
    if spare > 0:
        time.sleep_us(spare)

result = {
    "sensor": "MPU-6050 via simtoreal imu.py (I2C %d kHz)" % 400,
    "n_samples": N_SAMPLES,
    "sample_rate_hz": 1e6 / PERIOD_US,
    "raw_accel_roll_rad": stats(raw_roll),
    "raw_accel_pitch_rad": stats(raw_pitch),
    "filtered_roll_rad": stats(filt_roll),
    "filtered_pitch_rad": stats(filt_pitch),
    "az_g": stats(az),
    "i2c_read_time_us": stats(read_us),
    "sim_assumed_noise_std_rad": SIM_NOISE_STD_RAD,
}
for key in ("raw_accel_roll_rad", "raw_accel_pitch_rad", "filtered_roll_rad", "filtered_pitch_rad"):
    result[key]["std_as_fraction_of_sim_noise"] = result[key]["std"] / SIM_NOISE_STD_RAD

print("Noise (std, rad):  raw roll %.4f  raw pitch %.4f  filtered roll %.4f  filtered pitch %.4f"
      % (result["raw_accel_roll_rad"]["std"], result["raw_accel_pitch_rad"]["std"],
         result["filtered_roll_rad"]["std"], result["filtered_pitch_rad"]["std"]))
print("Simulator assumed noise std: %.4f rad" % SIM_NOISE_STD_RAD)
print("az mean %.3f g (should be about +1.0 when flat, -1.0 upside down)" % result["az_g"]["mean"])
print("I2C read time: mean %.0f us, max %d us" % (result["i2c_read_time_us"]["mean"], result["i2c_read_time_us"]["max"]))
save("device_d2_sensor", result)
