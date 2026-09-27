# Board test D4 (MicroPython): run the main.py that simtoreal GENERATED, unchanged, on the real hardware.
# Needs: ESP32 + MPU-6050 + two servos, wired as in library/simtoreal/config.yaml.
# WARNING: the servos WILL move.
# The generated file is uploaded as main_generated.py so it does not replace the board's own main.py.
# This script imports it and calls its own init_actuators() and policy() functions, timing each cycle
# with the generated PERIOD_MS, exactly like its `while True` loop does.
import sys
sys.path.insert(0, "/sim2real_tests")
for _m in ("main_generated", "policy_network", "imu", "servo", "results_util"):
    sys.modules.pop(_m, None)

import gc
import time
import main_generated as gen
from results_util import save, stats

N_CYCLES = 100
period_ms = getattr(gen, "PERIOD_MS", None)

first_obs = gen.observe()
n_obs = len(first_obs)
print("Generated main.py loaded: %d observations, %d actions, PERIOD_MS=%s"
      % (n_obs, len(gen.action_values), period_ms))
print("First observation:", [round(v, 4) for v in first_obs])

gen.init_actuators()
print("Actuators at their initial values. Running %d cycles (servos will move)..." % N_CYCLES)
time.sleep(1)

cycle_ms, trace, error = [], [], None
gc.collect()
t_begin = time.ticks_ms()
try:
    for cycle in range(N_CYCLES):
        start = time.ticks_ms()
        gen.policy()
        busy = time.ticks_diff(time.ticks_ms(), start)
        cycle_ms.append(busy)
        if cycle % 10 == 0:
            trace.append([cycle] + [round(v, 4) for v in gen.observe()] + [round(v, 4) for v in gen.action_values])
        if period_ms is not None and busy < period_ms:
            time.sleep_ms(period_ms - busy)
except Exception as e:  # report whatever the generated code raises
    error = "%s: %s" % (type(e).__name__, e)
wall_s = time.ticks_diff(time.ticks_ms(), t_begin) / 1000
gen.action_values[:] = [0.0] * len(gen.action_values)
gen.init_actuators()

done = len(cycle_ms)
timing = stats(cycle_ms) if cycle_ms else None
misses = sum(1 for c in cycle_ms if period_ms is not None and c > period_ms)
del gen
sys.modules.pop("main_generated", None)
sys.modules.pop("policy_network", None)
gc.collect()

print("Completed %d/%d cycles, error: %s" % (done, N_CYCLES, error))
if timing:
    print("Cycle time: mean %.0f ms (target %s ms) -> %.1f Hz, deadline misses %d/%d"
          % (timing["mean"], period_ms, done / wall_s, misses, done))
save("device_d4_generated_main", {
    "what": "library-generated main.py run unchanged on the ESP32",
    "n_observations": n_obs,
    "first_observation": first_obs,
    "cycles_requested": N_CYCLES,
    "cycles_completed": done,
    "error": error,
    "period_ms": period_ms,
    "cycle_time_ms": timing,
    "achieved_rate_hz": done / wall_s if wall_s else None,
    "deadline_misses": misses,
    "trace_columns": ["cycle", "roll_rad", "pitch_rad", "az_g", "target0_rad", "target1_rad",
                      "action_value0_rad", "action_value1_rad"],
    "trace_every_10th_cycle": trace,
})
