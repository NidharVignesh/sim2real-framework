# Board test D1 (MicroPython): does the generated policy run correctly on the ESP32, and how fast?
# Needs: only the ESP32 (no sensors or servos).
# Measures: RAM used by the policy, load time, correctness vs PyTorch, inference latency.
import sys
sys.path.insert(0, "/sim2real_tests")
for _m in ("policy_network", "test_vectors", "results_util"):
    sys.modules.pop(_m, None)

import gc
import os
import time

CONTROL_PERIOD_US = 20000  # 50 Hz, the rate the policy was trained at
N_TIMING = 200

gc.collect()
heap_total = gc.mem_free() + gc.mem_alloc()
free_before = gc.mem_free()
t0 = time.ticks_ms()
from policy_network import policy_forward, POLICY_INPUT_SIZE, POLICY_OUTPUT_SIZE
load_ms = time.ticks_diff(time.ticks_ms(), t0)
gc.collect()
free_after = gc.mem_free()

from test_vectors import OBS, EXPECTED
from results_util import save, stats

print("Policy loaded: %d inputs -> %d outputs, uses %d bytes of RAM (%d free of %d)"
      % (POLICY_INPUT_SIZE, POLICY_OUTPUT_SIZE, free_before - free_after, free_after, heap_total))

# 1) Correctness: compare the board's output with PyTorch's output for the same inputs
errors = []
for obs, expected in zip(OBS, EXPECTED):
    out = policy_forward(obs)
    errors.append(max(abs(out[i] - expected[i]) for i in range(POLICY_OUTPUT_SIZE)))
max_err = max(errors)
passed = max_err < 1e-4
print("Correctness vs PyTorch: max abs error = %.3e over %d inputs -> %s"
      % (max_err, len(errors), "PASS" if passed else "FAIL"))

# 2) Latency: time many forward passes
gc.collect()
times = []
for i in range(N_TIMING):
    obs = OBS[i % len(OBS)]
    t = time.ticks_us()
    policy_forward(obs)
    times.append(time.ticks_diff(time.ticks_us(), t))
lat = stats(times)
print("Inference latency: mean %.0f us, min %d us, max %d us (n=%d)"
      % (lat["mean"], lat["min"], lat["max"], lat["n"]))
print("That is %.1f%% of the 20 ms (50 Hz) control period" % (100 * lat["mean"] / CONTROL_PERIOD_US))

try:
    mpy_bytes = os.stat("/sim2real_tests/policy_network.mpy")[6]
except OSError:
    mpy_bytes = None

save("device_d1_policy", {
    "backend": "simtoreal generated MicroPython module (policy_network.mpy, precompiled with mpy-cross)",
    "policy_file_bytes": mpy_bytes,
    "heap_total_bytes": heap_total,
    "heap_free_before_load_bytes": free_before,
    "heap_free_after_load_bytes": free_after,
    "policy_ram_bytes": free_before - free_after,
    "load_time_ms": load_ms,
    "correctness_n_inputs": len(errors),
    "correctness_max_abs_error": max_err,
    "correctness_mean_abs_error": sum(errors) / len(errors),
    "correctness_pass_threshold": 1e-4,
    "correctness_passed": passed,
    "inference_latency_us": lat,
    "control_period_us": CONTROL_PERIOD_US,
    "latency_percent_of_control_period": 100 * lat["mean"] / CONTROL_PERIOD_US,
})
