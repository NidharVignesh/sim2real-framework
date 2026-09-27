// OPTIONAL board test (C / Arduino): correctness + latency of the generated C header on the ESP32.
// Flashing an Arduino sketch REPLACES MicroPython on the board. To go back to Thonny afterwards,
// re-flash MicroPython (Thonny: Tools > Options > Interpreter > "Install or update MicroPython").
//
// Do not open this copy directly: run `python tests/host/make_device_files.py` first, then open
// tests/build/arduino/policy_bench/policy_bench.ino (that folder also has policy_network.h and
// test_vectors.h next to it, which Arduino needs).
#include "policy_network.h"
#include "test_vectors.h"
#include <esp_timer.h>

void setup() {
  Serial.begin(115200);
  delay(1500);

  float max_err = 0;
  for (int i = 0; i < NUM_TESTS; i++) {
    float act[POLICY_OUTPUT_SIZE];
    policy_forward(test_obs[i], act);
    for (int j = 0; j < POLICY_OUTPUT_SIZE; j++) {
      float d = fabsf(act[j] - test_expected[i][j]);
      if (d > max_err) max_err = d;
    }
  }

  const int N = 10000;
  int64_t total = 0, mx = 0, mn = 1000000;
  float act[POLICY_OUTPUT_SIZE];
  for (int i = 0; i < N; i++) {
    int64_t t0 = esp_timer_get_time();
    policy_forward(test_obs[i % NUM_TESTS], act);
    int64_t dt = esp_timer_get_time() - t0;
    total += dt;
    if (dt > mx) mx = dt;
    if (dt < mn) mn = dt;
  }

  // One JSON line: copy it from the Serial Monitor into tests/results/device_c_policy.json
  Serial.printf("{\"test\":\"device_c_policy\",\"correctness_max_abs_error\":%.3e,"
                "\"correctness_passed\":%s,\"latency_mean_us\":%.2f,"
                "\"latency_min_us\":%lld,\"latency_max_us\":%lld,\"n\":%d}\n",
                max_err, max_err < 1e-4 ? "true" : "false",
                (float)total / N, mn, mx, N);
}

void loop() {}
