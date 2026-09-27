# Results & ERU Submission Guide

This replaces `LITERATURE_SURVEY_AND_RESULTS_GUIDE.md` and `BENCHMARK_AND_EXPERIMENTAL_GUIDE.md`, which contained specific experimental numbers (88.3% physical success rate, 74 µs ESP32 inference, DTW = 0.14 rad, RMSE = 4.82°, an ablation comparing CAD inertia vs. naive models, battery voltage sag figures, etc.) that did not correspond to any measurement actually taken. There is no UART log, timing capture, or physical trial record anywhere in this repo, and `library_plan.md`'s own checklist shows the physical hardware evaluation weeks are unchecked. Those numbers were fabricated by a prior AI session. They have been removed from `paper/extended_abstract.tex` and `paper/results and discussion.tex`.

This guide tells you exactly what is real right now, and exactly what to do to get real physical numbers before you submit.

---

## 1. What is real and already in the paper

**Verified from source code** (`library/simtoreal/`):
- `loaders.py` extracts actor-only weights from an SB3 PPO checkpoint, discarding the critic.
- `exporter.py` generates a C header with `const float` weight arrays and a `policy_forward()` that alternates between two compile-time-sized stack buffers (`buf_a`/`buf_b`) — zero heap allocation, but note: these are **local/stack** arrays inside the function, not global `static` arrays (the earlier guide's "static double-buffered" description was imprecise about this).
- `interface.py` synthesizes `observe() -> policy_forward() -> apply_actions()` from `config.yaml`.

**Verified by running the actual code in this session:**
- Current policy: `robo1_getup_ppo.zip`, actor is a 5→64→64→2 `tanh` MLP, **4,674 parameters** (computed by loading the model with `PPO.load` and counting).
- Generated C header for this policy: 74 lines, weight/bias constants total **≈18.3 KB** as float32 (4,674 × 4 bytes) — this is an arithmetic fact about the exported artifact, **not** a measured on-device Flash number.
- Real simulation results, reproduced live via `example_robots/SelfRisingRobot/eval_policy.py` (50 episodes/case):

  | Case | Nominal Succ. | Nominal t [s] | Domain-Rand Succ. | Domain-Rand t [s] |
  |---|---|---|---|---|
  | roll_pos | 100.0% | 1.36 | 98.0% | 1.66 |
  | roll_neg | 100.0% | 1.00 | 100.0% | 1.19 |
  | pitch_pos | 100.0% | 1.26 | 100.0% | 1.57 |
  | pitch_neg | 100.0% | 0.78 | 100.0% | 0.92 |
  | random | 86.0% | 1.32 | 88.0% | 1.58 |
  | upright | 100.0% | 0.02 | 100.0% | 0.35 |
  | **All falls (agg.)** | **97.2%** | 1.15 | **97.2%** | 1.38 |

  Domain-rand settings: body mass ±15%, joint damping/friction ±20%, actuator gain ±15% (battery-sag proxy), IMU sensor noise σ=0.015 rad. Reproduce with:
  ```bash
  cd example_robots/SelfRisingRobot
  python eval_policy.py --episodes 50 --json nominal_results.json
  python eval_policy.py --episodes 50 --domain-rand --json domain_rand_results.json
  ```
- Training curve (`train.log`): at 2,000,000 PPO steps, held-out eval success rate = 96.67%, mean reward = 359.5 ± 158.8.
- Robot mass 94.8 g: sums exactly from the per-component `mass=` attributes in `robo1.xml` (foot 17.6g + battery×2 14g + MPU-6050 2.1g + ESP32 10g + servo1 9g + arm1 11.4g + servo2 9g + arm2 21.6g).

**Verified citations** (checked against arXiv/publisher records in this session): Hwangbo et al. 2019 (Science Robotics), Rudin et al. 2022 (CoRL), Mittal et al. 2023 (IEEE RA-L, Orbit), Tobin et al. 2017 (IROS), Tan et al. 2018 (RSS), David et al. 2021 (MLSys, TFLM), Lin et al. 2020 (NeurIPS, MCUNet), Eschmann/Albani/Loianno 2024 (JMLR, RLtools — **note:** the old guide attributed RLtools to "Pessia, Neunert, Cazenave," which is wrong; the real authors are Eschmann, Albani, Loianno), and a new find, Zhou et al. 2026 (arXiv:2607.10309) on benchmarking the sim-to-real gap for embedded/AIoT RL, directly relevant to your "how do I benchmark this" question.

## 2. What is NOT real yet — and how to get it for real

You told me: no physical data yet. Here is exactly what to do, in order of effort, all achievable before a symposium deadline.

### Step A — Flash & SRAM (15–30 min, no trials needed)
1. Generate the header from the current model:
   ```bash
   cd library
   python -m simtoreal.cli ../example_robots/SelfRisingRobot/robo1_getup_ppo.zip -c simtoreal/config.yaml -l c -o policy_network.h
   ```
2. Create a minimal Arduino/PlatformIO sketch that `#include "policy_network.h"` and calls `policy_forward()` once in `setup()`.
3. Compile for your ESP32 board and read the real numbers straight from the build output:
   ```
   Sketch uses XXXXX bytes (XX%) of program storage space.
   Global variables use XXXXX bytes (XX%) of dynamic memory.
   ```
4. Compile a second, baseline sketch with *only* I2C + PWM (no policy header), and subtract — that isolates the policy's own Flash/SRAM contribution from the driver overhead.

### Step B — Inference latency & jitter (30–60 min, ESP32 required, no physical trial)
Flash this profiling loop and read the numbers over serial:
```cpp
#include "policy_network.h"
#include <esp_timer.h>
#define N 10000
void profile() {
  float obs[POLICY_INPUT_SIZE] = {0};
  float act[POLICY_OUTPUT_SIZE];
  int64_t total = 0, mx = 0, mn = 1000000;
  for (int i = 0; i < N; i++) {
    int64_t t0 = esp_timer_get_time();
    policy_forward(obs, act);
    int64_t dt = esp_timer_get_time() - t0;
    total += dt; if (dt > mx) mx = dt; if (dt < mn) mn = dt;
  }
  Serial.printf("mean=%.2f us  min=%lld  max=%lld\n", (float)total / N, mn, mx);
}
```
This gives you a genuine mean/min/max inference time and jitter — report these, not invented ones.

### Step C — Physical self-righting trials (the real experiment)
1. Flash the full pipeline (IMU read → `policy_forward` → servo write) at 50 Hz onto the assembled Robo1.
2. For each of the 4 canonical poses (`roll_pos`, `roll_neg`, `pitch_pos`, `pitch_neg`), run **N trials** (15 is a reasonable number for an extended abstract; state your actual N, don't round up). Manually place the robot in the pose each time and note: did it right itself (yes/no), and how long did it take (stopwatch or timestamp from serial log)?
3. Log success/failure and time-to-upright per trial — a simple spreadsheet is enough.
4. Compute success rate = successes/N and mean±SD of time-to-upright per pose, plus an aggregate over all trials. **Whatever numbers come out, that's what goes in the paper** — including if success rate is lower than simulation. A lower real number reported honestly is far stronger, and far safer, than an invented high one.

### Step D — Sim-vs-real trajectory comparison (optional, strengthens the paper)
1. Stream CSV telemetry over UART during trials:
   ```cpp
   Serial.printf("%.3f,%.2f,%.2f,%.3f,%.3f\n", millis()/1000.0f, roll, pitch, act[0], act[1]);
   ```
2. Capture it with `pyserial` into a CSV file.
3. Compare against a matching simulated rollout (see `example_robots/SelfRisingRobot/eval_policy.py` / `robo1_env.py` for how to run one and log `roll`/`pitch` per step).
4. Compute RMSE and, if you want, Dynamic Time Warping distance between the two time series (straightforward with `numpy`/`scipy`; ask if you want this script written once you have real CSVs to test it against).

### Step E — Update the paper
Once you have real numbers from A–D:
- `paper/extended_abstract.tex`, Section IV: replace the "Policy and Generated-Code Footprint" paragraph's analytical estimate with your measured Flash/SRAM/latency, and add a physical-trials paragraph/table analogous to Table I but with your real N and real success rates.
- Section V (Conclusion): change "has been built but not yet benchmarked" to state the real result once you have it.
- Recompile: `cd paper && pdflatex extended_abstract.tex && pdflatex extended_abstract.tex` (run twice for references), and confirm `pdfinfo extended_abstract.pdf` still reports 2 pages.

## 3. ERU Symposium — submission format

- Official template: `ERU Formats/1 ERU Extended-Abstract_Template.docx` — A4, two-column-equivalent IEEE-style layout (verified from the docx's own `pgSz`/`pgMar`/`cols` XML: 595.3×841.9 pt page = A4, Times New Roman), **strictly 2 pages including references**, headings: Introduction, Literature review, Materials and Methods, Results and Discussion, Conclusion, max 5 keywords.
- `extended_abstract.tex` already matches these constraints (IEEEtran conference class, A4, 2 columns) and compiles to exactly 2 pages.
- **Action needed from you:** copy the finished text from the compiled PDF into the official `.docx` template if the submission portal requires their Word file specifically (rather than a PDF in matching format) — check the call for abstracts / ask the ERU organizers which they require.
- If you decide to also do the A1 poster later, its guideline (`ERU Formats/4. ERUS Poster Guidelines.pdf`) wants: Introduction, Materials and Methods, Data and Results, Conclusion and Future Work, References, Acknowledgements, title font ≥36pt, body ≥20pt.

## 4. Bottom line

The paper as it stands is honest and submittable as-is: a real, working library, a real trained policy, and real simulation results including a genuine robustness ablation (nominal vs. domain-randomized physics) — that is a legitimate contribution for a symposium extended abstract, explicitly framed as "physical validation in progress." If you can spare even Step A + B (under an hour, no trial campaign needed) before the deadline, you can upgrade the footprint claim from analytical to measured, which meaningfully strengthens the submission. Step C is the one that would let you additionally claim a real physical success rate.
