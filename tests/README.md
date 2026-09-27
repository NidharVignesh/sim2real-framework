# Tests for the simtoreal library

These tests check one question: **does what `simtoreal` generates really do the same thing as the trained robot brain (the PyTorch policy), and does it run on the ESP32?**

Every test saves its numbers as a JSON file in `tests/results/`, so the paper uses real, repeatable numbers.

---

## The tests at a glance

| Test | What it checks | Runs on | Needs | Time |
|---|---|---|---|---|
| **Test 1** – export equivalence | Gives the same inputs to PyTorch, the generated MicroPython file and the generated C header, and measures how different the outputs are. | Laptop | nothing | ~1 min |
| **Test 2** – closed loop | Lets the generated C code control the robot in the MuJoCo simulator for 1,000 episodes and checks the results match PyTorch exactly. | Laptop | nothing | ~3 min |
| **Test 3** – generated `main.py` | Generates `main.py` from `config.yaml` and runs it with fake ESP32 hardware. 6 checks: pin types, number of observations, radians, servo commands like in training, 50 Hz loop, and that the generator also works for a different robot. | Laptop | nothing | ~10 s |
| **D1** – policy on the board | Loads the generated `policy_network.mpy` on the ESP32 and checks it gives the same answers as PyTorch. Measures RAM and time per decision. | ESP32 | ESP32 only | ~1 min |
| **D2** – sensor noise | Reads the MPU-6050 500 times while it lies still and measures the noise; compares with the noise the simulator assumed. | ESP32 | ESP32 + MPU-6050 | ~20 s |
| **D3** – hand-written control loop | Runs a hand-written loop for 500 cycles: IMU → policy → servos. Measures the time of each step and missed 20 ms deadlines. **Servos move.** | ESP32 | ESP32 + MPU-6050 + 2 servos | ~1.5 min |
| **D4** – generated `main.py` on the board | Runs the `main.py` the library generated, **unchanged**, for 100 cycles and records what it observes and commands. **Servos move.** | ESP32 | ESP32 + MPU-6050 + 2 servos | ~30 s |
| Optional – C on the board | Same idea as D1 but for the C header, using the Arduino IDE. Only the header can be checked this way (not the full loop). | ESP32 | Arduino IDE | ~10 min |

---

## Folder layout

```
tests/
├── README.md                  ← this file
├── host/                      ← runs on your laptop (normal Python, inside the venv)
│   ├── common.py              ← shared helpers (runs the library, saves JSON, compiles the C header)
│   ├── test_1_export_equivalence.py
│   ├── test_2_closed_loop.py
│   ├── test_3_generated_main.py
│   ├── make_device_files.py   ← builds every file the ESP32 needs
│   └── run_device_tests.py    ← uploads to the ESP32, runs D1–D4, saves the results
├── device/                    ← runs ON the ESP32 (MicroPython)
│   ├── results_util.py        ← saves results as JSON on the board and prints them
│   ├── test_d1_policy.py
│   ├── test_d2_sensor.py
│   ├── test_d3_control_loop.py
│   └── test_d4_generated_main.py
├── optional_arduino_c/policy_bench/policy_bench.ino
├── results/                   ← JSON results land here
└── build/                     ← generated files (made by the scripts, not saved in git)
```

---

## One-time setup

From the repo folder:

```bash
source venv/bin/activate
pip install -e library             # the simtoreal library (also installs mpy-cross)
pip install mpremote pyserial      # only needed for the automatic board runner
```

**What is `mpremote`?** The official MicroPython command-line tool. It does the same job as Thonny (copy files to the board, run a script, show the output), but from the terminal, so a script can do it automatically. You can keep using Thonny; just **close Thonny before using mpremote** (only one program can use the USB port at a time).

---

## Running the laptop tests

```bash
source venv/bin/activate
python tests/host/test_1_export_equivalence.py
python tests/host/test_2_closed_loop.py            # add --episodes 10 for a quick run
python tests/host/test_3_generated_main.py         # --config other.yaml to test another config
```

---

## Running the board tests – Option A: automatic (mpremote)

1. Plug in the ESP32 and close Thonny.
2. Run:
   ```bash
   python tests/host/run_device_tests.py                 # D1, D2, D3 and D4
   python tests/host/run_device_tests.py --tests d1 d4   # only some tests
   python tests/host/run_device_tests.py --port COM5     # on Windows use your COM port
   ```
3. The script builds the files **with the library** (`policy_network.mpy` and `main.py` from `config.yaml`), copies them to a folder `/sim2real_tests` on the board, runs each test, shows the board's output and saves `tests/results/device_d*.json`. Your own files at the root of the board (`main.py`, `imu.py`, …) are **not** touched.

---

## Running the board tests – Option B: by hand with Thonny

1. On the laptop: `python tests/host/make_device_files.py`. Everything the board needs is now in `tests/build/device_upload/`.
2. Open Thonny and connect to the ESP32 (bottom-right corner: *MicroPython (ESP32)*).
3. **View → Files**. Top half = your laptop, bottom half = the board.
4. In the **board** half: right-click → **New directory** → `sim2real_tests`.
5. In the **laptop** half: open `tests/build/device_upload/`, select **all** files, right-click → **Upload to /sim2real_tests**.
6. Open a test file (e.g. `test_d4_generated_main.py`) and press **Run (F5)**. Read the output in the Shell.
7. The result is saved on the board in `/sim2real_results/`. Right-click the `.json` file there → **Download to…** → `tests/results/`.

(The test scripts start with `sys.path.insert(0, "/sim2real_tests")`, so they use the files from that folder, not the older copies at the root of the board. The generated `main.py` is uploaded as `main_generated.py` so it cannot replace yours.)

---

## Deploying your robot files with Thonny (the normal way to use the library)

This is what you do without the tests: generate the 4 robot files and upload them.

1. Generate the policy and the control loop:
   ```bash
   cd library/simtoreal
   simtoreal ../../example_robots/SelfRisingRobot/robo1_getup_ppo.zip -c config.yaml
   ```
   This writes **`policy_network.mpy`** (the default format) and **`main.py`** into the current folder.
   Only if you need another format: `-l python` (→ `policy_network.py`) or `-l c` (→ `policy_network.h`).
   For older MicroPython firmware add e.g. `--micropython-version 1.22`.
2. The driver classes are `library/simtoreal/sensors/imu.py` and `library/simtoreal/actuators/servo.py`.
3. In Thonny (**View → Files**) upload these 4 files to the **root** of the board: `policy_network.mpy`, `main.py`, `imu.py`, `servo.py`.
   ⚠️ This **replaces** the `main.py`, `imu.py`, `servo.py` and `policy_network.mpy` that are on the board now. Download them first (right-click → *Download to…*) if you want to keep your hand-edited versions. Also delete any old `policy_network.py` on the board, otherwise MicroPython may load it instead of the `.mpy`.
4. Press the board's reset button (or Ctrl-D in Thonny). `main.py` starts automatically, moves the servos to their start position and runs the loop. Press Ctrl-C in Thonny to stop it.

What `main.py` does comes entirely from `config.yaml` – see the comments at the top of `library/simtoreal/interface.py` for all keys (`scale`, `offset`, `source: action`, `mode: delta`, `step`, `clip`, `min`, `max`, `initial`, `control: rate_hz`).

---

## Wiring the tests expect (from `library/simtoreal/config.yaml`)

| Part | ESP32 pin |
|---|---|
| MPU-6050 SCL / SDA | GPIO 22 / GPIO 21 |
| MPU-6050 VCC / GND | 3.3 V / GND |
| Servo 1 / Servo 2 signal | GPIO 18 / GPIO 19 |

If your wiring changes, change `config.yaml` (used by D4 and by the generated `main.py`) and the numbers at the top of `device/test_d2_sensor.py` and `device/test_d3_control_loop.py`. Servos can pull a lot of current; powering them from a separate 5 V supply (ground joined to the ESP32 ground) avoids the board resetting.

**Check the IMU is found** (paste into Thonny's Shell) – should print `[104]` (= 0x68):
```python
from machine import Pin, I2C
print(I2C(1, scl=Pin(22), sda=Pin(21)).scan())
```

---

## What the result files contain (main fields)

- **Test 1**: `max_abs_error`, `mean_abs_error` (~1e-6 means "the same up to float rounding"), `clipped_action_agreement_1e-4`, `gcc_warnings`.
- **Test 2**: success rate and time-to-upright for PyTorch vs generated C, per start pose, normal and randomized physics.
- **Test 3**: one entry per check (`A_…` to `F_…`) with `passed` and the values seen; also the generated `main.py` and the config used. `host_test3_generated_main_config_only.json` is the same test with only `az` added to the config and the **old** library (all checks failed).
- **D1**: `policy_ram_bytes`, `load_time_ms`, `correctness_max_abs_error`, `inference_latency_us`, `latency_percent_of_control_period`.
- **D2**: noise (`std`, radians) of roll/pitch vs the simulator's `0.015`; `az_g`; I2C read time.
- **D3**: time of each stage, `achieved_rate_hz`, `deadline_misses`, a short trace.
- **D4**: `n_observations`, `cycles_completed`, `error`, `cycle_time_ms`, `achieved_rate_hz`, a trace of observations and action values.

---

## Results so far (27 Sep 2026)

| Test | Result |
|---|---|
| Test 1 | Generated MicroPython: max error 1.2e-6. Generated C: max error 2.9e-6, compiles with 0 warnings. 100% of actions agree (4,609 inputs). |
| Test 2 | Generated C gives **exactly** the same success rate (97.2%) and time-to-upright as PyTorch, normal and randomized physics (1,000 episodes). Largest action difference 7.1e-7. |
| Test 3 | Old library: **0 of 4** checks passed, even after adding `az` to the config (pins written as text – the ESP32 answers `ValueError: invalid pin`; 3 of 5 observations; degrees; raw action sent to the servo). After the library changes: **6 of 6** pass, including a made-up second robot. |
| D1 | **Correct on the ESP32**: max error 9.5e-7 vs PyTorch. Uses **119.6 KB RAM (72% of the MicroPython heap)** and takes **155 ms per decision – 7.8× the 20 ms period** the policy was trained for. |
| D2 | IMU noise while still: 1.3–1.6 mrad raw, 0.2 mrad filtered – about **10× less** than the 15 mrad used in training (the simulator was cautious). I2C read 1.8 ms. The sensor lay upside down (`az` = −0.96 g) and tilted ~29° during the test; that does not change the noise number. |
| D3 | Hand-written loop, 500 cycles: **5.4 Hz** instead of 50 Hz. IMU 3.2 ms + servos 0.95 ms = **4.2 ms (21% of the 20 ms budget)**; policy **182 ms (98% of the loop)**. Only the network is too slow. |
| D4 | The **library-generated `main.py` runs unchanged** on the ESP32: 5 observations, 100/100 cycles, no errors, 171 ms per cycle (5.8 Hz, limited by the network). |

### Things these tests found

1. **C header did not compile** (fixed): `redefinition of 'tmp'` for any network with more than one layer. One line changed in `exporter.py`.
2. **Generated `main.py` could not run the robot** (fixed, see "Library changes" below). Test 3 and D4 now check it every time.
3. **The MicroPython policy is correct but too slow and too big** for 50 Hz on the ESP32. It is the only slow part of the loop. *Not changed yet* – options: faster MicroPython code (flat `array('f')` + `@micropython.native`) or the C header.
4. **The library's `imu.py` has no gyro calibration and its filter starts at 0°.** The filtered angle sits ~1.3° off (gyro bias), and at the current ~6 Hz loop the filter needs many seconds to reach the real angle – in D4 the pitch reading was still creeping at the end. The hand-edited `imu.py` on your board has a `calibrate()` method. *Not changed yet.*
5. **Servo resolution:** `servo.py` uses the 10-bit PWM duty (26–123 for 0–180°), i.e. **1.86° per step**. The policy moves at most 0.08 rad = 4.6° per step, so small actions are rounded away. `duty_u16()` would give ~0.03° steps. *Not changed yet.*
6. **Assumption:** servo angle 90° = joint angle 0 in the simulator, and a positive joint angle = a larger servo angle (`offset: 90.0` in `config.yaml`). Check when the robot is rebuilt.

### Library changes made (approved)

| File | Change |
|---|---|
| `exporter.py` | C header compile fix (one line); new `generate_mpy_network()` – writes the MicroPython module and precompiles it with mpy-cross. |
| `converter.py`, `cli.py` | Default output is now **`policy_network.mpy`**. `.py` / `.h` only with `-l python` / `-l c` (or an `-o` name ending in `.py` / `.h`). New `--micropython-version`. Checks that the config has as many observations/actions as the network has inputs/outputs. `main.py` is written next to the output file. |
| `interface.py` | Values written as real Python values (pins `22`, not `'22'`); `scale`/`offset` applied; `source: action` observations; `mode: delta` actions with `step`, `clip`, `min`, `max`, `initial`; `init_actuators()`; fixed-rate loop from `control: rate_hz`; custom placeholders stop with a clear message. Old configs still work. Nothing robot-specific is in the code – everything comes from the YAML. |
| `config.yaml` | Robo1 now fully described: roll/pitch (degrees → radians), `az`, both servo targets, servos on 18/19 in `delta` mode (0.08 rad steps, ±1.55 rad), 50 Hz. |
| `setup.py` | Adds `mpy-cross` as a dependency. |

---

## Optional: the C header on the board (Arduino IDE)

> ⚠️ Uploading an Arduino sketch **replaces MicroPython** on the board. Afterwards, re-install MicroPython from Thonny (Tools → Options → Interpreter → *Install or update MicroPython*) and re-upload your files.

1. `python tests/host/make_device_files.py`
2. Install the Arduino IDE; Boards Manager → **esp32** (by Espressif) → Install.
3. Open `tests/build/arduino/policy_bench/policy_bench.ino` (the folder also has `policy_network.h` and `test_vectors.h`).
4. Tools → Board → **ESP32 Dev Module**, choose the port, **Upload**.
5. Write down *"Sketch uses … bytes of program storage"* and *"Global variables use … bytes"* (real Flash and RAM).
6. Serial Monitor at **115200** baud prints one JSON line – save it as `tests/results/device_c_policy.json`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `could not enter raw repl` / port busy / endless `MPY: soft reboot` | Close Thonny. If the board's own `main.py` is printing in a loop, press Ctrl-C in Thonny first (or reset the board) and try again. |
| `OSError: [Errno 19] ENODEV` | The IMU is not answering on I2C. Check wiring and power, then run the scan above. |
| Upload stops with `unexpected read` / strange characters in the output | USB glitch or board reset (often servo current). The runner retries uploads automatically; power the servos separately if it keeps happening. |
| `MemoryError` on the board | Make sure only `policy_network.mpy` (not a `.py`) is on the board, and reset before running a test. |
