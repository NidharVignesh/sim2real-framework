# simtoreal

`simtoreal` version `0.1.0` converts trained reinforcement learning MLP policies into MicroPython bytecode (`.mpy`, default), MicroPython source (`.py`), or a C header (`.h`). An optional YAML hardware description generates a MicroPython `main.py` control loop. Conversion runs on the host; the board runs the generated files without PyTorch or Stable-Baselines3.

## Current structure

```text
library/
├── README.md
├── setup.py                 # Dependencies and CLI registration
├── library_plan.md          # Earlier architecture plan
├── main.py                  # Older generated loop with unfinished observations
├── policy_network.py        # Generated example: 4 inputs, 2 outputs
├── policy_network.mpy       # Compiled policy artifact
└── simtoreal/
    ├── __init__.py          # Public convert() API and version
    ├── cli.py               # Command-line entry point
    ├── converter.py         # Format selection and dimension checks
    ├── loaders.py           # SB3 PPO and PyTorch loaders
    ├── exporter.py          # C, Python, and .mpy generation
    ├── interface.py         # YAML-to-main.py generator
    ├── config.yaml          # Current example: 5 observations, 2 actions
    ├── config1.yaml         # Older, incompatible configuration schema
    ├── sensors/
    │   └── imu.py           # MicroPython MPU6050 driver
    └── actuators/
        └── servo.py         # MicroPython PWM servo driver
```

The checked-in `main.py` predates the current generator and contains TODOs. The checked-in Python policy has four inputs, while `config.yaml` requires five. Regenerate the policy and loop with a matching trained model before deployment. Check the compiled artifact's firmware compatibility before use.

## Installation

From the **repository root**:

```bash
python -m pip install -e ./library
```

From `library/`, use `python -m pip install -e .` instead. Use `python3` if that is your environment's Python command.

`setup.py` declares `numpy`, `torch`, `stable-baselines3`, `pyyaml`, and `mpy-cross` and registers the `simtoreal` command.

## Command-line usage

Run from the repository root, replacing the model paths:

```bash
# Default: policy_network.mpy in the current directory
simtoreal path/to/model.zip

# Generate main.py too; this configuration needs 5 inputs and 2 outputs
simtoreal path/to/model.zip -c library/simtoreal/config.yaml

# MicroPython source or C header
simtoreal path/to/model.zip -l python
simtoreal path/to/model.zip -l c

# Infer format from the extension
simtoreal path/to/actor.pt -o robot_policy.h

# Select a compiler version for older MicroPython firmware
simtoreal path/to/model.zip --micropython-version 1.22

simtoreal --help
```

| Option | Behavior |
| --- | --- |
| `model` | Required `.zip` (SB3 PPO), `.pt`, or `.pth` (PyTorch) |
| `-o`, `--output` | Output path; defaults to `policy_network` with the selected suffix |
| `-l`, `--lang` | `mpy`, `python`, or `c`; overrides extension inference and rewrites the suffix |
| `-c`, `--config` | Validates observation/action counts and generates `main.py` beside the policy |
| `--micropython-version` | Compiler version selection for `.mpy` only |

Without an explicit language or recognized `.mpy`, `.py`, or `.h` extension, output defaults to `.mpy`. Create destination directories beforehand. Existing output files at the selected paths are overwritten.

## Python API

```python
from simtoreal import convert

output = convert("ppo_model.zip")  # Returns "policy_network.mpy"
convert("ppo_model.zip", lang="python")
convert("actor.pt", output_path="robot_policy.h")
convert(
    "ppo_model.zip",
    config_path="library/simtoreal/config.yaml",
    lang="mpy",
    micropython_version="1.22",
)
```

## Hardware configuration

Use [simtoreal/config.yaml](simtoreal/config.yaml) as the current example. Observations and actions must match the model dimensions and training order. The converter checks counts, but does not verify training semantics or hardware settings.

| Entry | Fields and behavior |
| --- | --- |
| `control` | Optional `rate_hz`; schedules in milliseconds. Omit to loop as fast as possible. |
| Sensor observation | `name`, `type` (module), `class` (driver); calls `read()` and applies `raw * scale + offset`. |
| Action-state observation | `source: action` with `action: <name>` or `index: <zero-based index>`; reads the stored value in policy units, optionally scaled/offset. |
| Actuator action | `name`, `type`, `class`; sends `value * scale + offset` to `write()`. |
| Action mode | `absolute` (default) assigns output; `delta` adds `output * step` to the previous value and requires `step`. |
| Action bounds | `clip: [lo, hi]` clips raw output first; `min`/`max` bound the resulting stored value. |
| Initial state | `initial` defaults to `0.0`; sent to implemented actuators before the loop starts. |

Other keys are passed to driver constructors, such as `axis`, `scl`, `sda`, and `pin`. Sensor observations need scalar `read()` results; actuators need `write(command)`.

Entries marked `custom`, names containing `custom`, or entries missing `type`/`class` become TODOs. Unfinished custom observations raise `NotImplementedError`; custom actuator writes remain commented out. Action-state observations do not need a driver.

The current example runs at 50 Hz and observes roll/pitch in radians, vertical acceleration in g, and two servo targets. Actions are clipped to ±1, use a `0.08` radian delta step and target limits of ±1.55 radians, then convert to degrees with a 90-degree offset. I2C uses SCL 22/SDA 21 and servos GPIO 18/19. Adapt pins and calibration to your robot.

`config1.yaml` uses an older nested schema that the generator does not implement; it is not a working alternative to `config.yaml`.

## Deploying to MicroPython

1. Convert a matching model with configuration and `.mpy` or `.py` output.
2. Review generated `main.py`, complete custom entries, and check units, action mapping, pins, and calibration.
3. Copy these files into the board's filesystem:

   ```text
   main.py
   policy_network.mpy  # or policy_network.py
   imu.py             # from library/simtoreal/sensors/imu.py
   servo.py           # from library/simtoreal/actuators/servo.py
   ```

4. Check `.mpy` compatibility with the board's firmware and run `main.py`.

Generated loops always import `from policy_network import policy_forward`. If you select another output basename, update that import or rename the module. Driver imports use top-level names (`imu`, `servo`), so copy drivers as shown.

Precompiling avoids compiling large policy source on the board; weights and inference still consume runtime memory. The drivers require MicroPython's `machine` and `utime` modules and cannot run directly in host CPython.

## C output

The header provides weight/bias arrays, `POLICY_INPUT_SIZE`, `POLICY_OUTPUT_SIZE`, and `policy_forward()`. Two temporary buffers are sized to the widest layer.

```c
#include "policy_network.h"

void run_policy(const float *observations, float *actions) {
    policy_forward(observations, actions);
}
```

Provide correctly sized arrays and implement your firmware's sensor/actuator loop. The header uses standard `math.h` functions; some toolchains require `-lm`. With C output, `--config` still generates a **MicroPython** `main.py`, not a C hardware interface.

## Supported models and current limitations

- **SB3:** `.zip` files use `PPO.load()`. Only the actor MLP and final action layer are extracted; critic, feature extraction, and distribution/sampling logic are omitted. The loader assumes `tanh` hidden activations.
- **PyTorch:** `.pt`/`.pth` must contain a serialized model object supporting `eval()` and `named_modules()`, such as a simple `nn.Sequential` MLP. Standalone `state_dict` files are unsupported. Use trusted models because loading can fall back to full-object unpickling.
- **Architecture:** Linear layers are exported in traversal order as one feed-forward chain. CNNs, recurrent networks, branches, and arbitrary preprocessing are not reproduced.
- **Activations:** The PyTorch loader detects the first recognized activation, but exporters implement only `tanh` and ReLU. Other detected activations fall back to ReLU. All hidden layers use the same activation; the output layer is always linear.
- **Biases:** Python/`.mpy` export substitutes zeros for bias-free layers. C export references bias arrays even when absent, so use biased layers for C output.
- **Training parity:** Normalization, feature preprocessing, and action postprocessing are not automatically exported. Reproduce required transformations on the target; YAML supports affine scaling and action clipping/limits.
- **Scope:** ONNX, raw-weight loading, quantization, automatic numerical parity checks, and hardware validation are not implemented. `library_plan.md` describes earlier plans beyond the current implementation.
