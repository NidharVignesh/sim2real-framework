# simtoreal

Convert trained Reinforcement Learning policies (Stable-Baselines3 or pure PyTorch) into files a microcontroller such as the ESP32 can run: precompiled MicroPython (`.mpy`, the default), MicroPython source (`.py`) or a C header (`.h`). From a YAML hardware description it also generates the on-board control loop (`main.py`).

## Features

- Supports **Stable-Baselines3** PPO models (`.zip`)
- Supports **pure PyTorch** models (`.pt` / `.pth`)
- Automatically detects activation function (`tanh`, `relu`, ...)
- Generates pure C code with no external dependencies
- Safe double-buffering for any layer size
- Easy command-line interface

## Project Structure

```
simtoreal/
├── setup.py
├── README.md
└── simtoreal/
    ├── __init__.py
    ├── cli.py              # Command-line entry point
    ├── converter.py        # Main convert() function
    ├── loaders.py          # load_sb3_policy() & load_torch_policy()
    └── exporter.py         # generate_header()
```

## Installation

```bash
# From the project root
pip install -e .
```

## Usage

### 1. Command Line

```bash
# Default: precompiled MicroPython -> policy_network.mpy
simtoreal path/to/model.zip

# Also generate main.py (control loop) from a hardware description
simtoreal path/to/model.zip -c config.yaml

# Other formats, only when you need them
simtoreal path/to/model.zip -l python        # -> policy_network.py
simtoreal path/to/model.zip -l c             # -> policy_network.h
simtoreal path/to/model.pt -o robot_policy.h # format taken from the extension

# .mpy for an older MicroPython firmware
simtoreal path/to/model.zip --micropython-version 1.22
```

The YAML keys (`scale`, `offset`, `source: action`, `mode: delta`, `step`, `clip`, `min`, `max`, `initial`, `control: rate_hz`) are documented at the top of `simtoreal/interface.py`; `simtoreal/config.yaml` is a complete example.

### 2. From Python

```python
from simtoreal.converter import convert

convert("ppo_model.zip")                                   # policy_network.mpy
convert("ppo_model.zip", config_path="config.yaml")        # + main.py
convert("actor.pt", "policy_network.h")                    # C header
```

## Generated Header Example

After conversion you get a file like `policy_network.h` that contains:

```c
#define POLICY_INPUT_SIZE  24
#define POLICY_OUTPUT_SIZE 4

// Weight & bias arrays
const float policy_w0[...];
const float policy_b0[...];
...

// Activation function
static inline float act(float x) { return tanhf(x); }

// Forward function
static inline void policy_forward(const float* input, float* output);
```

<!-- ## How to use the generated header on ESP32 / MCU

```c
#include "policy_network.h"

void control_loop() {
    float obs[POLICY_INPUT_SIZE];
    float action[POLICY_OUTPUT_SIZE];

    // 1. Fill observation from sensors
    // obs[0] = ...
    // obs[1] = ...

    // 2. Run the policy
    policy_forward(obs, action);

    // 3. Send action to motors
    // set_motor(action[0], action[1], ...);
}
``` -->

## Supported Models

| Framework            | File type     | Notes                              |
|-----------------------|---------------|-------------------------------------|
| Stable-Baselines3    | `.zip`        | PPO (MlpPolicy) recommended        |
| Pure PyTorch         | `.pt` / `.pth`| `nn.Sequential` or simple MLP      |

## Notes

- Only the **actor / policy** network is exported (critic is ignored).
- The last layer is always treated as **linear** (no activation).
- Temporary buffers are automatically sized to the widest layer.
- Make sure observation normalization used during training is also applied on the microcontroller.

## Example Workflow

```bash
# 1. Train your agent (example with SB3)
python train.py   # saves model.zip

# 2. Convert to C header
simtoreal model.zip -o policy_network.h

# 3. Copy the header into your ESP-IDF / Arduino project
# 4. Include it and call policy_forward()
```