# simtoreal
## Library Specification & Architecture

**Library name:** `simtoreal`
**Target platform:** ESP32
**Purpose:** Convert a trained PPO policy network (Stable-Baselines3) into a C header file (`.h`) that runs inference directly on an ESP32, using an IMU-based observation input.

---

## 1. Scope

`simtoreal` sits between the training side and the hardware side of the pipeline. The library's only job is converting a trained model into a working `.h` file. It does not train models, does not run on the ESP32 itself, and does not (yet) verify correctness — that is a later phase.

## Where the library sits

```mermaid
flowchart TD
    A["MuJoCo + Gym Env<br/>(robo1.xml + assets + env.py)"] --> B["PPO Training<br/>(train_robo1.py / pretrain...)"]
    B --> C["Your Python Library<br/>(Network → .h)"]
    C --> D["policy_network.h<br/>(+ optional .c)"]
    D --> E["Embedded Target<br/>(ESP32)"]
```

## Structure

```mermaid
flowchart TB
    subgraph Library[simtoreal]
        direction TB

        Input["Input Layer<br/>• SB3 .zip<br/>• PyTorch model<br/>• ONNX<br/>• Raw weights"]
        Core["Core Converter<br/>• Load model<br/>• Extract MLP architecture<br/>• Flatten layers<br/>• Optional Quantize"]
        Output["Output Generator<br/>• Generate .h header<br/>• Generate weights as C arrays<br/>• Add inference function<br/>• Add usage comments"]
        Config["Config / CLI<br/>• layer names<br/>• data type float / int8<br/>• output path"]

        Input --> Core
        Core --> Output
        Config --> Core
        Config --> Output
    end
```

## WorkFlow

```mermaid
flowchart TD
    A["User / Training Script"] -->|1. Provide trained model<br/>robo1_getup_ppo.zip or .pt / .onnx| B["Library Entry Point<br/>convert model, out<br/>CLI or Python API"]
    B --> C["Model Loader<br/>SB3 / Torch / ONNX<br/>Extract policy network<br/>actor MLP only"]
    C --> D["Weight Exporter<br/>Convert tensors →<br/>C float / int8 arrays"]
    D --> F["Code Generator<br/>Write:<br/>• #ifndef guards<br/>• weight arrays<br/>• forward function<br/>• input/output sizes"]
    F --> G["policy_network.h"]
    G --> H["ESP32 Firmware<br/>includes the .h and calls forward"]
```