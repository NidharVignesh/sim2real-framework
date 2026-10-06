# Three-minute video: The simtoreal library

**Duration:** 3:00. **Narration:** 436 words, delivered at about 150 words per minute with brief visual pauses.
**Story:** deployment problem → our library → how it works → current progress → robot example → reusable workflow.

The library is the main subject. Introduce the robot only in the example segment at 2:08. Use actual code, simple workflow graphics, and the supplied footage; generated imagery is unnecessary for this technical explanation.

## 0:00–0:25 — Problem statement

**Voice-over:**
> How do we take a robot policy trained on a computer and run it on a small microcontroller? Training tools such as PyTorch and Stable-Baselines3 operate on the computer, while an ESP32 has limited memory and processing capacity. Deployment requires translating the policy, connecting sensors and actuators, and preserving the same input and output meaning used during training.

**Visual:** Title: **“simtoreal: From trained policies to microcontrollers”**. Show a training computer, a saved model, and an ESP32 block separated by a gap labeled “Deployment work”. Reveal three tasks: “Policy conversion”, “Hardware mapping”, “Control loop”. Use a clean slide or animated diagram.

## 0:25–0:45 — Our solution

**Voice-over:**
> Our project addresses this deployment work through the simtoreal library. It takes a compatible trained model and a YAML hardware configuration, exports the policy for embedded inference, and generates a MicroPython control loop. The goal is to make deployment repeatable, so developers can spend more time improving robot behavior and less time rewriting integration code.

**Visual:** Replace the gap with the library. Reveal this diagram one element at a time:

```text
Trained model ────────┐
                     ├──► simtoreal library ──► Policy + main.py
Hardware YAML ───────┘                              │
                                                   ▼
                                           Microcontroller
```

**Caption:** “Model + hardware configuration → deployment files”

## 0:45–1:10 — Step 1: Export the policy

**Voice-over:**
> First, the library loads a supported policy: an MLP actor from a Stable-Baselines3 PPO ZIP file, or a compatible PyTorch model. It extracts the network weights and produces Python source, compiled MicroPython bytecode, or a C header. The microcontroller runs the exported inference code without installing the desktop training framework. Training remains on the computer.

**Visual:** Show `library/simtoreal/cli.py`, then `converter.py` and its format selection. Keep code readable and highlight only relevant lines. Show three output cards:

| Output | Intended use |
| --- | --- |
| `policy_network.py` | MicroPython source |
| `policy_network.mpy` | Precompiled MicroPython module |
| `policy_network.h` | Integration into C firmware |

**Caption:** “Export inference for the target”

## 1:10–1:35 — Step 2: Describe the hardware

**Voice-over:**
> Next, the YAML configuration describes the sensors, actuators, connected pins, and control rate. It defines observation order, unit conversions, and how policy outputs become actuator commands. The converter checks that the configured input and output counts match the model. Developers still need to ensure those values have the same meaning as in training.

**Visual:** Open `library/simtoreal/config.yaml`. Highlight in order: `observations`, `scl`/`sda`, `actions`/`pin`, `scale`/`offset`, `mode`/`step`, and `rate_hz`. Label this as an **example configuration**, leaving detailed robot explanation until later.

**Caption:** “Pins • input order • units • action mapping”

## 1:35–1:55 — Step 3: Generate the control loop

**Voice-over:**
> From that configuration, the library generates main.py using existing sensor and actuator drivers. Its loop reads observations, evaluates the policy, and applies the actions at the configured interval. For MicroPython deployment, we copy the policy, main.py, and required drivers to the board. The C header is available for separate integration into C firmware.

**Visual:** Show `interface.py` generating `observe()`, `policy()`, and `apply_actions()`, or show a freshly generated `main.py`. Follow with a deployment file card:

```text
main.py
policy_network.py OR policy_network.mpy
required sensor and actuator drivers
```

**Caption:** “Read → infer → actuate → repeat”

## 1:55–2:08 — Where we are now

**Voice-over:**
> We currently have policy loaders, three export formats, YAML-based loop generation, and MPU-6050 and servo drivers. This establishes the deployment pipeline. Broader model support and measured hardware performance remain areas for further development.

**Visual:** Show four implemented capability cards: “Policy loading”, “Policy export”, “YAML integration”, “Hardware drivers”. Briefly show a smaller “Next: broader support and performance evaluation” caption.

## 2:08–2:40 — Example: the self-rising robot

**Voice-over:**
> To demonstrate the workflow, we use this ESP32 self-rising robot. Its policy is trained in MuJoCo using PPO. Five inputs describe its orientation, vertical acceleration, and current servo targets; two outputs adjust the servo joints. We export the trained policy and map the hardware through YAML. Here, the simulation shows recovery behavior, followed by a physical trial showing the robot moving toward upright after release. These clips illustrate the application; repeated tests are needed to quantify reliability.

**Visual sequence:**

| Video timeline | Footage |
| --- | --- |
| 2:08–2:13 | `../images/real_robo.png`; label “Example application: self-rising robot”. |
| 2:13–2:21 | `Screencast from 2026-10-06 22-36-02.mp4`, approximately source 8–16 s. Label “MuJoCo simulation”. Crop toward the robot and retain ground contact. |
| 2:21–2:25 | Brief workflow overlay: “PPO model → library + YAML → ESP32”. |
| 2:25–2:37 | `WhatsApp Video 2026-10-06 at 22.20.43.mp4`, approximately source 2.4–14.38 s, at normal speed. Keep placement, hand release, and subsequent motion visible. Label “Physical trial”. |
| 2:37–2:40 | Hold the final frame and return focus to “Powered by the simtoreal deployment workflow”. |

Source cuts are approximate guides from inspected frames; adjust to the exact release and recovery moments in the editor. The longer `WhatsApp Video 2026-10-06 at 22.20.42.mp4` is optional and is not needed for this three-minute edit.

## 2:40–3:00 — Reuse and closing

**Voice-over:**
> When the wiring and policy interface stay the same, we can reuse the configuration, retrain the policy, and export again. Hardware tests then guide further simulation and training changes. Our contribution is a reusable library that connects trained policies to resource-constrained microcontrollers, with the self-rising robot as one example of that workflow.

**Visual:** Return to the library diagram. Animate the feedback arrow from hardware testing to simulation and training. End on the library name, with the robot in a small supporting inset.

**Closing text:** **“simtoreal — Train. Convert. Configure. Deploy. Test.”**

## Recording and editing notes

- Keep the eight segments within their allocated time; adjust narration pace and visual holds to finish at exactly 3:00. Scene labels and editor instructions are not spoken.
- Capture the library files and YAML in a readable editor with a large font. Show only a few highlighted lines at a time.
- If showing generated output, regenerate with a matching model in a separate output folder. The checked-in `library/main.py` and Python policy are older examples and do not match the current five-input configuration.
- Show an actual conversion or upload only when recorded. Otherwise label the graphic “Workflow” or “Deployment files”.
- The generator connects existing drivers; it does not automatically create arbitrary hardware drivers. `.mpy` is compiled bytecode, not a claim of weight quantization or a guaranteed compression ratio.
- Use subtitles, short captions, and low background audio. Avoid adding unmeasured speed, memory, or success-rate claims.
