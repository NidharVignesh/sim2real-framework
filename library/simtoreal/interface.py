import yaml
from pathlib import Path

def generate_interface(config_path: str, output_path: str = "policy_interface.h"):
    """
    Generate a C header that maps sensors → NN input and NN output → actuators
    using the external YAML config.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    observations = cfg.get("observations", [])
    actions = cfg.get("actions", [])

    lines = []
    lines.append("// Auto-generated hardware interface")
    lines.append("// Do not edit manually - regenerate from config")
    lines.append("#pragma once")
    lines.append('#include "policy_network.h"')
    lines.append("#include <math.h>")
    lines.append("")
    lines.append("static inline float clampf(float v, float lo, float hi) {")
    lines.append("    return v < lo ? lo : (v > hi ? hi : v);")
    lines.append("}")
    lines.append("")

    # -------------------- fill_observation --------------------
    lines.append("/**")
    lines.append(" * Fill the observation vector from sensors.")
    lines.append(" * Replace the placeholder reads with your real sensor functions.")
    lines.append(" */")
    lines.append("static inline void fill_observation(float* obs) {")

    for i, obs in enumerate(observations):
        name   = obs.get("name", f"obs_{i}")
        source = obs.get("source", "sensor")
        scale  = float(obs.get("scale", 1.0))
        offset = float(obs.get("offset", 0.0))
        index  = obs.get("index", i)

        lines.append(f"    // [{i}] {name}  (source: {source})")
        lines.append(f"    // TODO: replace with real sensor read")
        lines.append(f"    float raw_{i} = 0.0f;  // <-- read_{source}({index})")
        lines.append(f"    obs[{i}] = raw_{i} * {scale}f + {offset}f;")
        lines.append("")

    lines.append("}")
    lines.append("")

    # -------------------- apply_action --------------------
    lines.append("/**")
    lines.append(" * Send network output to actuators.")
    lines.append(" * Replace the placeholder writes with your real actuator functions.")
    lines.append(" */")
    lines.append("static inline void apply_action(const float* action) {")

    for i, act in enumerate(actions):
        name   = act.get("name", f"action_{i}")
        target = act.get("target", "motor")
        scale  = float(act.get("scale", 1.0))
        offset = float(act.get("offset", 0.0))
        min_v  = float(act.get("min", -1.0))
        max_v  = float(act.get("max",  1.0))
        index  = act.get("index", i)

        lines.append(f"    // [{i}] {name}  → {target}")
        lines.append(f"    float a{i} = clampf(action[{i}] * {scale}f + {offset}f, {min_v}f, {max_v}f);")
        lines.append(f"    // TODO: replace with real actuator write")
        lines.append(f"    // set_{target}({index}, a{i});")
        lines.append("")

    lines.append("}")
    lines.append("")

    # -------------------- policy_step --------------------
    lines.append("/**")
    lines.append(" * Full control step: read sensors → run policy → write actuators")
    lines.append(" */")
    lines.append("static inline void policy_step(void) {")
    lines.append("    float obs[POLICY_INPUT_SIZE];")
    lines.append("    float action[POLICY_OUTPUT_SIZE];")
    lines.append("")
    lines.append("    fill_observation(obs);")
    lines.append("    policy_forward(obs, action);")
    lines.append("    apply_action(action);")
    lines.append("}")
    lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"Generated hardware interface: {output_path}")