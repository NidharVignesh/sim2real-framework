from pathlib import Path
import numpy as np

def _array_to_c(name: str, array: np.ndarray, dtype: str = "float") -> str:
    flat = array.flatten()
    values = ", ".join(f"{x:.8f}f" for x in flat)
    return f"const {dtype} {name}[{len(flat)}] = {{ {values} }};\n"

def generate_header(network: dict, output_path: str, header_guard: str = "POLICY_NETWORK_H"):
    layers = network["layers"]
    input_size = network["input_size"]
    output_size = network["output_size"]
    activation = network.get("activation", "tanh")

    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("#include <math.h>")
    lines.append("")
    lines.append(f"#define POLICY_INPUT_SIZE  {input_size}")
    lines.append(f"#define POLICY_OUTPUT_SIZE {output_size}")
    lines.append("")

    # Write weights and biases
    for i, layer in enumerate(layers):
        w = layer["weight"]
        b = layer["bias"]
        lines.append(_array_to_c(f"policy_w{i}", w))
        if b is not None:
            lines.append(_array_to_c(f"policy_b{i}", b))
        lines.append("")

    # Activation helper
    if activation == "tanh":
        lines.append("static inline float act(float x) { return tanhf(x); }")
    else:
        lines.append("static inline float act(float x) { return x > 0.0f ? x : 0.0f; } // ReLU")
    lines.append("")

    # Inside generate_header, replace the forward function generation with:

    # Find maximum layer size for a safe temporary buffer
    max_size = max(input_size, output_size)
    for layer in layers:
        max_size = max(max_size, layer["weight"].shape[0])

    lines.append("static inline void policy_forward(const float* input, float* output) {")
    lines.append(f"    float buf_a[{max_size}];")
    lines.append(f"    float buf_b[{max_size}];")
    lines.append(f"    float* curr = buf_a;")
    lines.append(f"    float* next = buf_b;")
    lines.append("")
    lines.append(f"    // Copy input")
    lines.append(f"    for (int i = 0; i < {input_size}; i++) curr[i] = input[i];")
    lines.append("")

    prev_size = input_size
    for i, layer in enumerate(layers):
        w = layer["weight"]
        out_size = w.shape[0]
        in_size  = w.shape[1]
        is_last  = (i == len(layers) - 1)

        lines.append(f"    // Layer {i}  ({in_size} → {out_size})")
        lines.append(f"    for (int o = 0; o < {out_size}; o++) {{")
        lines.append(f"        float sum = policy_b{i}[o];")
        lines.append(f"        for (int j = 0; j < {in_size}; j++) {{")
        lines.append(f"            sum += policy_w{i}[o * {in_size} + j] * curr[j];")
        lines.append(f"        }}")
        if is_last:
            lines.append(f"        next[o] = sum;  // linear output")
        else:
            lines.append(f"        next[o] = act(sum);")
        lines.append("    }")
        lines.append("")
        lines.append("    // swap buffers")
        lines.append("    float* tmp = curr; curr = next; next = tmp;")
        lines.append("")

    lines.append(f"    for (int i = 0; i < {output_size}; i++) output[i] = curr[i];")
    lines.append("}")
    lines.append("")
    lines.append(f"#endif // {header_guard}")

    Path(output_path).write_text("\n".join(lines))
    print(f"Generated: {output_path}")