import argparse
from .converter import convert


def main():
    parser = argparse.ArgumentParser(description="Convert an RL policy to MicroPython (.mpy / .py) or a C header for ESP32")
    parser.add_argument("model", help="Path to model (.zip or .pt)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: policy_network.mpy). Its extension picks the format if --lang is not given.")
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to config (YAML) file (optional, generates main.py next to the output)")
    parser.add_argument("-l", "--lang", choices=["mpy", "python", "c"], default=None,
                        help="mpy: precompiled MicroPython (default) | python: MicroPython source .py | c: C header .h")
    parser.add_argument("--micropython-version", default=None,
                        help="Build the .mpy for an older MicroPython firmware, e.g. 1.22 (default: newest supported)")

    args = parser.parse_args()

    convert(args.model, args.output, args.config, args.lang, args.micropython_version)


if __name__ == "__main__":
    main()
