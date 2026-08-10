import argparse
from .converter import convert
    
def main():
    parser = argparse.ArgumentParser(description="Convert RL policy to Micropython / C header for ESP32")
    parser.add_argument("model", help="Path to model (.zip or .pt)")
    parser.add_argument("-o", "--output", default="policy_network.py", help="Output network file (default: policy_network.py)")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config(YAML) file (optional,generates main.py)")
    parser.add_argument("-l", "--lang", choices=["python", "c"], default="python", help="Output language: python (MicroPython) or c (C header) (default: python)")

    args = parser.parse_args()

    convert(args.model, args.output, args.config, args.lang)

if __name__ == "__main__":
    main()