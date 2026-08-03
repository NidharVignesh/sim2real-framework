import argparse
from .converter import convert

def main():
    parser = argparse.ArgumentParser(description="Convert RL policy to C header for ESP32")
    parser.add_argument("model", help="Path to model (.zip or .pt)")
    parser.add_argument("-o", "--output", default="policy_network.h", help="Output .h file")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config(YAML) file (optional)")
    args = parser.parse_args()

    convert(args.model, args.output, args.config)

if __name__ == "__main__":
    main()