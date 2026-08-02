from pathlib import Path
from .loaders import load_sb3_policy, load_torch_policy
from .exporter import generate_header

def convert(model_path: str, output_path: str = "policy_network.h"):
    """
    Convert a trained policy to a C header file.

    Parameters
    ----------
    model_path : str
        Path to .zip (SB3) or .pt (PyTorch)
    output_path : str
        Where to write the .h file
    """
    path = Path(model_path)

    if path.suffix == ".zip":
        network = load_sb3_policy(model_path)
    elif path.suffix in {".pt", ".pth"}:
        network = load_torch_policy(model_path)
    else:
        raise ValueError("Unsupported model format. Use .zip (SB3) or .pt (PyTorch)")

    generate_header(network, output_path)
    return output_path