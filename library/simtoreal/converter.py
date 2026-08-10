from pathlib import Path
from .loaders import load_sb3_policy, load_torch_policy
from .exporter import generate_header , generate_python_network
from .interface import generate_interface
from typing import Optional

def convert(model_path: str, output_path: str = "policy_network.py", config_path: Optional[str] = None,lang: str = "python"):
    """
    Convert a trained policy to a runnable network file.

    Parameters
    ----------
    model_path : str
        Path to .zip (SB3) or .pt (PyTorch)None
    output_path : str
        Where to write the .h file
    lang : str
        "python" -> policy_network.py (MicroPython)
        "c" -> policy_network.h 
    """
    path = Path(model_path)

    if path.suffix == ".zip":
        network = load_sb3_policy(model_path)
    elif path.suffix in {".pt", ".pth"}:
        network = load_torch_policy(model_path)
    else:
        raise ValueError("Unsupported model format. Use .zip (SB3) or .pt (PyTorch)")


    if lang == "python":
        if not str(output_path).endswith(".py"):
            output_path = str(Path(output_path).with_suffix(".py"))
        generate_python_network(network, output_path)
    elif lang == "c":
        if not str(output_path).endswith(".h"):
            output_path = str(Path(output_path).with_suffix(".h"))
        generate_header(network, output_path)

    if config_path is not None:
            interface_path = str(Path(output_path).with_name(
                Path(output_path).stem + "_interface.h"
            ))
            generate_interface(config_path, interface_path)

    return output_path