from pathlib import Path
from typing import Optional

from .exporter import generate_header, generate_mpy_network, generate_python_network
from .interface import generate_interface, load_config
from .loaders import load_sb3_policy, load_torch_policy

LANG_SUFFIX = {"mpy": ".mpy", "python": ".py", "c": ".h"}
SUFFIX_LANG = {suffix: lang for lang, suffix in LANG_SUFFIX.items()}


def convert(model_path: str, output_path: Optional[str] = None, config_path: Optional[str] = None,
            lang: Optional[str] = None, micropython_version: Optional[str] = None):
    """
    Convert a trained policy to a file the microcontroller can run.

    Parameters
    ----------
    model_path : str
        Path to .zip (SB3) or .pt/.pth (PyTorch)
    output_path : str, optional
        Output file. Default: policy_network.mpy (or .py / .h to match `lang`).
    config_path : str, optional
        YAML hardware description; if given, main.py is generated next to the output file.
    lang : str, optional
        "mpy"    -> policy_network.mpy (precompiled MicroPython, the default)
        "python" -> policy_network.py  (MicroPython source)
        "c"      -> policy_network.h   (C header)
        If omitted, it is taken from the output file's extension, else "mpy".
    micropython_version : str, optional
        Only for "mpy": build for an older MicroPython firmware (e.g. "1.22").
    """
    if lang is None:
        lang = SUFFIX_LANG.get(Path(output_path).suffix, "mpy") if output_path else "mpy"
    if lang not in LANG_SUFFIX:
        raise ValueError(f"lang must be one of {list(LANG_SUFFIX)}, got {lang!r}")
    output_path = Path(output_path or "policy_network").with_suffix(LANG_SUFFIX[lang])

    path = Path(model_path)
    if path.suffix == ".zip":
        network = load_sb3_policy(model_path)
    elif path.suffix in {".pt", ".pth"}:
        network = load_torch_policy(model_path)
    else:
        raise ValueError("Unsupported model format. Use .zip (SB3) or .pt (PyTorch)")

    if config_path is not None:
        config = load_config(config_path)
        n_obs, n_act = len(config["observations"]), len(config["actions"])
        if n_obs != network["input_size"] or n_act != network["output_size"]:
            raise ValueError(
                f"{config_path} lists {n_obs} observations and {n_act} actions, but the policy has "
                f"{network['input_size']} inputs and {network['output_size']} outputs")

    if lang == "mpy":
        generate_mpy_network(network, str(output_path), micropython_version)
    elif lang == "python":
        generate_python_network(network, str(output_path))
    else:
        generate_header(network, str(output_path))

    if config_path is not None:
        generate_interface(config_path, str(output_path.with_name("main.py")))

    return str(output_path)
