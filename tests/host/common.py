"""Shared paths and helpers for the host-side (laptop) tests."""

import ctypes
import datetime
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"
RESULTS = TESTS / "results"
BUILD = TESTS / "build"
ROBOT_DIR = REPO / "example_robots" / "SelfRisingRobot"
MODEL = ROBOT_DIR / "robo1_getup_ppo.zip"

sys.path.insert(0, str(REPO / "library"))
sys.path.insert(0, str(ROBOT_DIR))

OBS_LOW = np.array([-np.pi, -np.pi, -1.5, -1.55, -1.55], dtype=np.float32)
OBS_HIGH = -OBS_LOW


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_result(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "test": name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": str(MODEL.relative_to(REPO)),
        "host": f"{platform.system()} {platform.machine()} / Python {platform.python_version()}",
        **data,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results -> {path.relative_to(REPO)}")
    return path


CONFIG = REPO / "library" / "simtoreal" / "config.yaml"


def build_artifacts() -> dict:
    """Run the simtoreal library exactly as a user would, writing its outputs into tests/build/.

    The default output (.mpy) is built together with main.py from config.yaml - the files you
    upload to the board. The .py and .h versions are built too, for the equivalence tests.
    """
    from simtoreal.converter import convert

    BUILD.mkdir(parents=True, exist_ok=True)
    mpy_path = convert(str(MODEL), str(BUILD / "policy_network.mpy"), config_path=str(CONFIG))
    py_path = convert(str(MODEL), str(BUILD / "policy_network.py"), lang="python")
    h_path = convert(str(MODEL), str(BUILD / "policy_network.h"), lang="c")
    return {"mpy": Path(mpy_path), "main": BUILD / "main.py", "py": Path(py_path), "h": Path(h_path)}


def load_python_policy(py_path: Path):
    spec = importlib.util.spec_from_file_location("generated_policy_network", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_c_policy(h_path: Path):
    """Compile the generated header with gcc (strict warnings) and return (callable, compiler_warnings)."""
    harness = BUILD / "c_harness.c"
    harness.write_text(
        '#include "policy_network.h"\n'
        "void run_policy(const float* in, float* out) { policy_forward(in, out); }\n"
    )
    lib = BUILD / "libpolicy.so"
    cmd = ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC",
           "-I", str(h_path.parent), "-o", str(lib), str(harness), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gcc failed to compile the generated header:\n{proc.stderr}")
    warnings = [line for line in proc.stderr.splitlines() if "warning:" in line]

    clib = ctypes.CDLL(str(lib))
    fn = clib.run_policy
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    fn.restype = None

    def policy(obs: np.ndarray, n_out: int = 2) -> np.ndarray:
        inp = np.ascontiguousarray(obs, dtype=np.float32)
        out = np.zeros(n_out, dtype=np.float32)
        fn(inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
           out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out

    return policy, warnings


def torch_actor_mean(model, obs_batch: np.ndarray) -> np.ndarray:
    """Raw (unclipped) deterministic action of the SB3 actor: the exact function the exporter translates."""
    import torch

    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32))
        latent = model.policy.mlp_extractor.policy_net(t)
        return model.policy.action_net(latent).numpy()


def collect_rollout_observations(model, episodes_per_case: int = 4, seed: int = 777) -> np.ndarray:
    """Observations the policy actually sees in simulation (more realistic than uniform random inputs)."""
    from robo1_env import FALLEN_POSES, Robo1GetupEnv

    env = Robo1GetupEnv()
    obs_all = []
    for case in (*FALLEN_POSES, "random"):
        for ep in range(episodes_per_case):
            obs, _ = env.reset(seed=seed + ep, options={"case": case})
            done = False
            while not done:
                obs_all.append(obs.copy())
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
    env.close()
    return np.array(obs_all, dtype=np.float32)
