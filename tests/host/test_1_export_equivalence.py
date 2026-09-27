"""Test 1 - Does the code simtoreal generates compute the same thing as the trained network?

Feeds the same inputs to (a) the original PyTorch actor, (b) the generated MicroPython
module (run here under CPython) and (c) the generated C header (compiled with gcc), and
reports how far apart their outputs are.

Run from the repo root:  python tests/host/test_1_export_equivalence.py
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

import common


def error_stats(reference: np.ndarray, candidate: np.ndarray) -> dict:
    err = np.abs(reference - candidate)
    return {
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "clipped_action_agreement_1e-4": float(np.mean(
            np.all(np.abs(np.clip(reference, -1, 1) - np.clip(candidate, -1, 1)) < 1e-4, axis=1))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-inputs", type=int, default=2000)
    args = parser.parse_args()

    artifacts = common.build_artifacts()
    model = PPO.load(str(common.MODEL), device="cpu")
    n_params = sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters()) + \
        sum(p.numel() for p in model.policy.action_net.parameters())

    rng = np.random.default_rng(0)
    random_obs = rng.uniform(common.OBS_LOW, common.OBS_HIGH,
                             size=(args.random_inputs, 5)).astype(np.float32)
    print("Collecting observations from real simulated rollouts...")
    rollout_obs = common.collect_rollout_observations(model)

    py_module = common.load_python_policy(artifacts["py"])
    c_policy, c_warnings = common.compile_c_policy(artifacts["h"])

    results = {}
    for label, obs_set in (("uniform_random_inputs", random_obs), ("simulated_rollout_inputs", rollout_obs)):
        reference = common.torch_actor_mean(model, obs_set)
        py_out = np.array([py_module.policy_forward(o.tolist()) for o in obs_set], dtype=np.float64)
        c_out = np.array([c_policy(o) for o in obs_set], dtype=np.float64)
        results[label] = {
            "n_inputs": int(len(obs_set)),
            "generated_micropython_module": error_stats(reference, py_out),
            "generated_c_header": error_stats(reference, c_out),
        }

    header_bytes = artifacts["h"].stat().st_size
    py_bytes = artifacts["py"].stat().st_size
    summary = {
        "network": "5-64-64-2 MLP, tanh hidden activations, linear output",
        "actor_parameters": int(n_params),
        "float32_weight_bytes": int(n_params * 4),
        "generated_c_header_source_bytes": header_bytes,
        "generated_micropython_source_bytes": py_bytes,
        "gcc_flags": "-std=c99 -O2 -Wall -Wextra",
        "gcc_compiled": True,
        "gcc_warnings": c_warnings,
        "results": results,
    }

    print(f"\nActor parameters: {n_params}  (float32 weights: {n_params * 4} bytes)")
    print(f"gcc -Wall -Wextra warnings: {len(c_warnings)}")
    for label, r in results.items():
        print(f"\n[{label}]  n={r['n_inputs']}")
        for backend in ("generated_micropython_module", "generated_c_header"):
            s = r[backend]
            print(f"  {backend:30s} max|err|={s['max_abs_error']:.2e}  mean|err|={s['mean_abs_error']:.2e}  "
                  f"agreement={100 * s['clipped_action_agreement_1e-4']:.2f}%")

    common.save_result("host_test1_export_equivalence", summary)


if __name__ == "__main__":
    main()
