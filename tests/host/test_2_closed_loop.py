"""Test 2 - Does the robot behave the same when the generated C code is in control?

Runs the MuJoCo get-up task twice with identical seeds: once driven by the original
PyTorch policy, once driven by the C header that simtoreal generated. Compares success
rate and time-to-upright, with nominal and with randomized physics.

Run from the repo root:  python tests/host/test_2_closed_loop.py
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

import common
from robo1_env import FALLEN_POSES, HOLD_STEPS, Robo1GetupEnv

CASES = (*FALLEN_POSES, "random")


def run_episodes(controller, case, episodes, domain_rand, seed, shadow=None):
    env = Robo1GetupEnv(domain_randomization=domain_rand)
    successes, times, max_diff = 0, [], 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep, options={"case": case})
        done = False
        while not done:
            action = controller(obs)
            if shadow is not None:
                max_diff = max(max_diff, float(np.max(np.abs(np.clip(action, -1, 1) - np.clip(shadow(obs), -1, 1)))))
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        if term:
            successes += 1
            times.append((env.step_count - HOLD_STEPS + 1) * env.dt)
    env.close()
    return {
        "success_rate": successes / episodes,
        "time_to_upright_mean_s": float(np.mean(times)) if times else None,
        "time_to_upright_std_s": float(np.std(times)) if times else None,
        "max_action_difference_vs_pytorch": max_diff if shadow is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    artifacts = common.build_artifacts()
    model = PPO.load(str(common.MODEL), device="cpu")
    c_policy, _ = common.compile_c_policy(artifacts["h"])

    def pytorch_controller(obs):
        return common.torch_actor_mean(model, obs[None, :])[0]

    def c_controller(obs):
        return c_policy(obs)

    report = {"episodes_per_case": args.episodes, "seed": args.seed, "physics": {}}
    for domain_rand in (False, True):
        physics = "randomized" if domain_rand else "nominal"
        print(f"\n=== {physics} physics ===")
        print(f"{'case':<11}{'PyTorch succ':>14}{'C succ':>9}{'PyTorch t':>11}{'C t':>8}{'max|da|':>10}")
        rows = {}
        for case in CASES:
            ref = run_episodes(pytorch_controller, case, args.episodes, domain_rand, args.seed)
            gen = run_episodes(c_controller, case, args.episodes, domain_rand, args.seed, shadow=pytorch_controller)
            rows[case] = {"pytorch_policy": ref, "generated_c_policy": gen}
            fmt_t = lambda r: f"{r['time_to_upright_mean_s']:.2f}s" if r["time_to_upright_mean_s"] is not None else "-"
            print(f"{case:<11}{100 * ref['success_rate']:>13.1f}%{100 * gen['success_rate']:>8.1f}%"
                  f"{fmt_t(ref):>11}{fmt_t(gen):>8}{gen['max_action_difference_vs_pytorch']:>10.1e}")

        agg = {}
        for key in ("pytorch_policy", "generated_c_policy"):
            agg[key] = {
                "success_rate": float(np.mean([rows[c][key]["success_rate"] for c in CASES])),
                "time_to_upright_mean_s": float(np.mean([rows[c][key]["time_to_upright_mean_s"] for c in CASES
                                                         if rows[c][key]["time_to_upright_mean_s"] is not None])),
            }
        rows["all_falls_aggregate"] = agg
        print(f"{'all falls':<11}{100 * agg['pytorch_policy']['success_rate']:>13.1f}%"
              f"{100 * agg['generated_c_policy']['success_rate']:>8.1f}%"
              f"{agg['pytorch_policy']['time_to_upright_mean_s']:>10.2f}s"
              f"{agg['generated_c_policy']['time_to_upright_mean_s']:>7.2f}s")
        report["physics"][physics] = rows

    common.save_result("host_test2_closed_loop_equivalence", report)


if __name__ == "__main__":
    main()
