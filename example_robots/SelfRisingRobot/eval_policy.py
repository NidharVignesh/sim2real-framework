"""Benchmark a trained robo1 policy: success rate and time-to-upright per fall case.

Success        the goal pose (upright, level, servos ~0) is held for HOLD_STEPS
               (0.8 s) before the 14 s episode limit.
Time-to-upright  seconds from the start until the goal pose is first reached
               and then held (successful episodes only).

Usage:
    python eval_policy.py                                  # robo1_getup_ppo.zip, 50 episodes per case
    python eval_policy.py --model best_model/best_model.zip --episodes 100
    python eval_policy.py --domain-rand                    # test robustness on randomized physics
    python eval_policy.py --render --episodes 3            # watch it
    python eval_policy.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from robo1_env import HOLD_STEPS, START_CASES, Robo1GetupEnv

HERE = Path(__file__).resolve().parent


def fmt_s(x: float, width: int) -> str:
    return f"{'-':>{width}}" if np.isnan(x) else f"{x:>{width - 1}.2f}s"


def run_case(model: PPO, case: str, episodes: int, domain_rand: bool, render: bool, seed: int) -> dict:
    env = Robo1GetupEnv(domain_randomization=domain_rand, render_mode="human" if render else None)
    times, successes, final_upright = [], 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep, options={"case": case})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render:
                time.sleep(env.dt)
        final_upright.append(info["upright"])
        if terminated:
            successes += 1
            times.append((env.step_count - HOLD_STEPS + 1) * env.dt)
    env.close()
    return {
        "case": case,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "time_mean": float(np.mean(times)) if times else float("nan"),
        "time_median": float(np.median(times)) if times else float("nan"),
        "time_std": float(np.std(times)) if times else float("nan"),
        "time_max": float(np.max(times)) if times else float("nan"),
        "final_upright_mean": float(np.mean(final_upright)),
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default=str(HERE / "robo1_getup_ppo.zip"))
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per case")
    parser.add_argument("--cases", nargs="+", default=list(START_CASES), choices=START_CASES)
    parser.add_argument("--domain-rand", action="store_true", help="Randomized physics + IMU noise")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json", type=str, default=None, help="Write results to this file")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"Model: {args.model} | physics: {'randomized' if args.domain_rand else 'nominal'} "
          f"| {args.episodes} episodes per case\n")
    header = f"{'case':<11}{'success':>9}{'t_mean':>9}{'t_median':>10}{'t_std':>8}{'t_max':>8}{'upright':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for case in args.cases:
        r = run_case(model, case, args.episodes, args.domain_rand, args.render, args.seed)
        results.append(r)
        print(f"{case:<11}{r['success_rate']:>9.1%}{fmt_s(r['time_mean'], 9)}{fmt_s(r['time_median'], 10)}"
              f"{fmt_s(r['time_std'], 8)}{fmt_s(r['time_max'], 8)}{r['final_upright_mean']:>9.3f}")

    fallen = [r for r in results if r["case"] != "upright"]
    if fallen:
        n = sum(r["episodes"] for r in fallen)
        succ = sum(r["success_rate"] * r["episodes"] for r in fallen)
        t = [r["time_mean"] for r in fallen if not np.isnan(r["time_mean"])]
        print("-" * len(header))
        print(f"{'all falls':<11}{succ / n:>9.1%}{fmt_s(float(np.mean(t)) if t else float('nan'), 9)}")

    if args.json:
        Path(args.json).write_text(json.dumps({"model": args.model, "domain_rand": args.domain_rand,
                                               "results": results}, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
