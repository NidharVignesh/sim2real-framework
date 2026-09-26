"""Load robo1 in the MuJoCo viewer with a trained policy driving the servos.

The robot starts from a random fall and the policy runs continuously, so it
gets back up whenever you knock it over (Ctrl + right-drag on a body to push it).

Usage:
    python play.py
    python play.py --model best_model/best_model.zip --case upright
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from robo1_env import START_CASES, Robo1GetupEnv

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default=str(HERE / "robo1_getup_ppo.zip"))
    parser.add_argument("--case", choices=START_CASES, default=None, help="Start case (default: random)")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = Robo1GetupEnv()
    obs, info = env.reset(options={"case": args.case})
    print(f"Start case: {info['case']}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, *_ = env.step(action)  # episode limits are ignored: runs until the window closes
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
