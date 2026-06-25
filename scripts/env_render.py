import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import envs.pick_single_ycb_ur10e, envs.pick_single_ycb_xarm6_robotiq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="PickSingleYCBXArm6Robotiq-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(
        args.env_name,
        obs_mode="rgb",
        control_mode=args.control_mode,
        render_mode="human",
    )

    env.reset(seed=args.seed)
    viewer = env.render()

    try:
        while True:
            viewer = env.render()
            if viewer is not None and getattr(viewer.window, "closed", False):
                break
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
