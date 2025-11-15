import argparse
import numpy as np
from env import make_floor_is_lava_env
# from agents.tabq import TabQ
# from agents.dqn import DQN

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["tabq", "dqn", "random"], required=True)
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=20)
args = parser.parse_args()

env = make_floor_is_lava_env(
    dynamic=args.dynamic,
    use_shaping=not args.sparse,
    seed=args.seed,
    render_mode="human"
)

if args.algo == "random":
    for ep in range(5):
        obs, info = env.reset(seed=args.seed)
        done, total_reward = False, 0
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_reward += r
            done = term or trunc
        print(f"Episode {ep+1}: Total reward = {total_reward:.3f}")
    env.close()
    exit()

