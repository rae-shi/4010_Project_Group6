import argparse
import numpy as np
from env import make_floor_is_lava_env
from agents.tabq import TabQ
# from agents.dqn import DQN

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["tabq", "dqn", "random"], required=True)
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=20)
args = parser.parse_args()

env = make_floor_is_lava_env(
    dynamic=args.dynamic,
    use_shaping=not args.sparse,
    seed=args.seed,
    render_mode="human"
)

if args.algo == "tabq":
    
    def state_id_fn(obs):
        C, H, W = obs.shape
        agent_layer = obs[4]  # if channel 4 is agent
        y, x = np.argwhere(agent_layer == 1.0)[0]
        return y * W + x

    n_states = env.unwrapped.width * env.unwrapped.height

    Pi, Q = TabQ(
        env=env,
        gamma=0.99,
        step_size=0.5,
        epsilon=0.1,
        max_episodes=args.episodes,
        n_states=n_states,
        state_id_fn=state_id_fn,
        seed=args.seed
    )

    print("Training complete.")
    print("Sample policy (first 20 states):", Pi[:20])

elif args.algo == "random":
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

