import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from env import make_floor_is_lava_env
from agents.dqn import DQN, ReplayBuffer
from agents.tabq import TabQ


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

elif str.lower(args.algo) == "dqn":
    device = torch.device("cpu")

    # Environment info
    obs_shape = env.observation_space.shape  # (C, H, W)
    n_actions = env.action_space.n

    # Networks
    policy_net = DQN(obs_shape[0], obs_shape[1], obs_shape[2], n_actions).to(device)
    target_net = DQN(obs_shape[0], obs_shape[1], obs_shape[2], n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayBuffer(max_experiences=10000)

    # Hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 5000
    TARGET_UPDATE = 1000

    steps_done = {"count": 0}

    def select_action(state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done["count"] / EPS_DECAY)
        steps_done["count"] += 1
        if np.random.rand() < eps_threshold:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = state.unsqueeze(0).to(device)
                return policy_net(state_t).argmax(dim=1).item()

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

        states      = states.to(device)
        actions     = actions.unsqueeze(1).to(device)
        rewards     = rewards.to(device)
        next_states = next_states.to(device)
        dones       = dones.to(device)

        # Current Q values
        state_action_values = policy_net(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_state_values = target_net(next_states).max(1)[0]
            expected_state_action_values = rewards + GAMMA * next_state_values * (1 - dones)

        # Loss calculation
        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training loop
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        state = torch.tensor(obs, dtype=torch.float32)
        done, total_reward = False, 0

        while not done:
            action = select_action(state)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            next_state = torch.tensor(next_obs, dtype=torch.float32)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model()

            # Update target network
            if steps_done["count"] % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep+1}: Total reward = {total_reward:.3f}")

    env.close()

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