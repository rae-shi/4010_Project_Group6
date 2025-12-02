import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
from env import make_floor_is_lava_env
from agents.heuristic import HeuristicAgent
from agents.tabq import TabQ
from agents.dqn import DQN, ReplayBuffer
from helpers import SHIFT_INTERVAL

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["tabq", "dqn", "ddqn", "heuristic"], required=True)
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic lava")
parser.add_argument("--sparse", action="store_true", help="Use sparse rewards (disable shaping)")
parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
parser.add_argument("--seed", type=int, default=20, help="Random seed")
parser.add_argument("--restart", action="store_true", help="Delete corresponding existing checkpoint/results and start fresh")
args = parser.parse_args()

def save_results(results, algo, dynamic, sparse, seed, extra_tag=""):
    """Saves episode metrics to CSV."""
    df = pd.DataFrame(results)
    
    dyn_str = "dynamic" if dynamic else "static"
    rew_str = "sparse" if sparse else "shaped"
    
    filename = f"results/{algo}_{dyn_str}_{rew_str}_s{seed}.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\nSaved results to: {filename}")

def get_checkpoint_path(algo, seed, dynamic, sparse):
    """Generates a unique filename for checkpoints to avoid collisions."""
    dyn_str = "dynamic" if dynamic else "static"
    rew_str = "sparse" if sparse else "shaped"
    
    # We only save checkpoints for DQN/DDQN (pth files)
    ext = "pth"
    
    # Example: checkpoints/ddqn_dynamic_shaped_s20.pth
    filename = f"checkpoints/{algo}_{dyn_str}_{rew_str}_s{seed}.{ext}"
    
    os.makedirs("checkpoints", exist_ok=True)
    return filename

def get_csv_path(algo, seed, dynamic, sparse):
    dyn_str = "dynamic" if dynamic else "static"
    rew_str = "sparse" if sparse else "shaped"
    filename = f"results/{algo}_{dyn_str}_{rew_str}_s{seed}.csv"
    os.makedirs("results", exist_ok=True)
    return filename

# --- Setup Environment ---
env = make_floor_is_lava_env(
    dynamic=args.dynamic,
    use_shaping=not args.sparse,
    seed=args.seed,
    render_mode=None # Keep human for demo, change to None for fast training
)

# --- 1. Tabular Q-Learning ---
if args.algo == "tabq":
    print(f"Running Tabular Q-Learning (Dynamic={args.dynamic}, Sparse={args.sparse})...")
    K = env.unwrapped.width - 2
    
    # ID = (y * W + x) * 4 + dir
    def state_id_fn(obs):
        # Get Spatial State (Position + Direction)
        direction = env.unwrapped.agent_dir 
        agent_layer = obs[4]
        y, x = np.argwhere(agent_layer == 1.0)[0]
        spatial_id = (y * env.unwrapped.width + x) * 4 + direction
        
        # Get Temporal State/Phase
        if args.dynamic:
            # Use the internal step count to know the phase
            step_count = env.unwrapped.step_count
            phase_index = (step_count // SHIFT_INTERVAL) % K
        else:
            # Static map always stays in Phase 0
            phase_index = 0
            
        # ID = (Spatial * Total_Phases) + Phase
        # make sure every (Pos, Dir) has a unique state for every Phase
        return int(spatial_id * K + phase_index)

    # Increase state space size to account for 4 directions
    n_states = env.unwrapped.width * env.unwrapped.height * 4 * K
    
    # Run TabQ
    Pi, Q, results = TabQ(
        env=env,
        gamma=0.99,
        step_size=0.1,
        epsilon=0.1,
        max_episodes=args.episodes,
        n_states=n_states,
        state_id_fn=state_id_fn,
        seed=args.seed
    )
    
    save_results(results, args.algo, args.dynamic, args.sparse, args.seed)
    print("Training complete. Results saved.")

# --- 2. Deep Q-Network (DQN & Double DQN) ---
elif args.algo in ["dqn", "ddqn"]:
    # Logic for DQN vs DDQN
    is_ddqn = (args.algo == "ddqn")
    print(f"Running {args.algo.upper()} (Dynamic={args.dynamic}, Sparse={args.sparse})...")
    
    # Check for GPU (Mac MPS or Nvidia CUDA) automatically
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (GPU) Acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU) Acceleration!")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    policy_net = DQN(obs_shape[0], obs_shape[1], obs_shape[2], n_actions).to(device)
    target_net = DQN(obs_shape[0], obs_shape[1], obs_shape[2], n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    memory = ReplayBuffer(max_experiences=50000)

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 100000
    TARGET_UPDATE = 1000
    steps_done = {"count": 0}

    results = []

    def select_action(state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done["count"] / EPS_DECAY)
        steps_done["count"] += 1
        if np.random.rand() < eps_threshold:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                return policy_net(state.unsqueeze(0).to(device)).argmax(dim=1).item()

    def optimize_model():
        if len(memory) < BATCH_SIZE: return
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

        states = states.to(device)
        actions = actions.unsqueeze(1).to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Q(s, a)
        state_action_values = policy_net(states).gather(1, actions)

        # Target Calculation
        with torch.no_grad():
            if is_ddqn:
                # Double DQN: Select action with PolicyNet, evaluate with TargetNet
                next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                next_state_values = target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: Max Q from TargetNet
                next_state_values = target_net(next_states).max(1)[0]
            
            expected_state_action_values = rewards + GAMMA * next_state_values * (1 - dones)

        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    chkpt_path = get_checkpoint_path(args.algo, args.seed, args.dynamic, args.sparse)
    csv_path = get_csv_path(args.algo, args.seed, args.dynamic, args.sparse)


    # --- RESTART LOGIC ---
    if args.restart:
        for path in [chkpt_path, csv_path]:
            if os.path.exists(path):
                print(f"-> Attempting to delete: {path}")
                try:
                    os.remove(path)
                    print("-> Deletion successful.")
                    print("WARNING: Please double check the .csv file is deleted. If not, please delete it manually.")
                except Exception as e:
                    print("-" * 50)
                    print(f"!! CRITICAL ERROR: Could not delete {os.path.basename(path)}")
                    print(f"!! Reason: {e}")
                    print("!! Please ensure the file is not open in another program (Terminal, Excel, etc.) and try again.")
                    print("-" * 50)
                    raise SystemExit("Checkpoint deletion failed. Halting execution.")
        
        print("Starting training from Episode 0.")

    # --- RESUME LOGIC ---
    start_episode = 0

    if os.path.exists(chkpt_path):
        print(f"Loading checkpoint from {chkpt_path}...")
        checkpoint = torch.load(chkpt_path)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get the episode number before processing the CSV
        start_episode = checkpoint['episode']
        print(f"Checkpoint found. Resuming from Episode {start_episode}")
        
        # Restore CSV history
        if os.path.exists(csv_path):
            print(f"Loading previous results from {csv_path}...")
            print("WARNING: Please double check the .csv file is deleted. If not, please delete it manually.")
            existing_df = pd.read_csv(csv_path)
            
            # Filter to ensure no duplicates
            existing_df = existing_df[existing_df["episode"] < start_episode]
            results = existing_df.to_dict('records')
            print(f"Restored {len(results)} episodes from history.")
            
        # Restore step count
        # Get from checkpoint
        if 'steps_done' in checkpoint:
            steps_done['count'] = checkpoint['steps_done']
        
        # Get from CSV
        elif results:
            print("Warning: 'steps_done' not in checkpoint. Restoring from CSV...")
            steps_done['count'] = results[-1]['total_steps']
            
        # Guess
        else:
            print("Warning: Step count lost. Estimating...")
            steps_done['count'] = start_episode * 100 # Assuming 100 steps/ep average
            
        print(f"Resuming steps_done: {steps_done['count']}")

    # Training Loop
    for ep in range(start_episode, args.episodes):
        obs, _ = env.reset(seed=args.seed) 
        state = torch.tensor(obs, dtype=torch.float32)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = select_action(state)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_state = torch.tensor(next_obs, dtype=torch.float32)

            memory.push(state, action, reward, next_state, term)
            state = next_state
            total_reward += reward
            steps += 1

            # Only train the network every 4 steps to save on computation
            if len(memory) > BATCH_SIZE and steps % 4 == 0:
                optimize_model()

            if steps_done["count"] % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # Save checkpoint every 100 episodes
        if (ep+1) % 100 == 0:
            print(f"Saving checkpoint to {chkpt_path}...")
            torch.save({
                'episode': ep+1,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'steps_done': steps_done['count']
            }, chkpt_path)

        # Record Metrics
        is_success = 1 if (term and reward > 0) else 0
        results.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "success": is_success,
            "total_steps": steps_done["count"]
        })
        
        if (ep+1) % 10 == 0:
            # Calculate current epsilon
            current_eps = EPS_END + (EPS_START - EPS_END) * \
                          np.exp(-1. * steps_done["count"] / EPS_DECAY)
            
            print(f"Ep {ep+1}: Reward={total_reward:.2f}, Steps={steps}, Success={is_success}, Total Steps={steps_done['count']}, Eps={current_eps:.3f}")

    env.close()
    save_results(results, args.algo, args.dynamic, args.sparse, args.seed)

# --- 3. Baselines (Heuristic) ---
elif args.algo == "heuristic":
    print(f"Running heuristic(Dynamic={args.dynamic}, Sparse={args.sparse})...")

    agent = HeuristicAgent(env)

    results = []
    total_step_counter = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.plan_step(env.unwrapped)
            
            obs, r, term, trunc, _ = env.step(action)
            total_reward += r
            steps += 1
            total_step_counter += 1
            done = term or trunc
            last_step_reward = r
            
        is_success = 1 if (term and last_step_reward > 0) else 0
        results.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "success": is_success,
            "total_steps": total_step_counter
        })

        if (ep+1) % 10 == 0:
            print(f"Ep {ep+1}: Reward={total_reward:.2f}, Steps={steps}, Success={is_success}, Total Steps={total_step_counter}")

    env.close()
    save_results(results, args.algo, args.dynamic, args.sparse, args.seed)