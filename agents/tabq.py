from typing import Callable, Tuple, List, Dict, Any
import numpy as np

def epsilon_greedy(Q, state, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    # Random tie-breaking for exploration
    q_values = Q[state]
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return int(rng.choice(best_actions))

def TabQ(
    env,
    gamma: float,
    step_size: float,          
    epsilon: float,
    max_episodes: int,
    n_states: int,
    state_id_fn: Callable[[np.ndarray], int],
    seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Standard Tabular Q-learning.
    """
    A = env.action_space.n
    rng = np.random.default_rng(seed)
    
    # Initialization
    Q = np.zeros((n_states, A), dtype=np.float32)
    results = [] 
    total_step_counter = 0

    for ep in range(max_episodes):
        obs, _ = env.reset(seed=seed)
        s = state_id_fn(obs)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # Use fixed epsilon
            a = epsilon_greedy(Q, s, epsilon, A, rng)
            obs2, r, term, trunc, _ = env.step(a)
            s2 = state_id_fn(obs2)
            done = term or trunc
            last_step_reward = r

            if term:
                td_target = r
            else:
                td_target = r + gamma * np.max(Q[s2])

            Q[s, a] += step_size * (td_target - Q[s, a])
            
            s = s2
            total_reward += r
            steps += 1
            total_step_counter += 1

        # Record metrics
        is_success = 1 if (term and last_step_reward > 0) else 0
        results.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "success": is_success,
            "total_steps": total_step_counter,
            "epsilon": epsilon
        })
        
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1}: Reward={total_reward:.2f}, Steps={steps}, Success={is_success}, Total Steps={total_step_counter}, Eps={epsilon:.3f}")

    Pi = np.argmax(Q, axis=1).astype(np.int32)
    return Pi, Q, results