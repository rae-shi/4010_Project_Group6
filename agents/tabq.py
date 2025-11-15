from typing import Callable, Tuple
import numpy as np
import random

def epsilon_greedy(Q, state, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    return int(np.argmax(Q[state]))

def TabQ(
    env,
    gamma: float,
    step_size: float,          
    epsilon: float,
    max_episodes: int,
    n_states: int,
    state_id_fn: Callable[[np.ndarray], int],   # obs -> int in [0, n_states)
    seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
      env: Gymnasium env (discrete actions)
      gamma, step_size (alpha), epsilon, max_episodes
      n_states: size of tabular state space
      state_id_fn: encoder from observation to state id
    Returns:
      Pi: np.ndarray[int] of shape (n_states,)  greedy policy
      Q:  np.ndarray[float] of shape (n_states, n_actions)
    """
    A = env.action_space.n
    Q = np.zeros((n_states, A), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for _ in range(max_episodes):
        obs, _ = env.reset(seed = seed)
        s = state_id_fn(obs)
        done = False
        while not done:
            a = epsilon_greedy(Q, s, epsilon, A, rng)
            obs2, r, term, trunc, _ = env.step(a)
            s2 = state_id_fn(obs2)
            done = term or trunc

            td_target = r + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s, a] += step_size * (td_target - Q[s, a])
            s = s2

    Pi = np.argmax(Q, axis=1).astype(np.int32)
    return Pi, Q
