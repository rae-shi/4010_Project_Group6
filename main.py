import gymnasium as gym
import minigrid
from minigrid.core.constants import OBJECT_TO_IDX
from gymnasium import spaces
import numpy as np
from helpers import manhattan, A_LEFT, A_RIGHT, A_FORWARD

# (1) Restrict actions
class ActionRestrictWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._allowed = [A_LEFT, A_RIGHT, A_FORWARD]
        self.action_space = spaces.Discrete(len(self._allowed))

    def action(self, act):
        return self._allowed[int(act)]


# (2) Convert full obs to multi-channel tensor
class GridChannelsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.H, self.W = env.unwrapped.width, env.unwrapped.height
        C = 6  # empty, wall, lava, goal, agent, time
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(C, self.H, self.W), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"]
        H, W, _ = img.shape
        OBJ = img[..., 0]

        empty = (OBJ == OBJECT_TO_IDX["empty"]).astype(np.float32)
        wall = (OBJ == OBJECT_TO_IDX["wall"]).astype(np.float32)
        lava = (OBJ == OBJECT_TO_IDX["lava"]).astype(np.float32)
        goal = (OBJ == OBJECT_TO_IDX["goal"]).astype(np.float32)

        # agent position
        ax, ay = self.unwrapped.agent_pos
        agent = np.zeros((H, W), dtype=np.float32)
        agent[ay, ax] = 1.0

        # time channel
        t = self.unwrapped.step_count
        T = self.unwrapped.max_steps
        time_frac = np.full((H, W), t / max(1, T), dtype=np.float32)

        stacked = np.stack([empty, wall, lava, goal, agent, time_frac], axis=0)
        return stacked


# (3) Reward shaping (potential-based)
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, use_shaping=True, step_penalty=-0.01, phi_scale=0.02):
        super().__init__(env)
        self.use_shaping = use_shaping
        self.step_penalty = step_penalty
        self.phi_scale = phi_scale
        self.prev_phi = 0.0
        self.goal_xy = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        grid = self.env.unwrapped.grid
        gx = gy = None
        for y in range(self.env.unwrapped.height):
            for x in range(self.env.unwrapped.width):
                cell = grid.get(x, y)
                if cell and cell.type == "goal":
                    gx, gy = x, y
                    break
            if gx is not None:
                break
        self.goal_xy = (gx, gy) if gx is not None else None

        if self.use_shaping and self.goal_xy is not None:
            ax, ay = self.env.unwrapped.agent_pos
            d = manhattan((ax, ay), self.goal_xy)
            self.prev_phi = -d
        else:
            self.prev_phi = 0.0

        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)

        if not self.use_shaping:
            return obs, r, terminated, truncated, info

        shaped = r + self.step_penalty
        if self.goal_xy is not None:
            ax, ay = self.env.unwrapped.agent_pos
            d = manhattan((ax, ay), self.goal_xy)
            phi = -d
            shaped += self.phi_scale * (phi - self.prev_phi)
            self.prev_phi = phi

        return obs, shaped, terminated, truncated, info


# (4) Env builder
def make_floor_is_lava_env(render_mode="human",
                           max_steps=200,
                           use_shaping=True,
                           phi_scale=0.02,
                           step_penalty=-0.01,
                           seed=0):
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0",
                   render_mode=render_mode,
                   max_steps=max_steps)
    env.reset(seed=seed)
    env = minigrid.wrappers.FullyObsWrapper(env)
    env = ActionRestrictWrapper(env)
    env = RewardShapingWrapper(env,
                               use_shaping=use_shaping,
                               step_penalty=step_penalty,
                               phi_scale=phi_scale)
    env = GridChannelsWrapper(env)
    return env


# (5) Quick test 
if __name__ == "__main__":
    env = make_floor_is_lava_env(render_mode="human",
                                 use_shaping=True,
                                 max_steps=200,
                                 seed=40)

    obs, info = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        print(f"Reward: {rew:.3f}")
    env.close()
