import gymnasium as gym
import minigrid
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import Lava, Floor
from gymnasium import spaces
import numpy as np
from helpers import manhattan, A_LEFT, A_RIGHT, A_FORWARD
import sys

# (1) Restrict actions
class ActionRestrictWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._allowed = [A_LEFT, A_RIGHT, A_FORWARD]
        self.action_space = spaces.Discrete(len(self._allowed))

    def action(self, act):
        return self._allowed[int(act)]


# (2) Convert full obs to multi-channel tensor (with phase support)
class GridChannelsWrapper(gym.ObservationWrapper):
    def __init__(self, env, phase_in_obs: bool = False, period=None, shift_interval=10):
        super().__init__(env)
        self.H, self.W = env.unwrapped.width, env.unwrapped.height
        self.phase_in_obs = phase_in_obs
        self.shift_interval = shift_interval  # same shift as DynamicLavaWrapper
        # Use grid width as period by default
        self.K = max(2, int(period if period is not None else self.W))
        # channels: empty, wall, lava, goal, agent, time/phase
        C = 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(C, self.H, self.W), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"]
        H, W, _ = img.shape
        OBJ = img[..., 0]

        empty = (OBJ == OBJECT_TO_IDX["empty"]).astype(np.float32)
        wall  = (OBJ == OBJECT_TO_IDX["wall"]).astype(np.float32)
        lava  = (OBJ == OBJECT_TO_IDX["lava"]).astype(np.float32)
        goal  = (OBJ == OBJECT_TO_IDX["goal"]).astype(np.float32)

        # agent position
        ax, ay = self.unwrapped.agent_pos
        agent = np.zeros((H, W), dtype=np.float32)
        agent[ay, ax] = 1.0

        # Time/phase channel
        if self.phase_in_obs:
            t = self.unwrapped.step_count
            # Match DynamicLavaWrapper's phase calculation
            phase_index = (t // self.shift_interval) % self.K
            phase = phase_index / (self.K - 1) if self.K > 1 else 0.0
            phase_chan = np.full((H, W), phase, dtype=np.float32)
        else:
            t = self.unwrapped.step_count
            T = self.unwrapped.max_steps
            phase_chan = np.full((H, W), t / max(1, T), dtype=np.float32)

        stacked = np.stack([empty, wall, lava, goal, agent, phase_chan], axis=0)
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

        # Don't shape lava death penalty - keep it clean at -1.0
        if info.get("lava_death", False):
            return obs, r, terminated, truncated, info

        shaped = r + self.step_penalty
        if self.goal_xy is not None:
            ax, ay = self.env.unwrapped.agent_pos
            d = manhattan((ax, ay), self.goal_xy)
            phi = -d
            shaped += self.phi_scale * (phi - self.prev_phi)
            self.prev_phi = phi

        return obs, shaped, terminated, truncated, info


# (4) Dynamic lava (periodic, Markov if phase is in observation)
class DynamicLavaWrapper(gym.Wrapper):
    """
    Periodic dynamic lava with shift mode. To keep the process Markov, 
    expose the phase (t // shift_interval) mod K in the observation (via GridChannelsWrapper.phase_in_obs=True).

    Mode: Original lava mask shifts one cell to the right each phase (wraps around)
    """
    def __init__(self, env, enabled=True, period=None, shift_interval=10):
        super().__init__(env)
        self.enabled = enabled
        # Will be set in reset based on grid width
        self.K = period
        self.shift_interval = shift_interval  # Lava shifts every N steps

        self._base_lava = None  # set of (x,y) that are lava in the base map
        self._W = None
        self._H = None
        self.current_phase = None  

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.enabled:
            return obs, info

        grid = self.env.unwrapped.grid
        self._W, self._H = self.env.unwrapped.width, self.env.unwrapped.height
        
        # Set period to interior width (excluding walls)
        if self.K is None:
            self.K = self._W - 2  # Interior columns only
        assert self.K >= 2, "period K must be >= 2"

        # Capture base lava positions
        base = set()
        for y in range(self._H):
            for x in range(self._W):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "lava":
                    base.add((x, y))
        self._base_lava = base

        # Ensure phase 0 grid is the base grid
        self._apply_phase(phase=0)
        self.current_phase = 0
        return obs, info

    def step(self, action):
        if self.enabled:
            # Apply phase BEFORE stepping so observation sees correct state
            t = self.env.unwrapped.step_count
            # Only shift every shift_interval steps
            next_phase = ((t + 1) // self.shift_interval) % self.K

            # Only apply if phase changed
            if next_phase != self.current_phase:
                self._apply_phase(next_phase)
                self.current_phase = next_phase
            
            # Check if lava appeared under the agent after phase shift
            ax, ay = self.env.unwrapped.agent_pos
            cell_under = self.env.unwrapped.grid.get(ax, ay)
            if cell_under is not None and cell_under.type == "lava":
                # Agent dies if lava appears under them
                # Get observation first, then force termination
                obs = self.env.unwrapped.gen_obs()
            
                # FORCE RENDER THE DEATH STATE
                if hasattr(self.env.unwrapped, 'render'):
                    self.env.unwrapped.render()
                    import time
                    time.sleep(0.5)  # Pause so you can see it
                
                return obs, -1.0, True, False, {"lava_death": True}
        
        # Step environment with updated grid
        obs, r, terminated, truncated, info = self.env.step(action)

        return obs, r, terminated, truncated, info

    def _apply_phase(self, phase: int):
        grid = self.env.unwrapped.grid
        W, H = self._W, self._H
        base = self._base_lava
        
        if base is None:
            return

        # First, identify all positions that currently have lava (may be shifted)
        current_lava = []
        for y in range(H):
            for x in range(W):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "lava":
                    current_lava.append((x, y))
        
        # Clear all current lava
        for (x, y) in current_lava:
            # Check if it's a wall or goal - don't remove those
            cell = grid.get(x, y)
            if cell is not None and cell.type in ["wall", "goal"]:
                continue
            grid.set(x, y, Floor())
        
        # Then set shifted positions to lava (shift base lava by phase)
        # Ensure x is remapped within interior columns [1, W-2] to avoid walls
        for (x, y) in base:
            # Map x to interior space, shift, then map back
            xs = 1 + ((x - 1 + phase) % (W - 2))
            # Don't overwrite goal
            cell = grid.get(xs, y)
            if cell is None or cell.type != "goal":
                grid.set(xs, y, Lava())


# (5) Env builder
def make_floor_is_lava_env(render_mode="human",
                           max_steps=200,
                           use_shaping=True,
                           phi_scale=0.02,
                           step_penalty=-0.01,
                           seed=0,
                           # NEW: dynamics options
                           dynamic=False,
                           phase_in_obs=True):
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0",
                   render_mode=render_mode,
                   max_steps=max_steps)
    env.reset(seed=seed)
    env = minigrid.wrappers.FullyObsWrapper(env)

    # Get grid width for period (use interior width to avoid walls)
    W = env.unwrapped.width
    interior_width = W - 2  # Exclude wall columns

    # Dynamic lava with shift mode (before shaping/obs so they see the updated grid)
    # shift_interval=10 means lava shifts position every 10 steps
    # period=interior_width means it takes interior_width shifts to complete a full cycle
    env = DynamicLavaWrapper(env,
                             enabled=bool(dynamic),
                             period=interior_width,
                             shift_interval=10)

    env = ActionRestrictWrapper(env)
    env = RewardShapingWrapper(env,
                               use_shaping=use_shaping,
                               step_penalty=step_penalty,
                               phi_scale=phi_scale)
    env = GridChannelsWrapper(env,
                              phase_in_obs=phase_in_obs,
                              period=interior_width,
                              shift_interval=10)
    return env


# (6) Test both static and dynamic environments
if __name__ == "__main__":
    
    mode = "dynamic"
    
    if mode == "dynamic":
        print("=" * 50)
        print("Testing DYNAMIC environment (lava shifts)")
        print("=" * 50)
        env = make_floor_is_lava_env(render_mode="human",
                                     use_shaping=True,
                                     max_steps=200,
                                     seed=40,
                                     dynamic=True,
                                     phase_in_obs=True)
    else:
        print("=" * 50)
        print("Testing STATIC environment (no lava movement)")
        print("=" * 50)
        env = make_floor_is_lava_env(render_mode="human",
                                     use_shaping=True,
                                     max_steps=200,
                                     seed=40,
                                     dynamic=False,
                                     phase_in_obs=False)

    obs, info = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        print(f"step={env.unwrapped.step_count:03d} reward={rew:.3f}")
    env.close()
