import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="human")

obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()   # random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()