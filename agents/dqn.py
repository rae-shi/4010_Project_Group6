from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels: int, height: int, width: int, n_actions: int):
        super().__init__()
        self.input_shape = (input_channels, height, width)
        self.n_actions = n_actions

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out(self.input_shape)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def _get_conv_out(self, shape):
            with torch.no_grad():
                dummy = torch.zeros(1, *shape)
                o = self.convolutional_layers(dummy)
                return o.numel() // o.size(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional_layers(x) # passes in through the convolutional layers
        x = torch.flatten(x, start_dim=1)
        q_values = self.fully_connected_layers(x) # passes the flattened convoluted vector into the fully connected layers giving recommending move to make
        return q_values

class ReplayBuffer:
    def __init__(self, max_experiences=1000):
        self.buffer = deque(maxlen=max_experiences)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)