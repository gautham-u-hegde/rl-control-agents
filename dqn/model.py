import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Simple Deep Q-Network for CartPole.
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)