import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, h1_nodes=64, h2_nodes=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, action_dim)

    def forward(self, state):
        # print(f"DQN Forward pass: state shape = {state.shape}")
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


def create_dqn(state_dim, action_dim):
    """Funkcja pomocnicza do tworzenia modelu DQN."""
    return DQN(state_dim, action_dim)
