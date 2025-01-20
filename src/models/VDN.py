import torch
import torch.nn as nn
from src.models.DQN import DQN


class VDN(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(VDN, self).__init__()
        self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])
        self.num_agents = num_agents

    def forward(self, states):
        """
        Args:
            states: Lista tensorów [num_agents x state_dim].
        Returns:
            global_q: Suma wartości Q dla wszystkich agentów.
        """
        print(f"VDN Forward pass: state shape = {states.shape}")
        q_values = [agent(state) for agent, state in zip(self.agents, states)]
        global_q = torch.stack(q_values, dim=0).sum(dim=0)  # Suma wartości Q
        return global_q

def create_vdn(state_dim, action_dim, num_agents):
    """Funkcja pomocnicza do tworzenia modelu VDN."""
    return VDN(state_dim, action_dim, num_agents)
