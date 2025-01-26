import torch
import torch.nn as nn

from src.models.DQN import DQN


class VDN(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super().__init__()
        self.agents = nn.ModuleList(
            [DQN(state_dim, action_dim) for _ in range(num_agents)]
        )
        self.num_agents = num_agents

    def forward(self, states):
        """
        Args:
            states: Lista tensorów o długości num_agents złożona z [batch_size , state_dim].
        Returns:
            q_values: Wartości Q dla wszystkich agentów.
        """
        q_values = [
            agent(state) for agent, state in zip(self.agents, states)
        ]  # q_values: lista [num_agents, [batch_size , action_dim]]
        q_values = torch.stack(
            q_values, dim=0
        )  # tensor: [num_agents, batch_size, action_dim]
        return q_values


def create_vdn(state_dim, action_dim, num_agents):
    return VDN(state_dim, action_dim, num_agents)
