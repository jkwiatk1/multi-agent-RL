import torch
import torch.nn as nn
from src.models.DQN import DQN


class VDN(nn.Module):
    """
    VDN zakłada, że dla każdego agenta mamy wartość Q,
    ale sumujemy te wartości w sposób liniowy w celu obliczenia globalnej wartości Q.
    """

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
            global_q: Suma wartości Q dla wszystkich agentów. tensor: [batch_size, action_dim]
        """
        q_values = [
            agent(state) for agent, state in zip(self.agents, states)
        ]  # q_values: lista o  długości num_agents złozona z tensorów [batch_size , action_dim]
        global_q = torch.stack(q_values, dim=0).sum(
            dim=0
        )  # tensor: [batch_size, action_dim]
        return global_q


def create_vdn(state_dim, action_dim, num_agents):
    """Funkcja pomocnicza do tworzenia modelu VDN."""
    return VDN(state_dim, action_dim, num_agents)
