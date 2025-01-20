import torch
import torch.nn as nn
from src.models.DQN import DQN

class QMIX(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(QMIX, self).__init__()
        self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])
        self.hypernetwork = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_agents * 2)  # Generuje wagi i biasy
        )
        self.state_dim = state_dim
        self.num_agents = num_agents

    def forward(self, states, global_state):
        """
        Args:
            states: Lista tensorów [num_agents x state_dim].
            global_state: Tensor globalnego stanu [state_dim].
        Returns:
            global_q: Wartość Q globalna.
        """
        q_values = torch.stack([agent(state) for agent, state in zip(self.agents, states)], dim=0)  # [num_agents x batch_size x action_dim]
        q_values_max = q_values.max(dim=2)[0]  # Maksymalne Q dla każdej akcji [num_agents x batch_size]

        weights_biases = self.hypernetwork(global_state)  # Wagi i biasy z sieci hiperparametrów
        weights, biases = weights_biases.chunk(2, dim=-1)

        weights = weights.view(-1, self.num_agents, 1)  # Dopasowanie do Q
        biases = biases.view(-1, 1)

        global_q = (q_values_max.unsqueeze(-1) * weights).sum(dim=1) + biases  # Monotoniczna agregacja
        return global_q


def create_qmix(state_dim, action_dim, num_agents):
    """Funkcja pomocnicza do tworzenia modelu QMIX."""
    return QMIX(state_dim, action_dim, num_agents)