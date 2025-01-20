import torch
import torch.nn as nn
from src.models.DQN import DQN

class Qatten(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Qatten, self).__init__()
        self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])
        self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=4)
        self.mixing_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.num_agents = num_agents

    def forward(self, states, global_state):
        """
        Args:
            states: Lista tensorów [num_agents x state_dim].
            global_state: Tensor globalnego stanu [state_dim].
        Returns:
            global_q: Wartość Q globalna.
        """
        states = torch.stack(states, dim=0)  # [num_agents x state_dim]
        attention_output, _ = self.attention(states, states, states)  # Mechanizm uwagi
        q_values = torch.mean(attention_output, dim=0)  # Średnia wartość uwagi [state_dim]

        global_q = self.mixing_network(global_state + q_values)  # Łączenie z globalnym stanem
        return global_q

def create_qatten(state_dim, action_dim, num_agents):
    """Funkcja pomocnicza do tworzenia modelu Qatten."""
    return Qatten(state_dim, action_dim, num_agents)