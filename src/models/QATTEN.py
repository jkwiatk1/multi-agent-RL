import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.DQN import DQN

class Qatten(nn.Module):
    """
    Qatten używa mechanizmu uwagi, by ważyć wpływ każdego agenta na globalną wartość Q
    """
    def __init__(self, state_dim, action_dim, num_agents):
        super().__init__()
        self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])
        self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=2)
        self.mixing_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.num_agents = num_agents

    def forward(self,  global_state, states):
        """
        Args:
            states: Lista tensorów [num_agents x state_dim].
            global_state: Tensor globalnego stanu [batch_size, state_dim x num_agents].
        Returns:
            global_q: Wartość Q globalna.
        """
        # states = torch.stack(states, dim=0)  # [num_agents x state_dim]
        attention_output, _ = self.attention(states, states, states)  # Mechanizm uwagi
        q_values = torch.mean(attention_output, dim=0)  # Średnia wartość uwagi [state_dim]

        global_q = self.mixing_network(global_state + q_values)  # Łączenie z globalnym stanem
        return global_q

def create_qatten(state_dim, action_dim, num_agents):
    """Funkcja pomocnicza do tworzenia modelu Qatten."""
    return Qatten(state_dim, action_dim, num_agents)


class Qatten2(nn.Module):
    """
    Implementacja Qatten, która używa mechanizmu atencji do obliczenia ważonej sumy wartości Q dla agentów.
    """
    def __init__(self, state_dim, action_dim, num_agents, embedding_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])

        # Mechanizm atencji
        self.key_layer = nn.Linear(state_dim, embedding_dim)
        self.query_layer = nn.Linear(state_dim, embedding_dim)
        self.value_layer = nn.Linear(state_dim, embedding_dim)

        # Warstwa mieszająca
        self.mixer = nn.Linear(embedding_dim, 1)

    def forward(self, states):
        """
        Args:
            states: Tensor o wymiarach [batch_size, state_dim x num_agents].
        Returns:
            global_q: Ważona suma wartości Q dla wszystkich agentów.
        """
        q_values = [agent(state) for agent, state in zip(self.agents, states)]  # [num_agents, action_dim]
        q_values = torch.stack(q_values, dim=0)  # Tensor [num_agents, action_dim]

        # Mechanizm atencji
        keys = self.key_layer(states)  # [num_agents, embedding_dim]
        queries = self.query_layer(states)  # [num_agents, embedding_dim]
        values = self.value_layer(states)  # [num_agents, embedding_dim]

        # Obliczenie wag atencji
        attention_weights = F.softmax(torch.matmul(queries, keys.transpose(0, 1)), dim=-1)  # [num_agents, num_agents]

        # Ważone wartości Q
        attended_values = torch.matmul(attention_weights, values)  # [num_agents, embedding_dim]
        global_q = self.mixer(attended_values.sum(dim=0))  # [1]

        return global_q


def create_qatten2(state_dim, action_dim, num_agents, embedding_dim=64):
    """Funkcja pomocnicza do tworzenia modelu Qatten."""
    return Qatten2(state_dim, action_dim, num_agents, embedding_dim)