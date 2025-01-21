import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.DQN import DQN


class Qatten(nn.Module):
    """
    Qatten to model z mechanizmem uwagi, który dynamicznie przypisuje wagi do wartości Q poszczególnych agentów.
    Wartości Q są agregowane z użyciem mechanizmu uwagi (attention mechanism).
    """

    def __init__(self, state_dim, action_dim, num_agents, attention_dim=128):
        super().__init__()
        # Lista agentów z indywidualnymi sieciami DQN
        self.agents = nn.ModuleList(
            [DQN(state_dim, action_dim) for _ in range(num_agents)]
        )
        # Waga uwagi dla każdego agenta
        self.attention = nn.ModuleList(
            [nn.Linear(action_dim, attention_dim) for _ in range(num_agents)]
        )
        self.attention_out = nn.Linear(attention_dim, 1)  # Wynik mechanizmu uwagi
        self.num_agents = num_agents

    def forward(self, states):
        """
        Args:
            states: Lista tensorów o długości num_agents złożona z [batch_size, state_dim].
        Returns:
            q_values: Tensor wartości Q dla każdego agenta przed agregacja, o rozmiarze [num_agents, batch_size, action_dim].
            weighted_q: Agregowana wartość Q przy użyciu mechanizmu uwagi dla wszystkich agentów, uwzględniająca mechanizm uwagi.
        """
        q_values = [
            agent(state) for agent, state in zip(self.agents, states)
        ]  # Lista o długości num_agents, każdy tensor o rozmiarze [batch_size, action_dim]

        # Przekształcenie wartości Q za pomocą mechanizmu uwagi
        attention_weights = []
        for i in range(self.num_agents):
            q_value = q_values[i]  # tensor [batch_size, action_dim]
            attention_input = F.relu(
                self.attention[i](q_value)
            )  # Przekształcamy Q agenta przez warstwę liniową attention_input: [batch_size, attention_dim]
            attention_weight = self.attention_out(
                attention_input
            )  # Obliczamy wagę dla agenta attention_weight: [batch_size,1]
            attention_weights.append(
                attention_weight
            )  # Lista wag dla agentów o długosci num_agents i rozmiarze [batch_size,1]

        # Stosujemy mechanizm uwagi: dla każdego stanu, agentom przypisywane są różne wagi
        attention_weights = torch.cat(
            attention_weights, dim=1
        )  # Łączymy wagi agentów w jeden tensor o rozmiarze [batch_size,num_agents]
        attention_weights = F.softmax(
            attention_weights, dim=1
        )  # Normalizujemy wagi, sumując je do 1

        # Agregowanie wartości Q agentów z uwzględnieniem wag
        global_q = torch.stack(
            q_values, dim=0
        )  # tensor: [num_agents, batch_size, action_dim]
        weighted_q = torch.sum(
            global_q * attention_weights.permute(1, 0).unsqueeze(2), dim=0
        )  # Tensor: [num_agents, batch_size, action_dim] * [batch_size,num_agents].permute(1, 0).unsqueeze(2)
        # = [batch_size, action_dim]

        return global_q, weighted_q


def create_qatten(state_dim, action_dim, num_agents, attention_dim=128):
    """Funkcja pomocnicza do tworzenia modelu Qatten."""
    return Qatten(state_dim, action_dim, num_agents, attention_dim)


# class Qatten(nn.Module):
#     """
#     Qatten używa mechanizmu uwagi, by ważyć wpływ każdego agenta na globalną wartość Q
#     """
#     def __init__(self, state_dim, action_dim, num_agents):
#         super().__init__()
#         self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])
#         self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=2)
#         self.mixing_network = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
#         self.num_agents = num_agents
#
#     def forward(self,  global_state, states):
#         """
#         Args:
#             states: Lista tensorów [num_agents x state_dim].
#             global_state: Tensor globalnego stanu [batch_size, state_dim x num_agents].
#         Returns:
#             global_q: Wartość Q globalna.
#         """
#         # states = torch.stack(states, dim=0)  # [num_agents x state_dim]
#         attention_output, _ = self.attention(states, states, states)  # Mechanizm uwagi
#         q_values = torch.mean(attention_output, dim=0)  # Średnia wartość uwagi [state_dim]
#
#         global_q = self.mixing_network(global_state + q_values)  # Łączenie z globalnym stanem
#         return global_q
#
# def create_qatten(state_dim, action_dim, num_agents):
#     """Funkcja pomocnicza do tworzenia modelu Qatten."""
#     return Qatten(state_dim, action_dim, num_agents)
#
#
# class Qatten2(nn.Module):
#     """
#     Implementacja Qatten, która używa mechanizmu atencji do obliczenia ważonej sumy wartości Q dla agentów.
#     """
#     def __init__(self, state_dim, action_dim, num_agents, embedding_dim=64):
#         super().__init__()
#         self.num_agents = num_agents
#         self.agents = nn.ModuleList([DQN(state_dim, action_dim) for _ in range(num_agents)])
#
#         # Mechanizm atencji
#         self.key_layer = nn.Linear(state_dim, embedding_dim)
#         self.query_layer = nn.Linear(state_dim, embedding_dim)
#         self.value_layer = nn.Linear(state_dim, embedding_dim)
#
#         # Warstwa mieszająca
#         self.mixer = nn.Linear(embedding_dim, 1)
#
#     def forward(self, states):
#         """
#         Args:
#             states: Tensor o wymiarach [batch_size, state_dim x num_agents].
#         Returns:
#             global_q: Ważona suma wartości Q dla wszystkich agentów.
#         """
#         q_values = [agent(state) for agent, state in zip(self.agents, states)]  # [num_agents, action_dim]
#         q_values = torch.stack(q_values, dim=0)  # Tensor [num_agents, action_dim]
#
#         # Mechanizm atencji
#         keys = self.key_layer(states)  # [num_agents, embedding_dim]
#         queries = self.query_layer(states)  # [num_agents, embedding_dim]
#         values = self.value_layer(states)  # [num_agents, embedding_dim]
#
#         # Obliczenie wag atencji
#         attention_weights = F.softmax(torch.matmul(queries, keys.transpose(0, 1)), dim=-1)  # [num_agents, num_agents]
#
#         # Ważone wartości Q
#         attended_values = torch.matmul(attention_weights, values)  # [num_agents, embedding_dim]
#         global_q = self.mixer(attended_values.sum(dim=0))  # [1]
#
#         return global_q
#
#
# def create_qatten2(state_dim, action_dim, num_agents, embedding_dim=64):
#     """Funkcja pomocnicza do tworzenia modelu Qatten."""
#     return Qatten2(state_dim, action_dim, num_agents, embedding_dim)
