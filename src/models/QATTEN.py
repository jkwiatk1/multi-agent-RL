import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DQN import DQN


class Qatten(nn.Module):
    """
    Qatten model: Wieloagentowy model Q-learning z mechanizmem uwagi.
    """

    def __init__(self, state_dim, action_dim, num_agents, attention_dim=32):
        """
        Args:
            state_dim (int): Rozmiar stanu wejściowego dla każdego agenta.
            action_dim (int): Rozmiar przestrzeni akcji dla każdego agenta.
            num_agents (int): Liczba agentów.
            attention_dim (int): Rozmiar osadzeń w mechanizmie uwagi.
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = attention_dim
        self.agents = nn.ModuleList(
            [DQN(state_dim, action_dim) for _ in range(num_agents)]
        )

        # Warstwy osadzeń dla uwagi
        self.query_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, attention_dim),
                    nn.ReLU(),
                    nn.Linear(attention_dim, attention_dim),
                )
                for _ in range(num_agents)
            ]
        )

        self.key_layers = nn.ModuleList(
            [nn.Linear(state_dim, attention_dim) for _ in range(num_agents)]
        )

        # Warstwy do skalowania wartości Q
        self.scaling_layer = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
        )

        # Warstwa dla wartości c
        self.constraint_layer = nn.Sequential(
            nn.Linear(state_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, states):
        """
        Args:
            states (List[torch.Tensor]): Lista stanów agentów, każdy o wymiarze [batch_size, state_dim].
        Returns:
            global_q_values (torch.Tensor): Wynikowe globalne wartości Q [batch_size, action_dim].
        """
        # batch_size = states.size(0)
        q_values = torch.stack(
            [agent(state) for agent, state in zip(self.agents, states)], dim=1
        )  # [batch_size, num_agents, action_dim]

        # Embeddingi uwagi (klucze i zapytania)
        keys = torch.stack(
            [key_layer(state) for key_layer, state in zip(self.key_layers, states)],
            dim=1,
        )  # [batch_size, num_agents, attention_dim]

        queries = torch.stack(
            [
                query_layer(state)
                for query_layer, state in zip(self.query_layers, states)
            ],
            dim=1,
        )  # [batch_size, num_agents, attention_dim]

        # Mechanizm uwagi
        attention_scores = torch.einsum("bnd,bmd->bnm", queries, keys) / (
            self.embed_dim**0.5
        )
        attention_weights = F.softmax(
            attention_scores, dim=-1
        )  # [batch_size, num_agents, num_agents]

        # Skalowanie wartości Q agentów
        weighted_qs = torch.einsum(
            "bnm,bna->bna", attention_weights, q_values
        )  # [batch_size, num_agents, action_dim]

        scaling_factors = torch.sigmoid(
            torch.stack([self.scaling_layer(state) for state in states], dim=1)
        )  # [batch_size, num_agents, 1]

        scaled_q_values = (
                scaling_factors * weighted_qs
        )  # [batch_size, num_agents, action_dim]
        summed_q_values = torch.mean(scaled_q_values, dim=1)
        # summed_q_values = scaled_q_values.sum(dim=1)  # [batch_size, action_dim]

        # Bias
        c_values = torch.stack([self.constraint_layer(state) for state in states],
                               dim=1)  # [batch_size, num_agents, attention_dim]
        c_values = torch.mean(c_values, dim=1)
        # c_values = self.constraint_layer(
        #     states[0]
        # )  # Zakładając, że wartości c są obliczane na podstawie jednego stanu (można rozważyć średnią)
        global_q_values = summed_q_values + c_values

        return global_q_values


def create_qatten(state_dim, action_dim, num_agents, attention_dim=128):
    return Qatten(state_dim, action_dim, num_agents, attention_dim)
