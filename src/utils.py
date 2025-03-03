import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, model, epsilon=0.1):
    """
    Wybór akcji na podstawie epsilon-greedy.
    Args:
        state: Obserwacja ze środowiska.
        model: Model.
        epsilon: Wartość epsilon dla eksploracji.
    Returns:
        Wybrana akcja (int).
    """
    if random.random() < epsilon:
        return random.choice(range(model.fc3.out_features))
    else:
        state = (
            torch.FloatTensor(state).to(device).unsqueeze(0)
        )  # Dodanie wymiaru batcha
        model.eval()
        with torch.no_grad():
            q_values = model(state)
        return torch.argmax(q_values).item()  # Akcja z najwyższą wartością Q


def train_step(
    batch,
    model,
    target_model,
    optimizer,
    discount_factor_g,
    num_agents=1,
    model_type="dqn",
):
    """
    Krok treningowy - aktualizacja wag sieci.

    Args:
        batch: Partia danych (stan, akcja, nagroda, kolejny stan, zakończenie epizodu).
        model: Aktualny model DQN/VDN/Qatten.
        target_model: Model docelowy.
        optimizer: Optymalizator.
        discount_factor_g: Współczynnik dyskontowania.
        num_agents: Liczba agentów (domyślnie 1 dla DQN).
        model_type: Typ modelu ("dqn", "vdn", "qatten").
    """
    states, actions, rewards, next_states, dones = zip(*batch)

    if model_type == "dqn":
        # Pojedynczy agent (DQN)
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.tensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # double DQN - target zwraca Q dla akcji wybranej przez model bazowy (podobno lepsze)
        q_values_taken = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_acts = model(next_states).max(1)[1]
            next_q_values = (
                target_model(next_states)
                .gather(1, next_state_acts.unsqueeze(-1))
                .squeeze(-1)
            )

        q_targets = rewards + discount_factor_g * next_q_values * (1 - dones)

    elif model_type in ["vdn", "qatten"]:
        # Wielu agentów
        states = [
            torch.tensor([s[i] for s in states if len(s) > i]).to(device)
            for i in range(num_agents)
        ]
        # states_2 = torch.stack([torch.tensor(s).to(device) for s in states])
        # TODO try # [num_agents, batch_size, state_dim]
        # next_states = torch.stack([torch.tensor(ns).to(device) for ns in next_states], dim=0)
        next_states = [
            torch.tensor([ns[i] for ns in next_states if len(ns) > i]).to(device)
            for i in range(num_agents)
        ]
        actions = [
            torch.tensor([a[i] for a in actions if len(a) > i]).to(device)
            for i in range(num_agents)
        ]
        rewards = [
            torch.tensor([r[i] for r in rewards if len(r) > i]).to(device)
            for i in range(num_agents)
        ]
        dones = torch.FloatTensor([d for d in dones]).to(device)

        if model_type == "vdn":
            # VDN: Suma wartości Q dla wszystkich agentów
            global_rewards = sum(rewards)  # tensor [batch_size]
            q_values = model(states)  # [num_agents, batch_size, action_dim]

            # Obliczanie Q_expected
            actions_combined = torch.stack(
                actions, dim=0
            )  # tensor [num_agents, batch_size]

            # Wybieranie wartości Q na podstawie podjętych akcji (dla każdego agenta)
            q_values_taken_per_agent = torch.stack(
                [
                    q_values[i].gather(1, actions_combined[i].unsqueeze(-1)).squeeze(-1)
                    for i in range(model.num_agents)
                ],
                dim=0,
            )  # [num_agents, batch_size]
            # TODO try q_values_taken_per_agent = q_values.gather(2, actions_combined.unsqueeze(-1)).squeeze(-1)
            q_values_taken = q_values_taken_per_agent.sum(dim=0)  # [batch_size]

            # Obliczanie Q_target
            with torch.no_grad():
                next_q_values = target_model(
                    next_states
                )  # [num_agents, batch_size, action_dim]

                next_q_values_max = torch.max(next_q_values, dim=2)[
                    0
                ]  # [num_agents, batch_size]

            q_targets = global_rewards + discount_factor_g * (
                1 - dones
            ) * next_q_values_max.sum(
                dim=0
            )  # [batch_size]

        elif model_type == "qatten":
            global_rewards = sum(rewards)  # tensor [batch_size]
            q_values = model(states)  # [batch_size, action_dim]

            # Q_expected
            actions_combined = torch.stack(actions, dim=1)  # [batch_size, num_agents]
            # Indeksy globalnych akcji (dla wartości Q)
            global_action_indices = torch.argmax(actions_combined, dim=1).unsqueeze(
                -1
            )  # [batch_size, 1]

            # Wybieranie wartości Q na podstawie podjętych akcji (globalnie)
            q_values_taken = q_values.gather(1, global_action_indices).squeeze(
                -1
            )  # [batch_size]

            # Q_target
            with torch.no_grad():
                next_q_values = target_model(next_states)  # [batch_size, action_dim]
                next_q_values_max = torch.max(next_q_values, dim=1)[0]  # [batch_size]

            q_targets = (
                global_rewards + discount_factor_g * (1 - dones) * next_q_values_max
            )  # [batch_size]

    loss = torch.nn.MSELoss()(q_values_taken, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_model(model, path):
    """
    Args:
        model: Model.
        path: Ścieżka do zapisu modelu.

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved in: {path}")


def plot_rewards(
    rewards,
    save_path=None,
    title="Training Progress",
    ylabel="Total Reward",
    label="Total Reward",
    x_range=None,
    x_label="Episode",
):
    """
    Args:
        rewards (list): Lista wyników dla kolejnych epizodów.
        save_path (str): Opcjonalna ścieżka do zapisu wykresu.
        title (str): Tytuł wykresu.
        ylabel (str): Opis osi Y.
        label (str): Etykieta dla legendy.
        x_range (list or tuple): Zakres osi X w formacie [start, end].
                                 Jeśli None, używana jest domyślna numeracja epizodów.
        x_label (str): Opis osi X.
    """
    plt.figure(figsize=(10, 6))

    if x_range is not None:
        assert (
            len(x_range) == 2
        ), "x_range musi być listą lub krotką o dwóch elementach: [start, end]."
        x_values = np.linspace(x_range[0], x_range[1], len(rewards))
    else:
        x_values = range(len(rewards))

    plt.plot(x_values, rewards, label=label, color="b")
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
