import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, model, epsilon=0.1):
    """
    Wybór akcji na podstawie epsilon-greedy.
    Args:
        state: Obserwacja ze środowiska.
        model: Model DQN.
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
        model: Aktualny model DQN/VDN/QMIX/Qatten.
        target_model: Model docelowy.
        optimizer: Optymalizator.
        discount_factor_g: Współczynnik dyskontowania.
        num_agents: Liczba agentów (domyślnie 1 dla DQN).
        model_type: Typ modelu ("dqn", "vdn", "qmix", "qatten").
    """
    # Rozpakowanie batcha
    states, actions, rewards, next_states, dones = zip(*batch)

    if model_type == "dqn":
        # Pojedynczy agent (DQN)
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.tensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # orginal DQN- target zwraca Q dla kolejnej akcji
        # q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # with torch.no_grad():
        #     next_q_values = target_model(next_states).max(dim=1)[0]

        # double DQN - target zwraca Q dla akcji wybranej przez model bazowy (podobno lepsze)
        q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_acts = model(next_states).max(1)[1]
            next_q_values = (
                target_model(next_states)
                .gather(1, next_state_acts.unsqueeze(-1))
                .squeeze(-1)
            )

        targets = rewards + discount_factor_g * next_q_values * (1 - dones)

    elif model_type in ["vdn", "qmix", "qatten"]:
        # Wielu agentów
        states = [
            torch.tensor([s[i] for s in states if len(s) > i]).to(device)
            for i in range(num_agents)
        ]
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
            # 2101
            # q_values = torch.stack(
            #     [
            #         model.agents[i](states[i])
            #         .gather(1, actions[i].unsqueeze(1))
            #         .squeeze(1)
            #         for i in range(num_agents)
            #     ],
            #     dim=0,
            # ).sum(dim=0)
            q_values = model(states)  # [batch_size, action_dim]

            # orginal DQN
            # with torch.no_grad():
            #     next_q_values = torch.stack(
            #         [
            #             target_model.agents[i](next_states[i]).max(dim=1)[0]
            #             for i in range(num_agents)
            #         ],
            #         dim=0,
            #     ).sum(dim=0)

            # Double DQN
            with torch.no_grad():
                next_q_values = target_model(
                    next_states
                )  # tensor [batch_size, action_dim]
                # 2101
                # next_state_acts = torch.stack(
                #     [
                #         model.agents[i](next_states[i]).max(1)[1]
                #         for i in range(num_agents)
                #     ],
                #     dim=0,
                # )
                # next_q_values = torch.stack(
                #     [
                #         target_model.agents[i](next_states[i])
                #         .gather(1, next_state_acts[i].unsqueeze(-1))
                #         .squeeze(-1)
                #         for i in range(num_agents)
                #     ],
                #     dim=0,
                # ).sum(dim=0)

            # # Sumuj nagrody globalnie
            global_rewards = sum(rewards)  # tensor [batch_size]
            # # Sumowanie nagród z uwzględnieniem brakujących wartości
            # fill_value = min([min(r) for r in rewards])  # Wartość do uzupełnienia braków
            # max_agents = len(batch)
            # # Wypełnianie brakujących nagród -1
            # rewards_padded = [
            #     torch.cat([r, torch.full((max_agents - len(r),), fill_value).to(device)])
            #     if len(r) < max_agents else r for r in rewards
            # ]
            # # Przekształcenie na tensor
            # rewards_tensor = torch.stack(rewards_padded, dim=0)  # [batch_size, num_agents]
            # # Sumowanie nagród po agentach (z wypełnionym -1)
            # global_rewards = rewards_tensor.sum(dim=0)

            # Oblicz maksymalne wartości Q dla następnych stanów
            next_q_values_max = torch.max(next_q_values, dim=1)[
                0
            ]  # tensor [batch_size]

            # Oblicz Q_target
            q_targets = global_rewards + discount_factor_g * next_q_values_max * (
                1 - dones
            )  # tensor o rozmiarze [batch_size]

            # Rozpakuj akcje dla każdego agenta
            actions_combined = torch.stack(
                actions, dim=0
            )  # tensor [num_agents, batch_size]

            # Wybierz wartości Q dla wykonanych akcji
            q_values_taken = q_values.gather(1, actions_combined.T).sum(
                dim=1
            )  # [batch_size]

        elif model_type == "qmix":
            # QMIX: Monotoniczne mieszanie wartości Q
            agent_q_values = torch.stack(
                [
                    model.agents[i](states[i])
                    .gather(1, actions[i].unsqueeze(1))
                    .squeeze(1)
                    for i in range(num_agents)
                ],
                dim=0,
            )
            with torch.no_grad():
                next_agent_q_values = torch.stack(
                    [
                        target_model.agents[i](next_states[i]).max(dim=1)[0]
                        for i in range(num_agents)
                    ],
                    dim=0,
                )

            global_state = torch.cat(states, dim=-1)
            global_next_state = torch.cat(next_states, dim=-1)
            q_values = model.mixing_network(global_state, agent_q_values)
            next_q_values = target_model.mixing_network(
                global_next_state, next_agent_q_values
            )

        elif model_type == "qatten":
            # Qatten: Mechanizm uwagi z globalnym stanem
            agent_q_values = torch.stack(
                [
                    model.agents[i](states[i])
                    .gather(1, actions[i].unsqueeze(1))
                    .squeeze(1)
                    for i in range(num_agents)
                ],
                dim=0,
            )

            # Przygotowanie globalnych stanów
            global_state = torch.cat(states, dim=-1)
            global_next_state = torch.cat(next_states, dim=-1)

            # Mechanizm uwagi i mieszanie
            q_values = model(global_state, agent_q_values)
            with torch.no_grad():
                next_agent_q_values = torch.stack(
                    [
                        target_model.agents[i](next_states[i]).max(dim=1)[0]
                        for i in range(num_agents)
                    ],
                    dim=0,
                )
                next_q_values = target_model(global_next_state, next_agent_q_values)

        # Suma nagród dla wszystkich agentów
        # targets = torch.stack(rewards, dim=0).sum(dim=0) + gamma * next_q_values * (
        #     1 - dones
        # )
        # max_len = max(r.size(0) for r in rewards)
        # rewards = [
        #     torch.nn.functional.pad(r, (0, max_len - r.size(0))) for r in rewards
        # ]
        # next_q_values = next_q_values[:max_len]
        #
        # rewards = torch.stack(rewards, dim=0)  # (num_agents, max_len)
        # dones = dones[:max_len]
        # targets = rewards.sum(dim=0) + discount_factor_g * next_q_values * (1 - dones)

    # loss = torch.nn.MSELoss()(q_values, targets)
    loss = torch.nn.MSELoss()(q_values_taken, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_model(model, path):
    """
    Zapisuje model
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
    Wykres dla danego eksperymentu.

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
