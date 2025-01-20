import os
import torch
import random
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
        return random.choice(range(model.fc3.out_features))  # Losowa akcja
    else:
        state = (
            torch.FloatTensor(state).to(device).unsqueeze(0)
        )  # Dodanie wymiaru batcha
        model.eval()
        with torch.no_grad():
            q_values = model(state)
        return torch.argmax(q_values).item()  # Akcja z najwyższą wartością Q


def train_step(
    batch, model, target_model, optimizer, gamma, num_agents=1, model_type="dqn"
):
    """
    Krok treningowy - aktualizacja wag sieci.

    Args:
        batch: Partia danych (stan, akcja, nagroda, kolejny stan, zakończenie epizodu).
        model: Aktualny model DQN/VDN/QMIX/Qatten.
        target_model: Model docelowy.
        optimizer: Optymalizator.
        gamma: Współczynnik dyskontowania.
        num_agents: Liczba agentów (domyślnie 1 dla DQN).
        model_type: Typ modelu ("dqn", "vdn", "qmix", "qatten").
    """
    # Rozpakowanie batcha
    states, actions, rewards, next_states, dones = zip(*batch)

    if model_type == "dqn":
        # Pojedynczy agent (DQN)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = target_model(next_states).max(dim=1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    elif model_type in ["vdn", "qmix", "qatten"]:
        # Wielu agentów
        states = [
            torch.FloatTensor([s[i] for s in states if len(s) > i]).to(device)
            for i in range(num_agents)
        ]
        next_states = [
            torch.FloatTensor([ns[i] for ns in next_states if len(ns) > i]).to(device)
            for i in range(num_agents)
        ]
        actions = [
            torch.LongTensor([a[i] for a in actions if len(a) > i]).to(device)
            for i in range(num_agents)
        ]
        rewards = [
            torch.FloatTensor([r[i] for r in rewards if len(r) > i]).to(device)
            for i in range(num_agents)
        ]
        dones = torch.FloatTensor([d for d in dones]).to(device)

        if model_type == "vdn":
            # VDN: Suma wartości Q dla wszystkich agentów
            q_values = torch.stack(
                [
                    model.agents[i](states[i])
                    .gather(1, actions[i].unsqueeze(1))
                    .squeeze(1)
                    for i in range(num_agents)
                ],
                dim=0,
            ).sum(dim=0)

            with torch.no_grad():
                next_q_values = torch.stack(
                    [
                        target_model.agents[i](next_states[i]).max(dim=1)[0]
                        for i in range(num_agents)
                    ],
                    dim=0,
                ).sum(dim=0)

            # print("states shape:", [s.shape for s in states])
            # print("actions shape:", [a.shape for a in actions])
            # print("rewards shape:", [r.shape for r in rewards])
            # print("next_q_values shape:", next_q_values.shape)


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
            # Qatten: Mechanizm uwagi
            q_values = model(states)
            with torch.no_grad():
                next_q_values = target_model(next_states)

        # Suma nagród dla wszystkich agentów
        # targets = torch.stack(rewards, dim=0).sum(dim=0) + gamma * next_q_values * (
        #     1 - dones
        # )
        max_len = max(r.size(0) for r in rewards)
        rewards = [torch.nn.functional.pad(r, (0, max_len - r.size(0))) for r in rewards]
        next_q_values = next_q_values[:max_len]

        rewards = torch.stack(rewards, dim=0)  # Kształt: (num_agents, max_len)
        dones = dones[:max_len]  # Dopasowanie wymiarów
        targets = rewards.sum(dim=0) + gamma * next_q_values * (1 - dones)

    # Oblicz stratę
    loss = torch.nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_model(model, path, reward, best_reward):
    """
    Zapisuje model, jeśli osiągnął najlepszy wynik.
    Args:
        model: Model DQN.
        path: Ścieżka do zapisu modelu.
        reward: Aktualna nagroda.
        best_reward: Najlepsza nagroda.
    Returns:
        Zaktualizowana najlepsza nagroda.
    """
    if reward > best_reward:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"New best model saved with reward: {reward}")
        return reward
    return best_reward


def plot_rewards(rewards, save_path=None, title="Training Progress"):
    """
    Wykres nagród dla danego eksperymentu.

    Args:
        rewards (list): Lista nagród dla kolejnych epizodów.
        save_path (str): Opcjonalna ścieżka do zapisu wykresu.
        title (str): Tytuł wykresu.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward", color="b")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
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
