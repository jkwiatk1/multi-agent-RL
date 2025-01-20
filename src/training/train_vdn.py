import os
import torch
import random
from collections import deque
from src.models.VDN import create_vdn
from src.environments.mpe import create_environment, close_environment
from src.utils import select_action, train_step, save_model, plot_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parametry treningowe
params = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.99,
    "epsilon": 0.1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 4,
    "learning_rate": 0.001,
    "num_episodes": 100,
    "model_save_path": "../results/vdn_mpe_model/",
    "render": False,
}


def train_vdn(params):
    env = create_environment(render=params["render"], api="parallel")
    num_agents = len(env.agents)  # Liczba agentów w środowisku
    model = create_vdn(
        params["state_dim"], params["action_dim"], num_agents=num_agents
    ).to(device)
    target_model = create_vdn(
        params["state_dim"], params["action_dim"], num_agents=num_agents
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    replay_buffer = deque(maxlen=10000)
    best_reward = -float("inf")
    rewards = []
    losses = []
    epsilon = params["epsilon"]

    for episode in range(params["num_episodes"]):
        observations, infos = env.reset()
        total_reward = 0

        epsilon = max(params["epsilon_min"], epsilon * params["epsilon_decay"])

        while env.agents:
            # Wybierz akcje dla wszystkich agentów
            actions = {
                agent: select_action(
                    observations[agent], model.agents[i], epsilon
                )
                for i, agent in enumerate(env.agents)
            }

            # Wykonaj akcje
            new_observations, rewards_dict, terminations, truncations, infos = env.step(
                actions
            )
            total_reward += sum(rewards_dict.values())

            # Zapisz doświadczenia do bufora replay
            replay_buffer.append(
                (
                    [observations[agent] for agent in env.agents],  # Stany
                    [actions[agent] for agent in env.agents],  # Akcje
                    [rewards_dict[agent] for agent in env.agents],  # Nagrody
                    [new_observations[agent] for agent in env.agents],  # Nowe stany
                    any(terminations.values())
                    or any(truncations.values()),  # Czy epizod zakończony
                )
            )

            # Przejdź do nowych obserwacji
            observations = new_observations

            # Trenuj model
            if len(replay_buffer) >= params["batch_size"]:
                batch = random.sample(replay_buffer, params["batch_size"])
                loss = train_step(
                    batch,
                    model,
                    target_model,
                    optimizer,
                    params["gamma"],
                    num_agents=num_agents,
                    model_type="vdn",
                )
                losses.append(loss)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        rewards.append(total_reward)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        best_reward = save_model(
            model,
            params["model_save_path"] + "best_vdn_model.pth",
            total_reward,
            best_reward,
        )

    close_environment(env)
    plot_rewards(
        rewards,
        save_path=params["model_save_path"] + "vdn_training_rewards.png",
        title="VDN Training Rewards",
    )

    plot_rewards(losses, save_path=params["model_save_path"] + "vdn_training_losses.png", title="Training Losses")


if __name__ == "__main__":
    train_vdn(params)
