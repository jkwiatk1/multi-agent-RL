import os
import time
import torch
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from src.models.VDN import create_vdn
from src.environments.mpe import create_environment, close_environment
from src.utils import select_action, train_step, save_model, plot_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parametry treningowe
params = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.99,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_episodes": 5000,
    "target_model_sync": 10,
    "model_save_path": "../results/vdn_mpe_model/",
    "render": False,
}


def train_vdn(params):
    experiment_path = params["model_save_path"] + f"{params['num_episodes']}_epochs/"
    os.makedirs(os.path.dirname(experiment_path), exist_ok=True)

    env = create_environment(render=params["render"], api="parallel")
    num_agents = len(env.agents)
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
    epsilon = params["epsilon"]

    writer = SummaryWriter(log_dir=experiment_path + "tensorboard_logs")
    start_time = time.time()

    # Statystyki treningowe
    rewards = []
    losses = []
    variances = []
    episode_times = []

    for episode in range(params["num_episodes"]):
        observations, infos = env.reset()
        episode_start = time.time()
        epsilon = max(params["epsilon_min"], epsilon * params["epsilon_decay"])

        total_reward = 0
        total_loss = 0

        while env.agents:
            actions = {
                agent: select_action(observations[agent], model.agents[i], epsilon)
                for i, agent in enumerate(env.agents)
            }

            new_observations, rewards_dict, terminations, truncations, infos = env.step(
                actions
            )
            if len(env.agents) == 0:
                continue

            total_reward += sum(rewards_dict.values())

            if any(terminations.values()) or any(truncations.values()):
                print("X")

            # Zapisz doświadczenia
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
                total_loss += loss

        losses.append(total_loss)
        rewards.append(total_reward)
        episode_times.append(time.time() - episode_start)
        writer.add_scalar("Episode/Reward", total_reward, episode)
        writer.add_scalar("Episode/Loss", total_loss, episode)
        writer.add_scalar("Episode/Time", episode_times[-1], episode)
        writer.add_scalar("Episode/Epsilon", epsilon, episode)
        # print(
        #     f"Episode {episode + 1}, Total Reward: {total_reward}, Total Loss: {total_loss}"
        # )

        if episode % params["target_model_sync"] == 0:
            print("New tgt_model loaded")
            target_model.load_state_dict(model.state_dict())
            window_size = params["target_model_sync"]
            mean_reward = np.mean(rewards[-window_size:])
            variance = np.var(rewards[-window_size:])
            variances.append(variance)
            print(
                f"Epizod {episode + 1}: Śr. nagroda (ostatnie {window_size}): {mean_reward}, Wariancja: {variance}"
            )
            writer.add_scalar("Stats/Mean Reward", mean_reward, episode)
            writer.add_scalar("Stats/Variance", variance, episode)

    save_model(
        model,
        experiment_path + "best_vdn_model.pth",
    )

    close_environment(env)

    # Wyniki
    training_time = time.time() - start_time
    mean_reward = np.mean(rewards)
    variance = np.var(rewards)
    print(f"Czas treningu: {training_time} sekund")
    print(f"Średnia nagroda: {mean_reward}")
    print(f"Wariancja nagród: {variance}")

    plot_rewards(
        rewards,
        save_path=experiment_path + "vdn_training_rewards.png",
        title="VDN Training Rewards",
    )

    plot_rewards(
        losses,
        save_path=experiment_path + "vdn_training_losses.png",
        title="Training Losses",
        ylabel="Total Loss",
        label="Total Loss",
    )

    plot_rewards(
        variances,
        save_path=experiment_path + "vdn_training_variance.png",
        title="Training Variance",
        ylabel="Variance",
        label="Variance",
        x_range=(0, params["num_episodes"]),
    )

    writer.close()


if __name__ == "__main__":
    train_vdn(params)
