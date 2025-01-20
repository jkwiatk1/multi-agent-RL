import os
import torch
import random
from collections import deque
from src.models.DQN import create_dqn
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
    "model_save_path": "../results/dqn_mpe_model/",
    "render": False,
}


def train(params):
    env = create_environment(render=params["render"])
    model = create_dqn(params["state_dim"], params["action_dim"]).to(device)
    target_model = create_dqn(params["state_dim"], params["action_dim"]).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    replay_buffer = deque(maxlen=10000)
    best_reward = -float("inf")
    rewards = []
    losses = []
    epsilon = params["epsilon"]

    for episode in range(params["num_episodes"]):
        env.reset()
        total_reward = 0

        epsilon = max(params["epsilon_min"], epsilon * params["epsilon_decay"])

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            action = (
                None
                if termination or truncation
                else select_action(observation, model, epsilon)
            )
            env.step(action)
            total_reward += reward

            if action is not None:
                replay_buffer.append(
                    (
                        observation,
                        action,
                        reward,
                        env.last()[0],
                        termination or truncation,
                    )
                )

            if len(replay_buffer) >= params["batch_size"]:
                batch = random.sample(replay_buffer, params["batch_size"])
                loss = train_step(batch, model, target_model, optimizer, params["gamma"])
                losses.append(loss)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        rewards.append(total_reward)  # Zapis nagrody dla każdego epizodu

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        best_reward = save_model(
            model,
            params["model_save_path"] + "best_dqn_model.pth",
            total_reward,
            best_reward,
        )

    close_environment(env)

    plot_rewards(
        rewards,
        save_path=params["model_save_path"] + "dqn_training_rewards.png",
        title="Training Rewards",
    )

    plot_rewards(losses, save_path=params["model_save_path"] + "dqn_training_losses.png", title="Training Losses")


if __name__ == "__main__":
    train(params)
