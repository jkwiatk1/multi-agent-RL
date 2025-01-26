import numpy as np
import torch

from src.environments.mpe import create_environment, close_environment
from src.models.DQN import create_dqn
from src.utils import select_action, plot_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_RESULT_PATH = (
    "../results/dqn_mpe_model/0.95_gamma/10000_epochs/0.0001_lr/0.9996_eps"
)


def evaluate(model, env, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            action = (
                None
                if termination or truncation
                else select_action(observation, model, epsilon=0.0)
            )
            env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    plot_rewards(
        total_rewards,
        save_path=MODEL_RESULT_PATH + "/evaluation_rewards.png",
        title="Evaluation Rewards",
    )


if __name__ == "__main__":
    state_dim = 18
    action_dim = 5
    best_model_path = MODEL_RESULT_PATH + "/best_dqn_model.pth"

    model = create_dqn(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded best model from:", best_model_path)

    env = create_environment(render=True)
    evaluate(model, env)
    close_environment(env)
