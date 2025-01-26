import numpy as np
import torch

from src.environments.mpe import create_environment, close_environment
from src.models.VDN import create_vdn
from src.utils import select_action, plot_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_RESULT_PATH = (
    "../results/vdn_mpe_model/0.95_gamma/25000_epochs/0.0001_lr/0.9996_eps"
)


def evaluate(model, env, num_episodes=10, num_agents=3):
    """
    Args:
        model: Model VDN.
        env: Środowisko wieloagentowe.
        num_episodes: Liczba epizodów ewaluacji.
        num_agents: Liczba agentów w środowisku.
    """
    total_rewards = []

    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0

        while env.agents:
            actions = {
                agent: select_action(observations[agent], model.agents[i], epsilon=0.0)
                for i, agent in enumerate(env.agents)
            }

            next_observations, rewards, terminations, truncations, infos = env.step(
                actions
            )
            episode_reward += sum(rewards.values())
            observations = next_observations
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
    num_agents = 3
    best_model_path = MODEL_RESULT_PATH + "/best_vdn_model.pth"

    model = create_vdn(state_dim, action_dim, num_agents=num_agents).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded best model from:", best_model_path)

    env = create_environment(render=True, api="parallel")
    evaluate(model, env, num_episodes=10, num_agents=num_agents)
    close_environment(env)
