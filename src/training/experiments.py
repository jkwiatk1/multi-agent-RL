import torch

from src.training.train_dqn import train_dqn
from src.training.train_qatten import train_qatten
from src.training.train_vdn import train_vdn

params_vdn_25k_9 = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.9,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_episodes": 25000,
    "target_model_sync": 70,
    "model_save_path": "../results/vdn_mpe_model/0.9_gamma/",
    "render": False,
}

params_vdn_25k_95 = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.95,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_episodes": 25000,
    "target_model_sync": 70,
    "model_save_path": "../results/vdn_mpe_model/0.95_gamma/",
    "render": False,
}

params_dqn_25k_9 = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.9,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_episodes": 25000,
    "target_model_sync": 70,
    "model_save_path": "../results/dqn_mpe_model/0.9_gamma/",
    "render": False,
}

params_dqn_25k_95 = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.95,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_episodes": 25000,
    "target_model_sync": 70,
    "model_save_path": "../results/dqn_mpe_model/0.95_gamma/",
    "render": False,
}

params_qatten_25k_9 = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.9,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_episodes": 25000,
    "target_model_sync": 70,
    "model_save_path": "../results/qatten_mpe_model/0.9_gamma/",
    "render": False,
}

params_qatten_25k_95 = {
    "state_dim": 18,
    "action_dim": 5,
    "gamma": 0.95,
    "epsilon": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_episodes": 25000,
    "target_model_sync": 70,
    "model_save_path": "../results/qatten_mpe_model/0.95_gamma/",
    "render": False,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_vdn(params_vdn_25k_95, device)
    train_vdn(params_vdn_25k_9, device)
    train_dqn(params_dqn_25k_95, device)
    train_dqn(params_dqn_25k_9, device)
    train_qatten(params_qatten_25k_95, device)
    train_qatten(params_qatten_25k_9, device)
