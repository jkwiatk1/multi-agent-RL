from pettingzoo.mpe import simple_spread_v3
from supersuit import (
    pad_action_space_v0,
    pad_observations_v0,
    black_death_v3,
    pettingzoo_env_to_vec_env_v1,
    flatten_v0,
)
from pettingzoo.utils.conversions import aec_to_parallel

def create_mpe_env(seed=42, max_cycles=25):
    """
    Tworzy środowisko MPE z odpowiednimi wrapperami do treningu DQN.
    """
    # Tworzenie środowiska
    env = simple_spread_v3.env(max_cycles=max_cycles)
    env.reset(seed=seed)

    # Konwersja z AECEnv na ParallelEnv
    parallel_env = aec_to_parallel(env)

    # Wrappery SuperSuit
    parallel_env = pad_observations_v0(parallel_env)
    parallel_env = pad_action_space_v0(parallel_env)
    parallel_env = black_death_v3(parallel_env)
    parallel_env = flatten_v0(parallel_env)

    # Konwersja do środowiska zgodnego z Gym
    vec_env = pettingzoo_env_to_vec_env_v1(parallel_env)

    # Dodanie ręcznego przetwarzania
    return GymCompatibleWrapper(vec_env)

class GymCompatibleWrapper:
    """
    Wrapper dla środowiska, aby przekształcić obserwacje do formatu kompatybilnego z Gym.
    """
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self._flatten_observations(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_observations(obs), reward, done, info

    def _flatten_observations(self, obs):
        # Spłaszczenie obserwacji dla wszystkich agentów
        if isinstance(obs, dict):
            return [o.flatten() for o in obs.values()]
        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)



















# from pettingzoo.mpe import simple_spread_v3
# from supersuit import (
#     pad_action_space_v0,
#     pad_observations_v0,
#     black_death_v3,
#     pettingzoo_env_to_vec_env_v1,
#     flatten_v0,
# )
# from pettingzoo.utils.conversions import aec_to_parallel
#
# def create_mpe_env(seed=42, max_cycles=25):
#     """
#     Tworzy środowisko MPE z odpowiednimi wrapperami do treningu DQN.
#     """
#     # Tworzenie środowiska
#     env = simple_spread_v3.env(max_cycles=max_cycles)
#     print(f"Initial observation space: {env.observation_space}")
#
#     # Ustawienie ziarna losowości na poziomie AECEnv
#     env.reset(seed=seed)
#
#     # Konwersja z AECEnv na ParallelEnv
#     parallel_env = aec_to_parallel(env)
#     print(f"After conversion to ParallelEnv, observation space: {parallel_env.observation_space}")
#
#     # Wrappery SuperSuit
#     parallel_env = pad_observations_v0(parallel_env)  # Uzupełnij brakujące obserwacje
#     print(f"After pad_observations_v0, observation space: {parallel_env.observation_space}")
#
#     parallel_env = pad_action_space_v0(parallel_env)  # Uzupełnij brakujące akcje
#     print(f"After pad_action_space_v0, action space: {parallel_env.action_space}")
#
#     parallel_env = black_death_v3(parallel_env)       # Obsłuż agentów, którzy "umierają" w trakcie gry
#     parallel_env = flatten_v0(parallel_env)           # Spłaszczenie obserwacji do wektora 1D
#     print(f"After flatten_v0, observation space: {parallel_env.observation_space}")
#
#     # Konwersja do środowiska zgodnego z Gym
#     vec_env = pettingzoo_env_to_vec_env_v1(parallel_env)
#     print(f"Final vectorized environment observation space: {vec_env.observation_space}")
#
#     return vec_env
