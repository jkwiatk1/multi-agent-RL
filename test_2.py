from pettingzoo.mpe import simple_spread_v3

# Tworzymy środowisko
env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)

# Teraz możemy uzyskać wymiary przestrzeni stanów dla agentów
# Przechodzimy po wszystkich agentach po resetowaniu środowiska
state_dims = [env.observation_space(agent).shape[0] for agent in env.agents]

# Sprawdzamy rozmiary przestrzeni stanów
print("State dimensions:", state_dims)
