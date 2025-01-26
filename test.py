from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(N=6, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)
env.close()