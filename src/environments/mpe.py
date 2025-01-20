from pettingzoo.mpe import simple_spread_v3


def create_environment(render=False, api="aec"):
    """
    Tworzy i resetuje środowisko.

    Args:
        render (bool): Czy włączyć renderowanie środowiska.

    Returns:
        env: Obiekt środowiska.
    """
    render_mode = "human" if render else None
    env = (
        simple_spread_v3.parallel_env(render_mode=render_mode)
        if api == "parallel"
        else simple_spread_v3.env(render_mode=render_mode)
    )
    env.reset(seed=42)
    # observations, infos = env.reset(seed=42)
    return env


def close_environment(env):
    """
    Zamyka środowisko.

    Args:
        env: Obiekt środowiska do zamknięcia.
    """
    env.close()
