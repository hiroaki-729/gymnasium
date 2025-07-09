"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

# Mujoco
# ----------------------------------------


def _raise_mujoco_py_error(*args: Any, **kwargs: Any):
    raise ImportError(
        "The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics)."
    )


# manipulation


register(id="Reacher-v2", entry_point=_raise_mujoco_py_error)

register(
    id="Hit_env",
    entry_point="gymnasium.envs.mujoco.hit_env:HitEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v4",
    entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v5",
    entry_point="gymnasium.envs.mujoco.reacher_v5:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

# --- For shimmy compatibility
def _raise_shimmy_error(*args: Any, **kwargs: Any):
    raise ImportError(
        'To use the gym compatibility environments, run `pip install "shimmy[gym-v21]"` or `pip install "shimmy[gym-v26]"`'
    )


# When installed, shimmy will re-register these environments with the correct entry_point
register(id="GymV21Environment-v0", entry_point=_raise_shimmy_error)
register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)
