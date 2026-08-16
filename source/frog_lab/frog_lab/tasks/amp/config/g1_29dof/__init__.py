"""Unitree G1 29-DOF AMP flat locomotion."""

import gymnasium as gym

from . import agents

gym.register(
    id="FrogLab-Isaac-AMP-Flat-Unitree-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29DOFAmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_ppo_cfg:G1_29DOFAmpFlatRunnerCfg",
    },
)

gym.register(
    id="FrogLab-Isaac-WASABI-Flat-Unitree-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wasabi_flat_env_cfg:G1_29DOFWasabiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.wasabi_ppo_cfg:G1_29DOFWasabiRunnerCfg",
    },
)
