"""Unitree G1 29-DOF AMP locomotion."""

import gymnasium as gym

from . import agents
from .flat_env_cfg import G1_29DOFAmpFlatEnvCfg
from .rough_env_cfg import G1_29DOFAmpRoughEnvCfg

gym.register(
    id="FrogLab-Isaac-AMP-Rough-Unitree-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1_29DOFAmpRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFAmpRoughRunnerCfg",
    },
)

gym.register(
    id="FrogLab-Isaac-AMP-Flat-Unitree-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1_29DOFAmpFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFAmpFlatRunnerCfg",
    },
)
