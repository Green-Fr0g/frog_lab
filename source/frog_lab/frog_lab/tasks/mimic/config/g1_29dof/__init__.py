import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="FrogLab-Isaac-Mimic-Flat-Unitree-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29DOFFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFFlatPPORunnerCfg",
    },
)

gym.register(
    id="FrogLab-Isaac-Mimic-Flat-Unitree-G1-29DOF-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29DOFFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFFlatPPORunnerCfg",
    },
)


gym.register(
    id="FrogLab-Isaac-Mimic-Flat-Unitree-G1-29DOF-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29DOFFlatLowFreqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFFlatLowFreqPPORunnerCfg",
    },
)
