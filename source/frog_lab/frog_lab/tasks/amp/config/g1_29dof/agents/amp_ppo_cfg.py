from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from frog_lab.rl_cfg.amp_cfg import AmpCfg, RslRlAmpAlgorithmCfg, RslRlAmpRunnerCfg
from frog_lab.tasks.amp.config.g1_29dof.flat_env_cfg import G1_29DOFAmpFlatEnvCfg


@configclass
class G1_29DOFAmpAlgorithmCfg(RslRlAmpAlgorithmCfg):
    pass


@configclass
class G1_29DOFAmpFlatRunnerCfg(RslRlAmpRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "g1_29dof_amp_flat"

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm: G1_29DOFAmpAlgorithmCfg = G1_29DOFAmpAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=AmpCfg(
            amp_reward_coef=0.1,
            amp_replay_buffer_size=2000000,
            amp_discr_hidden_dims=[1024, 512],
            amp_discr_activation="relu",
            discriminator_lr=1.0e-3,
            discriminator_optimizer="adam",
            amp_trunk_weight_decay=1.0e-3,
            amp_head_weight_decay=1.0e-2,
            grad_pen_coef=10.0,
            amp_task_reward_lerp=0.75,
            expert_state_key="amp_state",
            motion_loader_class_name="frog_lab.tasks.amp.utils.motion_loader:AMPBodyStateMotionLoader",
            motion_loader_kwargs={
                "motion_files": G1_29DOFAmpFlatEnvCfg().motion_dir,
                "body_names": tuple(G1_29DOFAmpFlatEnvCfg().amp_body_names),
                "anchor_name": G1_29DOFAmpFlatEnvCfg().anchor_body_name,
                "all_body_names": tuple(G1_29DOFAmpFlatEnvCfg().amp_all_body_names),
                "quat_order": "wxyz",
            },
        ),
    )
