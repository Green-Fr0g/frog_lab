from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from frog_lab.rl_cfg.wasabi_cfg import RslRlWasabiAlgorithmCfg, RslRlWasabiRunnerCfg, WasabiCfg

@configclass
class G1_29DOFWasabiAlgorithmCfg(RslRlWasabiAlgorithmCfg):
    pass


@configclass
class G1_29DOFWasabiRunnerCfg(RslRlWasabiRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "g1_29dof_wasabi_flat"

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: G1_29DOFWasabiAlgorithmCfg = G1_29DOFWasabiAlgorithmCfg(
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
        wasabi_cfg=WasabiCfg(
            wasabi_policy_state_key="wasabi_policy",
            wasabi_reference_state_key="wasabi_reference",
            wasabi_discr_hidden_dims=[512, 256],
            wasabi_discr_activation="elu",
            wasabi_normalize_input=True,
            wasabi_normalization_until=int(1e8),
            wasabi_reward_type="log",
            wasabi_reward_coef=1.0,
            wasabi_task_reward_weight=1.0,
            wasabi_loss_type="BCEWithLogitsLoss",
            wasabi_loss_coef=1.0,
            wasabi_grad_pen_coef=10.0,
            wasabi_grad_tolerance=0.0,
            wasabi_trunk_weight_decay=0.0,
            wasabi_head_weight_decay=0.0,
            wasabi_discriminator_optimizer="adamw",
            wasabi_discriminator_lr=1.0e-3,
        ),
    )
