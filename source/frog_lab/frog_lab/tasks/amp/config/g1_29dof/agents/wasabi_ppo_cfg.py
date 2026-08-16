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
            policy_state_key="wasabi_policy",
            reference_state_key="wasabi_reference",
            hidden_dims=[512, 256],
            activation="elu",
            normalize_input=True,
            normalization_until=int(1e8),
            reward_type="log",
            reward_coef=1.0,
            task_reward_weight=1.0,
            loss_type="BCEWithLogitsLoss",
            loss_coef=1.0,
            gradient_penalty_coef=10.0,
            gradient_tolerance=0.0,
            weight_decay_coef=0.0,
            logit_weight_decay_coef=0.0,
            discriminator_backbone_gradient_only=False,
            discriminator_optimizer="adamw",
            learning_rate=1.0e-3,
        ),
    )
