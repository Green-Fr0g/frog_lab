from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from frog_lab.tasks.amp.config.g1.common import (
    G1_29DOF_AMP_ALL_BODY_NAMES,
    G1_29DOF_AMP_ANCHOR_NAME,
    G1_29DOF_AMP_BODY_NAMES,
    G1_29DOF_AMP_MOTION_DIR,
)


@configclass
class G1_29DOFAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "AMPPPO"
    amp_cfg: dict = MISSING


def _make_actor_cfg() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )


def _make_critic_cfg() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )


def _make_algorithm_cfg() -> G1_29DOFAmpAlgorithmCfg:
    return G1_29DOFAmpAlgorithmCfg(
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
        amp_cfg={
            "amp_reward_coef": 1.0,
            "amp_replay_buffer_size": 1_000_000,
            "amp_discr_hidden_dims": [1024, 512],
            "discriminator_lr": 1.0e-3,
            "amp_trunk_weight_decay": 1.0e-3,
            "amp_head_weight_decay": 1.0e-2,
            "grad_pen_coef": 10.0,
            "amp_task_reward_lerp": 0.0,
            "time_between_frames": 0.02,
            "motion_loader_class_name": "frog_lab.tasks.amp.utils.motion_loader:G1AMPBodyStateMotionLoader",
            "motion_loader_kwargs": {
                "motion_files": str(G1_29DOF_AMP_MOTION_DIR),
                "body_names": G1_29DOF_AMP_BODY_NAMES,
                "anchor_name": G1_29DOF_AMP_ANCHOR_NAME,
                "all_body_names": G1_29DOF_AMP_ALL_BODY_NAMES,
                "quat_order": "wxyz",
            },
        },
    )


@configclass
class G1_29DOFAmpRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    actor: RslRlMLPModelCfg = _make_actor_cfg()
    critic: RslRlMLPModelCfg = _make_critic_cfg()
    algorithm: G1_29DOFAmpAlgorithmCfg = _make_algorithm_cfg()

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_29dof_amp"
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
    clip_actions = 100.0
    check_for_nan = True
    logger = "tensorboard"


@configclass
class G1_29DOFAmpRoughRunnerCfg(G1_29DOFAmpRunnerCfg):
    experiment_name = "g1_29dof_amp_rough"


@configclass
class G1_29DOFAmpFlatRunnerCfg(G1_29DOFAmpRunnerCfg):
    max_iterations = 1500
    experiment_name = "g1_29dof_amp_flat"
