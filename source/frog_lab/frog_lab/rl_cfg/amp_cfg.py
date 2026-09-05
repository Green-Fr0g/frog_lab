"""Configuration definitions for AMP runners and algorithms."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AmpCfg:
    """Configuration for the AMP discriminator and motion data."""

    amp_reward_coef: float = MISSING
    """The coefficient used to scale the AMP reward."""

    amp_replay_buffer_size: int = MISSING
    """The capacity of the AMP replay buffer."""

    amp_discr_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the AMP discriminator."""

    amp_discr_activation: str = MISSING
    """The activation function used by the AMP discriminator."""

    discriminator_lr: float = MISSING
    """The learning rate of the AMP discriminator."""

    discriminator_optimizer: str = MISSING
    """The optimizer used for the AMP discriminator."""

    amp_trunk_weight_decay: float = MISSING
    """The weight decay applied to the discriminator trunk."""

    amp_head_weight_decay: float = MISSING
    """The weight decay applied to the discriminator head."""

    grad_pen_coef: float = MISSING
    """The coefficient of the discriminator gradient penalty."""

    amp_task_reward_lerp: float = MISSING
    """The interpolation factor between task and AMP rewards."""

    amp_state_key: str = MISSING
    """The observation key used for the AMP expert state."""

    motion_loader_class_name: str = MISSING
    """The import path of the motion loader class."""

    motion_loader_kwargs: dict[str, Any] = MISSING
    """The keyword arguments passed to the motion loader."""


@configclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for an AMP algorithm based on PPO."""

    class_name: str = "AMPPPO"
    """The AMP algorithm class name."""

    amp_cfg: AmpCfg = MISSING
    """The AMP-specific algorithm configuration."""


@configclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for an on-policy runner using AMP."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The actor-critic policy configuration."""

    algorithm: RslRlAmpAlgorithmCfg = MISSING
    """The AMP algorithm configuration."""
