"""Configuration definitions for WASABI runners and algorithms."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class WasabiCfg:
    """Configuration for the WASABI discriminator and reward."""

    policy_state_key: str = MISSING
    """The observation key containing the policy state."""

    reference_state_key: str = MISSING
    """The observation key containing the reference state."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the WASABI discriminator."""

    activation: str = MISSING
    """The activation function used by the discriminator."""

    normalize_input: bool = MISSING
    """Whether to normalize discriminator inputs."""

    normalization_until: int | None = MISSING
    """The training step until which discriminator inputs are normalized."""

    reward_type: Literal["log", "quad", "wasserstein"] = MISSING
    """The type of reward produced by the discriminator."""

    reward_coef: float = MISSING
    """The coefficient of the WASABI reward."""

    task_reward_weight: float = MISSING
    """The weight of the task reward."""

    loss_type: Literal["BCEWithLogitsLoss", "MSELoss", "WassersteinLoss"] = MISSING
    """The loss used to train the discriminator."""

    loss_coef: float = MISSING
    """The coefficient of the discriminator loss."""

    gradient_penalty_coef: float = MISSING
    """The coefficient of the discriminator gradient penalty."""

    gradient_tolerance: float = MISSING
    """The target gradient norm tolerance for the gradient penalty."""

    weight_decay_coef: float = MISSING
    """The weight decay coefficient for the discriminator backbone."""

    logit_weight_decay_coef: float = MISSING
    """The weight decay coefficient for discriminator logits."""

    discriminator_backbone_gradient_only: bool = MISSING
    """Whether to backpropagate discriminator gradients through the backbone only."""

    discriminator_optimizer: str = MISSING
    """The optimizer used for the discriminator."""

    learning_rate: float | None = MISSING
    """The discriminator learning rate override."""


@configclass
class RslRlWasabiAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for a WASABI algorithm based on PPO."""

    class_name: str = "WasabiPPO"
    """The WASABI algorithm class name."""

    wasabi_cfg: WasabiCfg = MISSING
    """The WASABI-specific algorithm configuration."""


@configclass
class RslRlWasabiRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for an on-policy runner using WASABI."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The actor-critic policy configuration."""

    algorithm: RslRlWasabiAlgorithmCfg = MISSING
    """The WASABI algorithm configuration."""
