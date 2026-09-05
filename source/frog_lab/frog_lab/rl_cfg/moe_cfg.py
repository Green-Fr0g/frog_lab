"""Configuration definitions for PPO with a mixture-of-experts policy."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for a PPO actor-critic with mixture-of-experts heads."""

    class_name: str = "ActorCriticMoE"
    """The policy class name."""

    actor_num_experts: int = MISSING
    """The number of expert networks in the actor mixture-of-experts head."""

    critic_num_experts: int = MISSING
    """The number of expert networks in the critic mixture-of-experts head."""

    actor_gate_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor mixture-of-experts gating network."""

    critic_gate_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic mixture-of-experts gating network."""


@configclass
class RslRlMoERunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for an on-policy runner using a mixture-of-experts policy."""

    policy: RslRlMoEActorCriticCfg = MISSING
    """The mixture-of-experts actor-critic policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The PPO algorithm configuration."""
