"""Shared helpers and validation utilities for robot asset configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def pd_gains(
    armature: float,
    natural_frequency: float = 10.0,
    damping_ratio: float = 2.0,
) -> tuple[float, float]:
    """Return implicit-PD stiffness and damping for an armature value."""
    omega = 2.0 * math.pi * natural_frequency
    return armature * omega**2, 2.0 * damping_ratio * armature * omega


def validate_joint_groups(
    all_joint_names: Sequence[str],
    control_joint_names: Sequence[str],
    amp_joint_names: Sequence[str],
) -> None:
    """Validate the required ALL/CONTROL/AMP joint-set relationship."""
    all_names = tuple(all_joint_names)
    control_names = tuple(control_joint_names)
    amp_names = tuple(amp_joint_names)
    if len(set(all_names)) != len(all_names):
        raise ValueError("all_joint_names contains duplicate names")
    if len(set(control_names)) != len(control_names):
        raise ValueError("control_joint_names contains duplicate names")
    if len(set(amp_names)) != len(amp_names):
        raise ValueError("amp_joint_names contains duplicate names")
    missing_control = [name for name in control_names if name not in all_names]
    missing_amp = [name for name in amp_names if name not in control_names]
    if missing_control:
        raise ValueError(f"control joints are not in all joints: {missing_control}")
    if missing_amp:
        raise ValueError(f"AMP joints are not in control joints: {missing_amp}")


def validate_body_names(body_names: Sequence[str], root_link_name: str) -> None:
    """Validate body-name uniqueness and root-link membership."""
    names = tuple(body_names)
    if len(set(names)) != len(names):
        raise ValueError("body_names contains duplicate names")
    if root_link_name not in names:
        raise ValueError(f"root link '{root_link_name}' is not in body_names")


def action_scale_from_limits(
    joint_names: Sequence[str],
    effort_limits: Mapping[str, float],
    stiffness: Mapping[str, float],
    factor: float = 0.25,
) -> dict[str, float]:
    """Build per-joint position-action scales from explicit actuator values."""
    missing = [name for name in joint_names if name not in effort_limits or name not in stiffness]
    if missing:
        raise ValueError(f"Missing effort or stiffness values for joints: {missing}")
    return {
        name: factor * effort_limits[name] / stiffness[name]
        for name in joint_names
        if stiffness[name]
    }
