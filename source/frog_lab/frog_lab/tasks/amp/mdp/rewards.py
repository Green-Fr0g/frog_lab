"""AMP-specific reward terms compatible with IsaacLab's reward manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Reward tracking commanded planar linear velocity at an anchor body."""
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    anchor_lin_vel_w = asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]
    command_lin_vel_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    command_lin_vel_w = quat_apply(yaw_quat(anchor_quat_w), command_lin_vel_b)
    lin_vel_error = torch.sum(torch.square(command_lin_vel_w - anchor_lin_vel_w), dim=-1)
    return torch.exp(-lin_vel_error / std**2)


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Reward tracking commanded yaw rate and suppressing roll/pitch rates."""
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    anchor_ang_vel_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
    ang_vel_z_error = torch.square(command[:, 2] - anchor_ang_vel_w[:, 2])
    anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)
    return torch.exp(-(ang_vel_z_error + ang_vel_xy_error) / std**2)


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Penalize squared body-frame angular velocity around the x and y axes."""
    asset: Articulation = env.scene[body_cfg.name]
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
    return torch.sum(torch.square(body_ang_vel_b[..., :2]), dim=-1).sum(dim=-1)
