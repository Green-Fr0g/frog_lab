from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_views(env: ManagerBasedRLEnv, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg):
    asset = env.scene[anchor_cfg.name]
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    return asset, anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w


def robot_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return body positions in the anchor frame."""
    _, anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w = _get_body_views(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return body orientations in the anchor frame as the first two matrix columns."""
    _, anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w = _get_body_views(env, anchor_cfg, body_cfg)
    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return body linear velocities in each body frame."""
    asset = env.scene[anchor_cfg.name]
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    num_bodies = body_lin_vel_w.shape[1]
    body_lin_vel_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_lin_vel_w.reshape(-1, 3)).reshape(
        env.num_envs, num_bodies, 3
    )
    return body_lin_vel_b.reshape(env.num_envs, num_bodies * 3)


def robot_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Return body angular velocities in each body frame."""
    asset = env.scene[anchor_cfg.name]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    num_bodies = body_ang_vel_w.shape[1]
    body_ang_vel_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_ang_vel_w.reshape(-1, 3)).reshape(
        env.num_envs, num_bodies, 3
    )
    return body_ang_vel_b.reshape(env.num_envs, num_bodies * 3)
