"""WASABI observation terms.

These terms mirror the five state terms used by InstinctLab.  The reference
terms read the current frame owned by :class:`WasabiMotionReference`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from frog_lab.tasks.amp.utils.wasabi_motion_reference import WasabiMotionReference

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================== WASABI OBSERVATIONS ==============================


def _reference(env: ManagerBasedRLEnv) -> WasabiMotionReference:
    return WasabiMotionReference.for_env(env)


def _joint_ids(asset_cfg: SceneEntityCfg, count: int, device: torch.device) -> torch.Tensor:
    if getattr(asset_cfg, "joint_ids", None) is None or len(asset_cfg.joint_ids) == 0:
        return torch.arange(count, device=device)
    return torch.as_tensor(asset_cfg.joint_ids, device=device, dtype=torch.long)


def projected_gravity_reference_as_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    del asset_cfg
    return _reference(env).projected_gravity_b()


def joint_pos_rel_reference_as_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reference = _reference(env)
    robot = env.scene[robot_cfg.name]
    ids = _joint_ids(asset_cfg, reference.joint_pos.shape[-1], reference.device)
    return reference.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


def joint_vel_rel_reference_as_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reference = _reference(env)
    robot = env.scene[robot_cfg.name]
    ids = _joint_ids(asset_cfg, reference.joint_vel.shape[-1], reference.device)
    return reference.joint_vel[:, ids] - robot.data.default_joint_vel[:, ids]


def base_lin_vel_reference_as_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    del asset_cfg
    return _reference(env).base_lin_vel_b()


def base_ang_vel_reference_as_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    del asset_cfg
    return _reference(env).base_ang_vel_b()


def projected_gravity_wasabi_policy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.projected_gravity_b


def base_lin_vel_wasabi_policy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)


def base_ang_vel_wasabi_policy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)
