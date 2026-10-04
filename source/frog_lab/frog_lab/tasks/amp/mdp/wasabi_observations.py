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
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return torch.arange(count, device=device)
    if isinstance(joint_ids, slice):
        return torch.arange(count, device=device)[joint_ids]
    if len(joint_ids) == 0:
        return torch.arange(count, device=device)
    return torch.as_tensor(joint_ids, device=device, dtype=torch.long)


def _zero_reference_vector(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)


def _zero_reference_joint_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    default_joint_pos = robot.data.default_joint_pos
    ids = _joint_ids(asset_cfg, default_joint_pos.shape[-1], default_joint_pos.device)
    return torch.zeros(
        (env.num_envs, ids.numel()),
        device=default_joint_pos.device,
        dtype=default_joint_pos.dtype,
    )


def projected_gravity_reference_as_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    del asset_cfg
    if not WasabiMotionReference.is_initialized(env):
        return _zero_reference_vector(env)
    return _reference(env).projected_gravity_b()


def joint_pos_rel_reference_as_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not WasabiMotionReference.is_initialized(env):
        return _zero_reference_joint_state(env, asset_cfg, robot_cfg)
    reference = _reference(env)
    robot = env.scene[robot_cfg.name]
    reference_ids, asset_ids = reference.resolve_joint_mapping(robot, asset_cfg)
    return reference.joint_pos[:, reference_ids] - robot.data.default_joint_pos[:, asset_ids]


def joint_vel_rel_reference_as_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not WasabiMotionReference.is_initialized(env):
        return _zero_reference_joint_state(env, asset_cfg, robot_cfg)
    reference = _reference(env)
    robot = env.scene[robot_cfg.name]
    reference_ids, asset_ids = reference.resolve_joint_mapping(robot, asset_cfg)
    return reference.joint_vel[:, reference_ids] - robot.data.default_joint_vel[:, asset_ids]


def base_lin_vel_reference_as_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    del asset_cfg
    if not WasabiMotionReference.is_initialized(env):
        return _zero_reference_vector(env)
    return _reference(env).base_lin_vel_b()


def base_ang_vel_reference_as_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    del asset_cfg
    if not WasabiMotionReference.is_initialized(env):
        return _zero_reference_vector(env)
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
