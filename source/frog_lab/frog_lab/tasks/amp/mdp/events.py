from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionResetManager:
    """Caches AMP motion frames and resets environments from sampled frames."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self._frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, motion_dir: str, device: str | torch.device) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        if motion_dir in self._frames:
            return

        files = self._collect_motion_files(motion_dir)
        if not files:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {motion_dir}")

        frame_lists: dict[str, list[torch.Tensor]] = {
            "root_pos": [],
            "root_quat": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "joint_pos": [],
            "joint_vel": [],
        }
        for file in files:
            data = np.load(file)
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
                if key not in data:
                    raise KeyError(f"AMP motion file '{file}' is missing key '{key}'.")

            frame_lists["root_pos"].append(torch.as_tensor(data["body_pos_w"][:, 0, :], device=device, dtype=torch.float32))
            frame_lists["root_quat"].append(
                torch.as_tensor(data["body_quat_w"][:, 0, :], device=device, dtype=torch.float32)
            )
            frame_lists["root_lin_vel"].append(
                torch.as_tensor(data["body_lin_vel_w"][:, 0, :], device=device, dtype=torch.float32)
            )
            frame_lists["root_ang_vel"].append(
                torch.as_tensor(data["body_ang_vel_w"][:, 0, :], device=device, dtype=torch.float32)
            )
            frame_lists["joint_pos"].append(torch.as_tensor(data["joint_pos"], device=device, dtype=torch.float32))
            frame_lists["joint_vel"].append(torch.as_tensor(data["joint_vel"], device=device, dtype=torch.float32))

        self._frames[motion_dir] = {key: torch.cat(value, dim=0) for key, value in frame_lists.items()}

    def reset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        if motion_dir not in self._frames:
            self.init(motion_dir, env.device)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        frames = self._frames[motion_dir]
        frame_ids = torch.randint(0, frames["root_pos"].shape[0], (len(env_ids),), device=env.device)
        asset: Articulation = env.scene[asset_cfg.name]

        root_pos = frames["root_pos"][frame_ids].clone()
        root_pos[:, :2] += env.scene.env_origins[env_ids, :2]
        root_pos[:, 2] += env.scene.env_origins[env_ids, 2]
        root_pose = torch.cat((root_pos, frames["root_quat"][frame_ids]), dim=-1)
        root_velocity = torch.cat((frames["root_lin_vel"][frame_ids], frames["root_ang_vel"][frame_ids]), dim=-1)

        joint_pos = frames["joint_pos"][frame_ids][:, asset_cfg.joint_ids]
        joint_vel = frames["joint_vel"][frame_ids][:, asset_cfg.joint_ids]
        joint_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos = joint_pos.clamp(joint_limits[..., 0], joint_limits[..., 1])

        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    @staticmethod
    def _collect_motion_files(motion_dir: str) -> list[Path]:
        path = Path(motion_dir)
        if path.is_file() and path.suffix == ".npz":
            return [path]
        return sorted(path.rglob("*.npz"))


def init_motion_loader(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
) -> None:
    """Load AMP motion data during startup."""
    del env_ids
    MotionResetManager.get().init(motion_dir, env.device)


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
) -> None:
    """Reset selected environments from random AMP motion frames."""
    MotionResetManager.get().reset(env, env_ids, motion_dir, asset_cfg)
