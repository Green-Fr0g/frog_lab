from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_str_list(values) -> list[str]:
    return [str(name) for name in np.asarray(values).tolist()]


def _index_by_names(available: Sequence[str], requested: Sequence[str], kind: str) -> list[int]:
    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(
            f"AMP motion {kind} names {missing} were not found. Available {kind} names: {list(available)}"
        )
    return [lookup[name] for name in requested]


@dataclass
class _MotionClip:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_names: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]


class MotionResetManager:
    """Caches AMP motion frames and resets environments from sampled frames."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self._frames: dict[str, list[_MotionClip]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(
        self,
        motion_dir: str,
        device: str | torch.device,
        root_name: str,
        all_body_names: tuple[str, ...],
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        if root_name not in all_body_names:
            raise ValueError(f"AMP root body '{root_name}' is not in all_body_names.")
        root_index = all_body_names.index(root_name)
        cache_key = f"{motion_dir}:{root_index}:{tuple(all_body_names)}"
        if cache_key in self._frames:
            return

        files = self._collect_motion_files(motion_dir)
        if not files:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {motion_dir}")
        clips: list[_MotionClip] = []
        for file in files:
            data = np.load(file)
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
                if key not in data:
                    raise KeyError(f"AMP motion file '{file}' is missing key '{key}'.")

            body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            body_lin_vel_w = torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            body_ang_vel_w = torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
            joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)

            if "body_names" not in data:
                raise KeyError(
                    f"AMP motion file '{file}' is missing 'body_names'. "
                    "Re-export the motion with scripts/mimic/csv_to_npz.py."
                )
            if "joint_names" not in data:
                raise KeyError(
                    f"AMP motion file '{file}' is missing 'joint_names'. "
                    "Re-export the motion with scripts/mimic/csv_to_npz.py."
                )

            motion_body_names = _as_str_list(data["body_names"])
            if len(motion_body_names) != body_pos_w.shape[1]:
                raise ValueError(
                    f"AMP motion file '{file}' has {len(motion_body_names)} body_names "
                    f"but body_pos_w has {body_pos_w.shape[1]} bodies."
                )
            body_indexes = _index_by_names(motion_body_names, all_body_names, "body")
            body_pos_w = body_pos_w[:, body_indexes]
            body_quat_w = body_quat_w[:, body_indexes]
            body_lin_vel_w = body_lin_vel_w[:, body_indexes]
            body_ang_vel_w = body_ang_vel_w[:, body_indexes]

            joint_names = _as_str_list(data["joint_names"])
            if len(joint_names) != joint_pos.shape[1]:
                raise ValueError(
                    f"AMP motion file '{file}' has {len(joint_names)} joint_names "
                    f"but joint_pos has {joint_pos.shape[1]} joints."
                )

            clips.append(
                _MotionClip(
                    root_pos=body_pos_w[:, root_index, :],
                    root_quat=body_quat_w[:, root_index, :],
                    root_lin_vel=body_lin_vel_w[:, root_index, :],
                    root_ang_vel=body_ang_vel_w[:, root_index, :],
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    joint_names=tuple(joint_names),
                )
            )

        self._frames[cache_key] = clips

    def reset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        root_name: str,
        all_body_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        cache_key = f"{motion_dir}:{all_body_names.index(root_name)}:{tuple(all_body_names)}"
        if cache_key not in self._frames:
            self.init(motion_dir, env.device, root_name, all_body_names)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        clips = self._frames[cache_key]
        clip_ids = torch.randint(0, len(clips), (len(env_ids),), device=env.device)
        frame_ids = torch.zeros_like(clip_ids)
        for clip_id in clip_ids.unique().tolist():
            selected = clip_ids == clip_id
            frame_ids[selected] = torch.randint(clips[clip_id].num_frames, (int(selected.sum().item()),), device=env.device)

        asset: Articulation = env.scene[asset_cfg.name]
        asset_joint_names = tuple(getattr(asset.data, "joint_names", ()))
        if not asset_joint_names:
            raise RuntimeError("AMP asset does not expose joint_names for motion alignment.")

        root_pos = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        root_quat = torch.empty((len(env_ids), 4), device=env.device, dtype=torch.float32)
        root_lin_vel = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        root_ang_vel = torch.empty((len(env_ids), 3), device=env.device, dtype=torch.float32)
        joint_pos = torch.empty((len(env_ids), len(asset_joint_names)), device=env.device, dtype=torch.float32)
        joint_vel = torch.empty_like(joint_pos)

        for clip_id in clip_ids.unique().tolist():
            selected = clip_ids == clip_id
            clip = clips[clip_id]
            clip_frame_ids = frame_ids[selected]
            root_pos[selected] = clip.root_pos[clip_frame_ids]
            root_quat[selected] = clip.root_quat[clip_frame_ids]
            root_lin_vel[selected] = clip.root_lin_vel[clip_frame_ids]
            root_ang_vel[selected] = clip.root_ang_vel[clip_frame_ids]

            clip_joint_pos = clip.joint_pos[clip_frame_ids]
            clip_joint_vel = clip.joint_vel[clip_frame_ids]
            joint_indexes = _index_by_names(clip.joint_names, asset_joint_names, "joint")
            joint_pos[selected] = clip_joint_pos[:, joint_indexes]
            joint_vel[selected] = clip_joint_vel[:, joint_indexes]

        root_pos = root_pos.clone()
        root_pos[:, :2] += env.scene.env_origins[env_ids, :2]
        root_pos[:, 2] += env.scene.env_origins[env_ids, 2]
        root_pose = torch.cat((root_pos, root_quat), dim=-1)
        root_velocity = torch.cat((root_lin_vel, root_ang_vel), dim=-1)

        joint_pos = joint_pos[:, asset_cfg.joint_ids]
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
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
    root_name: str,
    all_body_names: tuple[str, ...],
) -> None:
    """Load AMP motion data during startup."""
    del env_ids
    MotionResetManager.get().init(motion_dir, env.device, root_name, all_body_names)


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
) -> None:
    """Reset selected environments from random AMP motion frames."""
    MotionResetManager.get().reset(env, env_ids, motion_dir, root_name, all_body_names, asset_cfg)
