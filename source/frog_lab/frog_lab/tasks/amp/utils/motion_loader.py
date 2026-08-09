"""G1 AMP motion loader.

This loader follows the AMP_mjlab G1 discriminator contract: each expert sample is a
transition ``(state, next_state)`` where each state is built from body-local position,
orientation, linear velocity, and angular velocity.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch


class G1AMPBodyStateMotionLoader:
    """Load G1 NPZ motions and sample AMP body-state transitions."""

    REQUIRED_KEYS = (
        "fps",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(
        self,
        *,
        device: str,
        time_between_frames: float,
        motion_files: str | Sequence[str] | None = None,
        motion_file: str | None = None,
        body_names: Sequence[str] | None = None,
        anchor_name: str | None = None,
        all_body_names: Sequence[str] | None = None,
        body_indices: Sequence[int] | None = None,
        anchor_index: int | None = None,
        quat_order: str = "wxyz",
    ) -> None:
        """Initialize the motion loader.

        Args:
            device: Torch device.
            time_between_frames: Algorithm transition duration. The loader samples
                adjacent NPZ frames for now; this value is kept for the algorithm
                contract and future interpolation support.
            motion_files: A NPZ file, directory, or list of NPZ files.
            motion_file: Backward-compatible alias for a single NPZ file or directory.
            body_names: Body names used by AMP discriminator.
            anchor_name: Anchor body name, usually pelvis/torso/root.
            all_body_names: Ordered body names matching the NPZ body axis.
            body_indices: Direct body indices, used when names are not available.
            anchor_index: Direct anchor body index.
            quat_order: Quaternion order stored in NPZ, either ``"wxyz"`` or ``"xyzw"``.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.quat_order = quat_order

        paths = self._resolve_motion_files(motion_files if motion_files is not None else motion_file)
        self._body_indices, self._anchor_index = self._resolve_body_indices(
            body_names=body_names,
            anchor_name=anchor_name,
            all_body_names=all_body_names,
            body_indices=body_indices,
            anchor_index=anchor_index,
        )

        self.motion_names: list[str] = []
        self._body_pos_b_list: list[torch.Tensor] = []
        self._body_ori_b_list: list[torch.Tensor] = []
        self._body_lin_vel_b_list: list[torch.Tensor] = []
        self._body_ang_vel_b_list: list[torch.Tensor] = []
        self._fps_list: list[float] = []

        for path in paths:
            self._load_motion(path)

        if not self._body_pos_b_list:
            raise ValueError("No AMP motion data loaded.")

        self.fps = self._fps_list[0]
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / float(self.fps)

    @property
    def observation_dim(self) -> int:
        """Return the dimension of one AMP state."""
        return (3 + 6 + 3 + 3) * len(self._body_indices)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield expert AMP transitions ``(state, next_state)``."""
        num_motions = len(self._body_pos_b_list)
        for batch_idx in range(num_mini_batch):
            motion_idx = batch_idx % num_motions
            body_pos_b = self._body_pos_b_list[motion_idx]
            body_ori_b = self._body_ori_b_list[motion_idx]
            body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]

            num_frames = body_pos_b.shape[0]
            if num_frames < 2:
                raise ValueError("AMP motion clips must contain at least two frames.")
            idxs = torch.randint(0, num_frames - 1, (mini_batch_size,), device=body_pos_b.device)
            next_idxs = idxs + 1

            yield (
                self._pack_state(body_pos_b, body_ori_b, body_lin_vel_b, body_ang_vel_b, idxs),
                self._pack_state(body_pos_b, body_ori_b, body_lin_vel_b, body_ang_vel_b, next_idxs),
            )

    def _load_motion(self, path: str) -> None:
        data = np.load(path)
        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"Motion file '{path}' is missing keys: {missing}")

        body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
        body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
        body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
        body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)
        body_quat_w = self._to_wxyz(body_quat_w)

        anchor_pos_w = body_pos_w[:, self._anchor_index]
        anchor_quat_w = body_quat_w[:, self._anchor_index]
        target_pos_w = body_pos_w[:, self._body_indices]
        target_quat_w = body_quat_w[:, self._body_indices]
        target_lin_vel_w = body_lin_vel_w[:, self._body_indices]
        target_ang_vel_w = body_ang_vel_w[:, self._body_indices]

        body_pos_b = self._quat_apply_inverse(
            anchor_quat_w[:, None, :].expand_as(target_quat_w),
            target_pos_w - anchor_pos_w[:, None, :],
        )
        body_quat_b = self._quat_multiply(
            self._quat_inverse(anchor_quat_w[:, None, :].expand_as(target_quat_w)),
            target_quat_w,
        )
        body_ori_b = self._matrix_first_two_columns(body_quat_b)
        body_lin_vel_b = self._quat_apply_inverse(target_quat_w, target_lin_vel_w)
        body_ang_vel_b = self._quat_apply_inverse(target_quat_w, target_ang_vel_w)

        self.motion_names.append(os.path.splitext(os.path.basename(path))[0])
        self._fps_list.append(float(data["fps"]))
        self._body_pos_b_list.append(body_pos_b)
        self._body_ori_b_list.append(body_ori_b)
        self._body_lin_vel_b_list.append(body_lin_vel_b)
        self._body_ang_vel_b_list.append(body_ang_vel_b)

    @staticmethod
    def _pack_state(
        body_pos_b: torch.Tensor,
        body_ori_b: torch.Tensor,
        body_lin_vel_b: torch.Tensor,
        body_ang_vel_b: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = indices.shape[0]
        return torch.cat(
            [
                body_pos_b[indices].reshape(batch_size, -1),
                body_ori_b[indices].reshape(batch_size, -1),
                body_lin_vel_b[indices].reshape(batch_size, -1),
                body_ang_vel_b[indices].reshape(batch_size, -1),
            ],
            dim=-1,
        )

    @staticmethod
    def _resolve_motion_files(path_or_paths: str | Sequence[str] | None) -> list[str]:
        if path_or_paths is None:
            raise ValueError("G1AMPBodyStateMotionLoader requires motion_files or motion_file.")

        if isinstance(path_or_paths, str):
            if os.path.isdir(path_or_paths):
                paths = [
                    os.path.join(root, filename)
                    for root, _, files in os.walk(path_or_paths)
                    for filename in sorted(files)
                    if filename.endswith(".npz")
                ]
            else:
                paths = [path_or_paths]
        else:
            paths = list(path_or_paths)

        paths = sorted(paths)
        if not paths:
            raise ValueError(f"No NPZ motion files found in: {path_or_paths}")
        invalid = [path for path in paths if not os.path.isfile(path)]
        if invalid:
            raise ValueError(f"Invalid motion files: {invalid}")
        return paths

    @staticmethod
    def _resolve_body_indices(
        *,
        body_names: Sequence[str] | None,
        anchor_name: str | None,
        all_body_names: Sequence[str] | None,
        body_indices: Sequence[int] | None,
        anchor_index: int | None,
    ) -> tuple[list[int], int]:
        if body_indices is not None and anchor_index is not None:
            return list(body_indices), int(anchor_index)

        if body_names is None or anchor_name is None or all_body_names is None:
            raise ValueError(
                "Provide either body_indices + anchor_index, or body_names + anchor_name + all_body_names."
            )

        all_names = list(all_body_names)
        return [all_names.index(name) for name in body_names], all_names.index(anchor_name)

    def _to_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        if self.quat_order == "wxyz":
            return quat
        if self.quat_order == "xyzw":
            return quat[..., [3, 0, 1, 2]]
        raise ValueError(f"Unsupported quat_order: {self.quat_order}")

    @staticmethod
    def _quat_inverse(quat: torch.Tensor) -> torch.Tensor:
        return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)

    @staticmethod
    def _quat_multiply(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        w0, x0, y0, z0 = q0.unbind(-1)
        w1, x1, y1, z1 = q1.unbind(-1)
        return torch.stack(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dim=-1,
        )

    @classmethod
    def _quat_apply_inverse(cls, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(vec[..., :1])
        vec_quat = torch.cat([zeros, vec], dim=-1)
        return cls._quat_multiply(cls._quat_multiply(cls._quat_inverse(quat), vec_quat), quat)[..., 1:]

    @staticmethod
    def _matrix_first_two_columns(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(-1)
        two_s = 2.0 / quat.square().sum(dim=-1).clamp_min(1e-12)

        m00 = 1.0 - two_s * (y * y + z * z)
        m01 = two_s * (x * y - z * w)
        m10 = two_s * (x * y + z * w)
        m11 = 1.0 - two_s * (x * x + z * z)
        m20 = two_s * (x * z - y * w)
        m21 = two_s * (y * z + x * w)
        return torch.stack([m00, m01, m10, m11, m20, m21], dim=-1)
