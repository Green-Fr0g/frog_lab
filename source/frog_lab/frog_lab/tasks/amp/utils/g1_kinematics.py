"""Offline kinematics utilities for G1 AMP motion conversion."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class URDFJoint:
    """Minimal URDF joint data needed for forward kinematics."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


class G1URDFFK:
    """Small self-contained URDF FK helper for G1 CSV motion conversion."""

    def __init__(self, urdf_path: str, root_link: str = "pelvis") -> None:
        self.urdf_path = urdf_path
        self.root_link = root_link
        self.joints = self._parse_joints(urdf_path)
        self.children: dict[str, list[URDFJoint]] = defaultdict(list)
        for joint in self.joints:
            self.children[joint.parent].append(joint)

    def forward(
        self,
        root_pos_w: np.ndarray,
        root_quat_xyzw: np.ndarray,
        joint_pos: dict[str, float],
        body_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute world body poses for selected links.

        Args:
            root_pos_w: Root link position in world frame, shape ``(3,)``.
            root_quat_xyzw: Root link quaternion in scipy order, shape ``(4,)``.
            joint_pos: Mapping from joint name to scalar joint position.
            body_names: Ordered link names to export.

        Returns:
            A tuple ``(body_pos_w, body_quat_xyzw)`` with shapes ``(B, 3)`` and
            ``(B, 4)``.
        """
        root_rotation = Rotation.from_quat(root_quat_xyzw).as_matrix()
        root_transform = np.eye(4)
        root_transform[:3, :3] = root_rotation
        root_transform[:3, 3] = root_pos_w

        transforms = {self.root_link: root_transform}
        queue = deque([self.root_link])
        while queue:
            parent = queue.popleft()
            parent_transform = transforms[parent]
            for joint in self.children.get(parent, ()):
                transforms[joint.child] = parent_transform @ self._joint_transform(
                    joint, joint_pos.get(joint.name, 0.0)
                )
                queue.append(joint.child)

        missing = [name for name in body_names if name not in transforms]
        if missing:
            raise ValueError(f"URDF FK could not resolve body links: {missing}")

        body_pos_w = np.stack([transforms[name][:3, 3] for name in body_names], axis=0)
        body_quat_xyzw = np.stack(
            [Rotation.from_matrix(transforms[name][:3, :3]).as_quat() for name in body_names], axis=0
        )
        return body_pos_w, _normalize_quat_xyzw(body_quat_xyzw)

    @staticmethod
    def _parse_joints(urdf_path: str) -> list[URDFJoint]:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints: list[URDFJoint] = []
        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib["name"]
            joint_type = joint_elem.attrib.get("type", "fixed")
            if joint_type == "floating":
                continue

            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                continue

            origin_elem = joint_elem.find("origin")
            origin_xyz = _parse_xyz(origin_elem.attrib.get("xyz", "0 0 0") if origin_elem is not None else "0 0 0")
            origin_rpy = _parse_xyz(origin_elem.attrib.get("rpy", "0 0 0") if origin_elem is not None else "0 0 0")

            axis_elem = joint_elem.find("axis")
            axis = _parse_xyz(axis_elem.attrib.get("xyz", "1 0 0") if axis_elem is not None else "1 0 0")
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0.0:
                axis = axis / axis_norm

            joints.append(
                URDFJoint(
                    name=name,
                    joint_type=joint_type,
                    parent=parent_elem.attrib["link"],
                    child=child_elem.attrib["link"],
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    axis=axis,
                )
            )
        return joints

    @staticmethod
    def _joint_transform(joint: URDFJoint, joint_position: float) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", joint.origin_rpy).as_matrix()
        transform[:3, 3] = joint.origin_xyz

        if joint.joint_type in ("revolute", "continuous"):
            rotation_transform = np.eye(4)
            rotation_transform[:3, :3] = Rotation.from_rotvec(joint.axis * joint_position).as_matrix()
            transform = transform @ rotation_transform
        elif joint.joint_type != "fixed":
            raise ValueError(f"Unsupported URDF joint type '{joint.joint_type}' for joint '{joint.name}'.")
        return transform


def resample_motion(
    root_pos_w: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample root pose and joint positions from ``input_fps`` to ``output_fps``."""
    if root_pos_w.shape[0] < 2:
        raise ValueError("Motion must contain at least two frames.")

    frame_count = root_pos_w.shape[0]
    duration = (frame_count - 1) / input_fps
    source_t = np.arange(frame_count, dtype=np.float64) / input_fps
    target_count = int(np.floor(duration * output_fps)) + 1
    target_t = np.arange(target_count, dtype=np.float64) / output_fps
    target_t[-1] = min(target_t[-1], source_t[-1])

    root_pos_resampled = _interp_columns(source_t, target_t, root_pos_w)
    joint_pos_resampled = _interp_columns(source_t, target_t, joint_pos)
    root_quat_resampled = Rotation.from_quat(root_quat_xyzw)
    root_quat_resampled = _slerp(source_t, root_quat_resampled, target_t).as_quat()
    return root_pos_resampled, _normalize_quat_xyzw(root_quat_resampled), joint_pos_resampled


def finite_difference_linear(values: np.ndarray, fps: float) -> np.ndarray:
    """Compute linear finite-difference velocity with stable endpoint handling."""
    edge_order = 2 if values.shape[0] > 2 else 1
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=edge_order)


def finite_difference_angular(quat_xyzw: np.ndarray, fps: float) -> np.ndarray:
    """Compute world-frame angular velocity from body-to-world quaternions."""
    frame_count = quat_xyzw.shape[0]
    flat_quat = quat_xyzw.reshape(frame_count, -1, 4)
    ang_vel = np.zeros((frame_count, flat_quat.shape[1], 3), dtype=np.float64)
    rotations = Rotation.from_quat(flat_quat.reshape(-1, 4)).as_matrix().reshape(frame_count, -1, 3, 3)
    dt = 1.0 / fps

    for frame_idx in range(frame_count):
        if frame_idx == 0:
            prev_idx, next_idx, scale = 0, 1, dt
        elif frame_idx == frame_count - 1:
            prev_idx, next_idx, scale = frame_count - 2, frame_count - 1, dt
        else:
            prev_idx, next_idx, scale = frame_idx - 1, frame_idx + 1, 2.0 * dt

        delta = rotations[next_idx] @ np.swapaxes(rotations[prev_idx], -1, -2)
        ang_vel[frame_idx] = Rotation.from_matrix(delta).as_rotvec() / scale

    return ang_vel.reshape(quat_xyzw.shape[:-1] + (3,))


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert scipy quaternion order ``xyzw`` to training order ``wxyz``."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def _interp_columns(source_t: np.ndarray, target_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    flat_values = values.reshape(values.shape[0], -1)
    out = np.empty((target_t.shape[0], flat_values.shape[1]), dtype=np.float64)
    for col in range(flat_values.shape[1]):
        out[:, col] = np.interp(target_t, source_t, flat_values[:, col])
    return out.reshape((target_t.shape[0],) + values.shape[1:])


def _slerp(source_t: np.ndarray, rotations: Rotation, target_t: np.ndarray) -> Rotation:
    from scipy.spatial.transform import Slerp

    return Slerp(source_t, rotations)(target_t)


def _parse_xyz(text: str) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    return quat_xyzw / np.clip(norm, 1.0e-12, None)
