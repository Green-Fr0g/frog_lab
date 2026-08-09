#!/usr/bin/env python3
"""Convert G1 29-DOF AMP CSV motions into body-state NPZ assets."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
G1_KINEMATICS_PATH = REPO_ROOT / "source" / "frog_lab" / "frog_lab" / "tasks" / "amp" / "utils" / "g1_kinematics.py"
spec = importlib.util.spec_from_file_location("g1_kinematics", G1_KINEMATICS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load {G1_KINEMATICS_PATH}")
g1_kinematics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g1_kinematics
spec.loader.exec_module(g1_kinematics)

G1URDFFK = g1_kinematics.G1URDFFK
finite_difference_angular = g1_kinematics.finite_difference_angular
finite_difference_linear = g1_kinematics.finite_difference_linear
quat_xyzw_to_wxyz = g1_kinematics.quat_xyzw_to_wxyz
resample_motion = g1_kinematics.resample_motion


G1_29DOF_AMP_RAW_MOTION_DIR = REPO_ROOT / "motion_data" / "amp" / "g1"
G1_29DOF_AMP_MOTION_DIR = (
    REPO_ROOT / "source" / "frog_lab" / "frog_lab" / "assets" / "motions" / "g1" / "amp" / "locomotion"
)
G1_29DOF_AMP_ALL_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=G1_29DOF_AMP_RAW_MOTION_DIR,
        help="Directory containing raw G1 AMP CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=G1_29DOF_AMP_MOTION_DIR,
        help="Directory where converted NPZ files will be written.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "model" / "g1" / "urdf" / "g1_29dof_rev_1_0.urdf",
        help="G1 29-DOF URDF path.",
    )
    parser.add_argument("--input-fps", type=float, default=120.0, help="Raw CSV frame rate.")
    parser.add_argument("--output-fps", type=float, default=50.0, help="Converted NPZ frame rate.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fk = G1URDFFK(str(args.urdf), root_link="pelvis")

    converted = 0
    skipped = 0
    for csv_path in csv_paths:
        output_path = args.output_dir / f"{csv_path.stem}.npz"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        motion = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        if motion.ndim != 2 or motion.shape[1] != 36:
            raise ValueError(f"{csv_path} must have shape (frames, 36), got {motion.shape}")

        root_pos_w = motion[:, 0:3]
        root_quat_xyzw = motion[:, 3:7]
        joint_pos = motion[:, 7:36]

        root_pos_w, root_quat_xyzw, joint_pos = resample_motion(
            root_pos_w,
            root_quat_xyzw,
            joint_pos,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
        )

        body_pos_w = np.empty((root_pos_w.shape[0], len(G1_29DOF_AMP_ALL_BODY_NAMES), 3), dtype=np.float64)
        body_quat_xyzw = np.empty((root_pos_w.shape[0], len(G1_29DOF_AMP_ALL_BODY_NAMES), 4), dtype=np.float64)
        for frame_idx in range(root_pos_w.shape[0]):
            frame_joint_pos = dict(zip(G1_29DOF_JOINT_NAMES, joint_pos[frame_idx], strict=True))
            body_pos_w[frame_idx], body_quat_xyzw[frame_idx] = fk.forward(
                root_pos_w[frame_idx],
                root_quat_xyzw[frame_idx],
                frame_joint_pos,
                G1_29DOF_AMP_ALL_BODY_NAMES,
            )

        joint_vel = finite_difference_linear(joint_pos, args.output_fps)
        body_lin_vel_w = finite_difference_linear(body_pos_w, args.output_fps)
        body_ang_vel_w = finite_difference_angular(body_quat_xyzw, args.output_fps)

        np.savez_compressed(
            output_path,
            fps=np.asarray(args.output_fps, dtype=np.float32),
            body_names=np.asarray(G1_29DOF_AMP_ALL_BODY_NAMES),
            joint_names=np.asarray(G1_29DOF_JOINT_NAMES),
            root_pos_w=root_pos_w.astype(np.float32),
            root_quat_w=quat_xyzw_to_wxyz(root_quat_xyzw).astype(np.float32),
            joint_pos=joint_pos.astype(np.float32),
            joint_vel=joint_vel.astype(np.float32),
            body_pos_w=body_pos_w.astype(np.float32),
            body_quat_w=quat_xyzw_to_wxyz(body_quat_xyzw).astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=body_ang_vel_w.astype(np.float32),
            source_file=str(csv_path),
            input_fps=np.asarray(args.input_fps, dtype=np.float32),
        )
        converted += 1
        print(f"converted {csv_path.name} -> {output_path}")

    print(f"done: converted={converted}, skipped={skipped}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
