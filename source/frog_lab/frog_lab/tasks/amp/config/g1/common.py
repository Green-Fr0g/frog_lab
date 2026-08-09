from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[7]

G1_29DOF_AMP_ANCHOR_NAME = "torso_link"
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
G1_29DOF_AMP_BODY_NAMES = tuple(name for name in G1_29DOF_AMP_ALL_BODY_NAMES if name != G1_29DOF_AMP_ANCHOR_NAME)
G1_29DOF_AMP_RAW_MOTION_DIR = REPO_ROOT / "motion_data" / "amp" / "g1"
G1_29DOF_AMP_MOTION_DIR = REPO_ROOT / "source" / "frog_lab" / "frog_lab" / "assets" / "motions" / "g1" / "amp" / "locomotion"
