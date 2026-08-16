from __future__ import annotations

import os

from isaaclab.utils import configclass

from frog_lab.assets.g1_mimic import G1_ACTION_SCALE, G1_CYLINDER_CFG
from frog_lab.tasks.amp import AMP_DIR
from frog_lab.tasks.amp.wasabi_env_cfg import WasabiFlatEnvCfg


@configclass
class G1_29DOFWasabiFlatEnvCfg(WasabiFlatEnvCfg):
    """G1 29-DOF flat WASABI task."""

    base_link_name = "torso_link"
    root_link_name = "pelvis"
    foot_link_name = ".*_ankle_roll_link"
    anchor_body_name = "torso_link"
    motion_dir = os.path.join(AMP_DIR, "config", "g1_29dof", "motions")

    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
        "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    link_names = [
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
        "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link", "left_shoulder_pitch_link",
        "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
        "left_wrist_pitch_link", "left_wrist_yaw_link", "right_shoulder_pitch_link",
        "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
        "right_wrist_pitch_link", "right_wrist_yaw_link",
    ]
    all_body_names = ["pelvis", *link_names]
    wasabi_body_names = [
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "left_shoulder_roll_link",
        "left_elbow_link", "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
        "right_wrist_yaw_link",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.wasabi_policy.joint_pos_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.wasabi_policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.wasabi_reference.joint_pos_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.wasabi_reference.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.events.init_wasabi_motion_reference.params.update(
            {
                "motion_files": self.motion_dir,
                "body_names": tuple(self.wasabi_body_names),
                "anchor_name": self.anchor_body_name,
                "root_name": self.root_link_name,
                "all_body_names": tuple(self.all_body_names),
                "joint_names": tuple(self.joint_names),
                "time_between_frames": 0.02,
            }
        )
        self.events.reset_wasabi_motion_reference.params["asset_cfg"].joint_names = self.joint_names

        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        self.rewards.is_terminated.weight = -200.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.feet_slide.weight = -0.2
        self.rewards.undesired_contacts.weight = -1.0
        self.disable_zero_weight_rewards()
