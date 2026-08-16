from __future__ import annotations

import os

from isaaclab.utils import configclass

from frog_lab.assets.g1_mimic import G1_ACTION_SCALE, G1_CYLINDER_CFG
from frog_lab.tasks.amp import AMP_DIR
from frog_lab.tasks.amp.amp_env_cfg import AmpFlatEnvCfg


@configclass
class G1_29DOFAmpFlatEnvCfg(AmpFlatEnvCfg):
    """G1 29-DOF velocity-conditioned AMP on flat terrain."""

    base_link_name = "torso_link"
    root_link_name = "pelvis"
    foot_link_name = ".*_ankle_roll_link"
    anchor_body_name = "torso_link"
    motion_dir = os.path.join(AMP_DIR, "config", "g1_29dof", "motions", "WalkandRun")

    link_names = [
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]
    amp_all_body_names = ["pelvis", *link_names]

    joint_names = [
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
    ]

    amp_body_names = [
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)
        self.commands.base_velocity.ranges.heading = (-1.57, 1.57)

        self.events.init_motion_loader.params["motion_dir"] = self.motion_dir
        self.events.init_motion_loader.params["root_name"] = self.root_link_name
        self.events.init_motion_loader.params["all_body_names"] = tuple(self.amp_all_body_names)
        self.events.reset_from_motion.params["motion_dir"] = self.motion_dir
        self.events.reset_from_motion.params["root_name"] = self.root_link_name
        self.events.reset_from_motion.params["all_body_names"] = tuple(self.amp_all_body_names)
        self.events.reset_from_motion.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.critic.body_pos_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.critic.body_pos_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.critic.body_ori_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.critic.body_ori_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp_state.body_pos_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_pos_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp_state.body_ori_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_ori_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp_state.body_lin_vel_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_lin_vel_b.params["body_cfg"].body_names = self.amp_body_names
        self.observations.amp_state.body_ang_vel_b.params["anchor_cfg"].body_names = (self.anchor_body_name,)
        self.observations.amp_state.body_ang_vel_b.params["body_cfg"].body_names = self.amp_body_names

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

        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2.params["target_height"] = 0.78
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.base_height.params["minimum_height"] = 0.2
        self.terminations.bad_orientation.params["limit_angle"] = 0.7

        self.disable_zero_weight_rewards()
