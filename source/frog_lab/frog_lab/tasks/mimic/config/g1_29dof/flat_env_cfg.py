import os

from isaaclab.utils import configclass

from frog_lab.assets.g1_mimic import G1_ACTION_SCALE, G1_CYLINDER_CFG
from frog_lab.tasks.mimic import MIMIC_DIR
from frog_lab.tasks.mimic.config.g1_29dof.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from frog_lab.tasks.mimic.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1_29DOFFlatEnvCfg(TrackingEnvCfg):

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

    anchor_body_name = "torso_link"
    motion_file = os.path.join(MIMIC_DIR, "config", "g1_29dof", "motions", "G1_gangnam_style_V01.npz")

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        self.commands.motion.pose_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }
        self.commands.motion.joint_position_range = (-0.1, 0.1),
        self.commands.motion.motion_file = self.motion_file
        self.commands.motion.anchor_body_name = self.anchor_body_name
        self.commands.motion.body_names = [
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
        ]

        self.events.base_com.params["asset_cfg"].body_names = self.anchor_body_name
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }
        
        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 1.0
        self.rewards.motion_body_lin_vel.weight = 1.0
        self.rewards.motion_body_ang_vel.weight = 1.0
        self.rewards.action_rate_l2.weight = -1e-1
        self.rewards.joint_limit.weight = -10.0
        self.rewards.undesired_contacts.weight = -0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
        ]
        self.terminations.anchor_pos.params["threshold"] = 0.25
        self.terminations.anchor_ori.params["threshold"] = 0.8
        self.terminations.ee_body_pos.params["threshold"] = 0.25
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1_29DOFFlatWoStateEstimationEnvCfg(G1_29DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1_29DOFFlatLowFreqEnvCfg(G1_29DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
