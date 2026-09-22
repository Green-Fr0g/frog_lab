"""Configuration for the DeepRobotics DR02-Pro humanoid."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from frog_lab.assets import FROG_LAB_DATA_DIR


DR02_URDF_PATH = f"{FROG_LAB_DATA_DIR}/DR02/urdf/DR02-pro.urdf"

DR02_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=DR02_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0 * 57.1,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=False,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            "left_hip_y_joint": -0.1,
            "left_hip_x_joint": 0.0,
            "left_hip_z_joint": 0.0,
            "left_knee_joint": 0.2,
            "left_ankle_y_joint": -0.1,
            "left_ankle_x_joint": 0.0,
            "right_hip_y_joint": -0.1,
            "right_hip_x_joint": 0.0,
            "right_hip_z_joint": 0.0,
            "right_knee_joint": 0.2,
            "right_ankle_y_joint": -0.1,
            "right_ankle_x_joint": 0.0,
            "waist_z_joint": 0.0,
            "left_shoulder_y_joint": 0.0,
            "left_shoulder_x_joint": 0.15,
            "left_shoulder_z_joint": 0.0,
            "left_elbow_joint": 1.35,
            "right_shoulder_y_joint": 0.0,
            "right_shoulder_x_joint": -0.15,
            "right_shoulder_z_joint": 0.0,
            "right_elbow_joint": 1.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 330.0,
                ".*_hip_z_joint|.*_ankle_y_joint|waist_z_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 105.0,
                ".*_ankle_x_joint": 35.0,
            },
            velocity_limit_sim={".*": 1.0e7},
            stiffness={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 250.0,
                ".*_hip_z_joint": 180.0,
                ".*_ankle_y_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 100.0,
                "waist_z_joint": 150.0,
                ".*_ankle_x_joint": 40.0,
            },
            damping={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 6.0,
                ".*_hip_z_joint": 4.0,
                ".*_ankle_y_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 2.5,
                "waist_z_joint": 3.0,
                ".*_ankle_x_joint": 1.0,
            },
        )
    },
)

DR02_ROOT_LINK_NAME = "base_link"
DR02_ALL_JOINT_NAMES = (
    "waist_z_joint", "waist_x_joint", "waist_y_joint", "left_shoulder_y_joint", "left_shoulder_x_joint",
    "left_shoulder_z_joint", "left_elbow_joint", "left_wrist_z_joint", "left_wrist_y_joint",
    "left_wrist_x_joint", "right_shoulder_y_joint", "right_shoulder_x_joint", "right_shoulder_z_joint",
    "right_elbow_joint", "right_wrist_z_joint", "right_wrist_y_joint", "right_wrist_x_joint",
    "neck_z_joint", "neck_y_joint", "left_hip_y_joint", "left_hip_x_joint", "left_hip_z_joint",
    "left_knee_joint", "left_ankle_y_joint", "left_ankle_x_joint", "right_hip_y_joint", "right_hip_x_joint",
    "right_hip_z_joint", "right_knee_joint", "right_ankle_y_joint", "right_ankle_x_joint",
)
DR02_CONTROL_JOINT_NAMES = tuple(
    name for name in DR02_ALL_JOINT_NAMES if name not in {"neck_z_joint", "neck_y_joint"}
)
DR02_AMP_JOINT_NAMES = DR02_CONTROL_JOINT_NAMES
DR02_BODY_NAMES = (
    "base_link", "waist_z_link", "waist_x_link", "body", "left_shoulder_y_link", "left_shoulder_x_link",
    "left_shoulder_z_link", "left_elbow_link", "left_wrist_z_link", "left_wrist_y_link", "left_wrist_x_link",
    "right_shoulder_y_link", "right_shoulder_x_link", "right_shoulder_z_link", "right_elbow_link",
    "right_wrist_z_link", "right_wrist_y_link", "right_wrist_x_link", "neck_link", "head_link",
    "left_hip_y_link", "left_hip_x_link", "left_hip_z_link", "left_knee_link", "left_ankle_y_link",
    "left_ankle_x_link", "right_hip_y_link", "right_hip_x_link", "right_hip_z_link", "right_knee_link",
    "right_ankle_y_link", "right_ankle_x_link",
)
DR02_ACTION_SCALE = {}
for actuator_cfg in DR02_CFG.actuators.values():
    effort_limits = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    if not isinstance(effort_limits, dict):
        effort_limits = {name: effort_limits for name in actuator_cfg.joint_names_expr}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in actuator_cfg.joint_names_expr}
    for name in actuator_cfg.joint_names_expr:
        if name in effort_limits and name in stiffness and stiffness[name]:
            DR02_ACTION_SCALE[name] = 0.25 * effort_limits[name] / stiffness[name]

DR02_ACTION_SCALE = {
    name: value for name, value in DR02_ACTION_SCALE.items() if name in DR02_CONTROL_JOINT_NAMES
}
