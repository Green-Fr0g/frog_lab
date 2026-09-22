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

