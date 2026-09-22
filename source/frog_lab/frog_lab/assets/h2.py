"""Configuration for the Unitree H2 humanoid."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from frog_lab.assets import FROG_LAB_DATA_DIR


H2_URDF_PATH = f"{FROG_LAB_DATA_DIR}/h2/urdf/h2.urdf"


H2_ROOT_LINK_NAME = "pelvis"
H2_ALL_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
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
H2_CONTROL_JOINT_NAMES = tuple(
    name for name in H2_ALL_JOINT_NAMES if name not in {"head_pitch_joint", "head_yaw_joint"}
)
H2_AMP_JOINT_NAMES = H2_CONTROL_JOINT_NAMES
H2_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_pitch_link",
    "head_yaw_link",
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
)
H2_ACTION_SCALE: dict[str, float] = {}


H2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=H2_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03),
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.25,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=360.0,
            stiffness=200.0,
            damping=4.0,
            armature=0.01,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=19.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=66.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=120.0,
            stiffness=150.0,
            damping=3.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=54.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=25.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)


H2_DOF_ORDER = list(H2_ALL_JOINT_NAMES)
