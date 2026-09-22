"""Configuration for the Unitree G1 23-DOF humanoid."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from frog_lab.assets import FROG_LAB_DATA_DIR

G1_23DOF_URDF_PATH = f"{FROG_LAB_DATA_DIR}/g1/urdf/g1_23dof_rev_1_0.urdf"

G1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=str(G1_23DOF_URDF_PATH),
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
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll_.*"],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "N5020-16-parallel": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

G1_23DOF_ROOT_LINK_NAME = "pelvis"
G1_23DOF_ALL_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
)
G1_23DOF_CONTROL_JOINT_NAMES = G1_23DOF_ALL_JOINT_NAMES
G1_23DOF_AMP_JOINT_NAMES = G1_23DOF_CONTROL_JOINT_NAMES
G1_23DOF_BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_rubber_hand", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_rubber_hand",
)
G1_23DOF_ACTION_SCALE = {}
for actuator_cfg in G1_23DOF_CFG.actuators.values():
    effort_limits = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    if not isinstance(effort_limits, dict):
        effort_limits = {name: effort_limits for name in actuator_cfg.joint_names_expr}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in actuator_cfg.joint_names_expr}
    for name in actuator_cfg.joint_names_expr:
        if name in effort_limits and name in stiffness and stiffness[name]:
            G1_23DOF_ACTION_SCALE[name] = 0.25 * effort_limits[name] / stiffness[name]

G1_23DOF_ACTION_SCALE = {
    name: value for name, value in G1_23DOF_ACTION_SCALE.items() if name in G1_23DOF_CONTROL_JOINT_NAMES
}
