"""Configuration for the Booster T1 humanoid."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from frog_lab.assets import FROG_LAB_DATA_DIR


T1_URDF_PATH = f"{FROG_LAB_DATA_DIR}/t1/urdf/t1.urdf"

T1_ROOT_LINK_NAME = "Trunk"
T1_ALL_JOINT_NAMES = (
    "AAHead_yaw", "Head_pitch", "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
    "Left_Elbow_Yaw", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
    "Right_Elbow_Yaw", "Waist", "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
T1_CONTROL_JOINT_NAMES = tuple(T1_ALL_JOINT_NAMES[2:])
T1_AMP_JOINT_NAMES = T1_CONTROL_JOINT_NAMES
T1_BODY_NAMES = (
    "Trunk", "H1", "H2", "AL1", "AL2", "AL3", "left_hand_link", "AR1", "AR2", "AR3",
    "right_hand_link", "Waist", "Hip_Pitch_Left", "Hip_Roll_Left", "Hip_Yaw_Left", "Shank_Left",
    "Ankle_Cross_Left", "left_foot_link", "Hip_Pitch_Right", "Hip_Roll_Right", "Hip_Yaw_Right",
    "Shank_Right", "Ankle_Cross_Right", "right_foot_link",
)

_NATURAL_FREQ = 2.0 * 3.1415926535 * 10.0
_DAMPING_FACTOR = 2.0 * 2.0 * _NATURAL_FREQ


def _pd_gains(armature: float) -> tuple[float, float]:
    """Return the official Booster default stiffness and damping for a motor inertia."""
    return armature * _NATURAL_FREQ**2, armature * _DAMPING_FACTOR


T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=T1_URDF_PATH,
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
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim=38.3,
            velocity_limit_sim=17.59,
            stiffness=_pd_gains(0.0282528)[0],
            damping=_pd_gains(0.0282528)[1],
            armature=0.0282528,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=68.0,
            velocity_limit_sim=14.66,
            stiffness=_pd_gains(0.0478125)[0],
            damping=_pd_gains(0.0478125)[1],
            armature=0.0478125,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 96.0,
                ".*_Hip_Roll": 68.0,
                ".*_Hip_Yaw": 68.0,
                ".*_Knee_Pitch": 130.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 16.76,
                ".*_Hip_Roll": 14.66,
                ".*_Hip_Yaw": 14.66,
                ".*_Knee_Pitch": 14.66,
            },
            stiffness={
                ".*_Hip_Pitch": _pd_gains(0.0523908)[0],
                ".*_Hip_Roll": _pd_gains(0.0478125)[0],
                ".*_Hip_Yaw": _pd_gains(0.0478125)[0],
                ".*_Knee_Pitch": _pd_gains(0.095625)[0],
            },
            damping={
                ".*_Hip_Pitch": _pd_gains(0.0523908)[1],
                ".*_Hip_Roll": _pd_gains(0.0478125)[1],
                ".*_Hip_Yaw": _pd_gains(0.0478125)[1],
                ".*_Knee_Pitch": _pd_gains(0.095625)[1],
            },
            armature={
                ".*_Hip_Pitch": 0.0523908,
                ".*_Hip_Roll": 0.0478125,
                ".*_Hip_Yaw": 0.0478125,
                ".*_Knee_Pitch": 0.095625,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={".*_Ankle_Pitch": 76.0, ".*_Ankle_Roll": 76.0},
            velocity_limit_sim={".*_Ankle_Pitch": 12.57, ".*_Ankle_Roll": 12.57},
            stiffness=_pd_gains(0.0339552)[0],
            damping=_pd_gains(0.0339552)[1],
            # The parallel ankle wrapper doubles the reflected armature.
            armature=2.0 * 0.0339552,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=12.57,
            stiffness=_pd_gains(0.0018)[0],
            damping=_pd_gains(0.0018)[1],
            armature=0.0018,
        ),
    },
)


T1_ACTION_SCALE = {}
for actuator_cfg in T1_CFG.actuators.values():
    effort_limits = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    if not isinstance(effort_limits, dict):
        effort_limits = {name: effort_limits for name in actuator_cfg.joint_names_expr}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in actuator_cfg.joint_names_expr}
    for name in actuator_cfg.joint_names_expr:
        if name in effort_limits and name in stiffness and stiffness[name]:
            T1_ACTION_SCALE[name] = 0.25 * effort_limits[name] / stiffness[name]

T1_ACTION_SCALE = {
    name: value for name, value in T1_ACTION_SCALE.items() if name in T1_CONTROL_JOINT_NAMES
}
