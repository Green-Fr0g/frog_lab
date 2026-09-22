"""Configuration for the Booster T1 humanoid."""

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from frog_lab.assets import FROG_LAB_DATA_DIR


T1_URDF_PATH = f"{FROG_LAB_DATA_DIR}/t1/urdf/t1.urdf"



NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

ARMATURE_ARM = 0.0282528
ARMATURE_WAIST = 0.0478125
ARMATURE_HIP_PITCH = 0.0523908
ARMATURE_KNEE = 0.095625
ARMATURE_ANKLE = 0.0339552
ARMATURE_HEAD = 0.0018

STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ**2
STIFFNESS_WAIST = ARMATURE_WAIST * NATURAL_FREQ**2
STIFFNESS_HIP_PITCH = ARMATURE_HIP_PITCH * NATURAL_FREQ**2
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * NATURAL_FREQ**2

DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ
DAMPING_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_WAIST * NATURAL_FREQ
DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_HIP_PITCH * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
DAMPING_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_HEAD * NATURAL_FREQ


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
            stiffness=STIFFNESS_ARM,
            damping=DAMPING_ARM,
            armature=ARMATURE_ARM,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=68.0,
            velocity_limit_sim=14.66,
            stiffness=STIFFNESS_WAIST,
            damping=DAMPING_WAIST,
            armature=ARMATURE_WAIST,
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
                ".*_Hip_Pitch": STIFFNESS_HIP_PITCH,
                ".*_Hip_Roll": STIFFNESS_WAIST,
                ".*_Hip_Yaw": STIFFNESS_WAIST,
                ".*_Knee_Pitch": STIFFNESS_KNEE,
            },
            damping={
                ".*_Hip_Pitch": DAMPING_HIP_PITCH,
                ".*_Hip_Roll": DAMPING_WAIST,
                ".*_Hip_Yaw": DAMPING_WAIST,
                ".*_Knee_Pitch": DAMPING_KNEE,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_HIP_PITCH,
                ".*_Hip_Roll": ARMATURE_WAIST,
                ".*_Hip_Yaw": ARMATURE_WAIST,
                ".*_Knee_Pitch": ARMATURE_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={".*_Ankle_Pitch": 76.0, ".*_Ankle_Roll": 76.0},
            velocity_limit_sim={".*_Ankle_Pitch": 12.57, ".*_Ankle_Roll": 12.57},
            stiffness=STIFFNESS_ANKLE,
            damping=DAMPING_ANKLE,
            # The parallel ankle wrapper doubles the reflected armature.
            armature=2.0 * ARMATURE_ANKLE,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=12.57,
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=ARMATURE_HEAD,
        ),
    },
)

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

def _t1_actuator_value(joint_name: str, attribute: str) -> float | None:
    """Look up a per-joint actuator value the way IsaacLab resolves it.

    The actuator group whose ``joint_names_expr`` matches the joint owns it, and within that group
    the first regex pattern that fully matches the joint wins. Entries matching no pattern resolve
    to 0.0, which leaves the joint undriven (i.e. free-floating).
    """
    for actuator_cfg in T1_CFG.actuators.values():
        if not any(re.fullmatch(pattern, joint_name) for pattern in actuator_cfg.joint_names_expr):
            continue
        values = getattr(actuator_cfg, attribute, None)
        if values is None:
            return None
        if not isinstance(values, dict):
            return float(values)
        for pattern, value in values.items():
            if re.fullmatch(pattern, joint_name):
                return float(value)
        return 0.0
    return None


def _t1_action_scale() -> dict[str, float]:
    """Map each controlled joint to its action scale, ``0.25 * effort_limit / stiffness``."""
    scales = {}
    for joint_name in T1_CONTROL_JOINT_NAMES:
        effort_limit = _t1_actuator_value(joint_name, "effort_limit_sim")
        stiffness = _t1_actuator_value(joint_name, "stiffness")
        if effort_limit and stiffness:
            scales[joint_name] = 0.25 * effort_limit / stiffness
    return scales


T1_ACTION_SCALE = _t1_action_scale()
