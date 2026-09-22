"""Configuration for the EngineAI PM01 humanoid."""

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from frog_lab.assets import FROG_LAB_DATA_DIR


PM01_URDF_PATH = f"{FROG_LAB_DATA_DIR}/pm01/urdf/pm01.urdf"

# PM01 motor and control parameters from the official Isaac Lab asset config.
ARMATURE_Q90 = 0.0453
EFFORT_LIMIT_Q90 = 164.0
VELOCITY_LIMIT_Q90 = 26.3
ARMATURE_Q25 = 0.0067
EFFORT_LIMIT_Q25 = 52.0
VELOCITY_LIMIT_Q25 = 35.2
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ


PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=PM01_URDF_PATH,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_HIP_PITCH.*": -0.06,
            ".*_KNEE_PITCH.*": 0.12,
            ".*_ANKLE_PITCH.*": -0.06,
            ".*_ELBOW_PITCH.*": -0.25,
            "J14_SHOULDER_ROLL_L": 0.15,
            "J19_SHOULDER_ROLL_R": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP.*", ".*KNEE.*"],
            effort_limit_sim={
                ".*_HIP_PITCH.*": EFFORT_LIMIT_Q90,
                ".*_HIP_ROLL.*": EFFORT_LIMIT_Q90,
                ".*_HIP_YAW.*": EFFORT_LIMIT_Q25,
                ".*_KNEE_PITCH.*": EFFORT_LIMIT_Q90,
            },
            velocity_limit_sim={
                ".*_HIP_PITCH.*": VELOCITY_LIMIT_Q90,
                ".*_HIP_ROLL.*": VELOCITY_LIMIT_Q90,
                ".*_HIP_YAW.*": VELOCITY_LIMIT_Q25,
                ".*_KNEE_PITCH.*": VELOCITY_LIMIT_Q90,
            },
            stiffness={
                ".*_HIP_PITCH.*": STIFFNESS_Q90,
                ".*_HIP_ROLL.*": STIFFNESS_Q90,
                ".*_HIP_YAW.*": STIFFNESS_Q25,
                ".*_KNEE_PITCH.*": STIFFNESS_Q90,
            },
            damping={
                ".*_HIP_PITCH.*": DAMPING_Q90,
                ".*_HIP_ROLL.*": DAMPING_Q90,
                ".*_HIP_YAW.*": DAMPING_Q25,
                ".*_KNEE_PITCH.*": DAMPING_Q90,
            },
            armature={
                ".*_HIP_PITCH.*": ARMATURE_Q90,
                ".*_HIP_ROLL.*": ARMATURE_Q90,
                ".*_HIP_YAW.*": ARMATURE_Q25,
                ".*_KNEE_PITCH.*": ARMATURE_Q90,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=0.5,
            armature=ARMATURE_Q25,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J12_WAIST_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=DAMPING_Q25,
            armature=ARMATURE_Q25,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*"],
            effort_limit_sim={".*SHOULDER.*": EFFORT_LIMIT_Q25, ".*ELBOW.*": EFFORT_LIMIT_Q25},
            velocity_limit_sim={".*SHOULDER.*": VELOCITY_LIMIT_Q25, ".*ELBOW.*": VELOCITY_LIMIT_Q25},
            stiffness={".*SHOULDER.*": STIFFNESS_Q25, ".*ELBOW.*": STIFFNESS_Q25},
            damping={".*SHOULDER.*": DAMPING_Q25, ".*ELBOW.*": DAMPING_Q25},
            armature={".*SHOULDER.*": ARMATURE_Q25, ".*ELBOW.*": ARMATURE_Q25},
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["J23_HEAD_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q25,
            velocity_limit_sim=VELOCITY_LIMIT_Q25,
            stiffness=STIFFNESS_Q25,
            damping=DAMPING_Q25,
            armature=ARMATURE_Q25,
        ),
    },
)


PM01_ROOT_LINK_NAME = "LINK_BASE"
PM01_ALL_JOINT_NAMES = (
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J06_HIP_PITCH_R", "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW", "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L", "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R", "J23_HEAD_YAW",
)
PM01_CONTROL_JOINT_NAMES = tuple(PM01_ALL_JOINT_NAMES[:-1])
PM01_AMP_JOINT_NAMES = PM01_CONTROL_JOINT_NAMES
PM01_BODY_NAMES = (
    "LINK_BASE", "LINK_HIP_PITCH_L", "LINK_HIP_ROLL_L", "LINK_HIP_YAW_L", "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L", "LINK_ANKLE_ROLL_L", "LINK_HIP_PITCH_R", "LINK_HIP_ROLL_R", "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R", "LINK_ANKLE_PITCH_R", "LINK_ANKLE_ROLL_R", "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_ROLL_L", "LINK_SHOULDER_YAW_L", "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L", "LINK_SHOULDER_PITCH_R", "LINK_SHOULDER_ROLL_R", "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R", "LINK_HEAD_YAW",
)


def _pm01_actuator_value(joint_name: str, attribute: str) -> float | None:
    """Look up a per-joint actuator value the way IsaacLab resolves it.

    The actuator group whose ``joint_names_expr`` matches the joint owns it, and within that group
    the first regex pattern that fully matches the joint wins. Entries matching no pattern resolve
    to 0.0, which leaves the joint undriven (i.e. free-floating).
    """
    for actuator_cfg in PM01_CFG.actuators.values():
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


def _pm01_action_scale() -> dict[str, float]:
    """Map each controlled joint to its action scale, ``0.25 * effort_limit / stiffness``."""
    scales = {}
    for joint_name in PM01_CONTROL_JOINT_NAMES:
        effort_limit = _pm01_actuator_value(joint_name, "effort_limit_sim")
        stiffness = _pm01_actuator_value(joint_name, "stiffness")
        if effort_limit and stiffness:
            scales[joint_name] = 0.25 * effort_limit / stiffness
    return scales


PM01_ACTION_SCALE = _pm01_action_scale()
