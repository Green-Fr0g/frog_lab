from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from frog_lab.tasks.amp import mdp
from frog_lab.tasks.amp.config.g1.common import G1_29DOF_AMP_ANCHOR_NAME, G1_29DOF_AMP_BODY_NAMES
from frog_lab.tasks.locomotion.config.g1_29dof.rough_env_cfg import G1_29DOFRoughEnvCfg
from frog_lab.tasks.locomotion.locomotion_env_cfg import ObservationsCfg as LocomotionObservationsCfg


@configclass
class AMPObservationsCfg(LocomotionObservationsCfg):
    @configclass
    class AmpStateCfg(ObsGroup):
        body_pos_b = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=()),
                "body_cfg": SceneEntityCfg("robot", body_names=()),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_ori_b = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=()),
                "body_cfg": SceneEntityCfg("robot", body_names=()),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_lin_vel_b = ObsTerm(
            func=mdp.robot_body_lin_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=()),
                "body_cfg": SceneEntityCfg("robot", body_names=()),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        body_ang_vel_b = ObsTerm(
            func=mdp.robot_body_ang_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=()),
                "body_cfg": SceneEntityCfg("robot", body_names=()),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    amp_state: AmpStateCfg = AmpStateCfg()


@configclass
class G1_29DOFAmpRoughEnvCfg(G1_29DOFRoughEnvCfg):
    observations: AMPObservationsCfg = AMPObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.amp_state.body_pos_b.params["anchor_cfg"].body_names = (G1_29DOF_AMP_ANCHOR_NAME,)
        self.observations.amp_state.body_pos_b.params["body_cfg"].body_names = G1_29DOF_AMP_BODY_NAMES
        self.observations.amp_state.body_ori_b.params["anchor_cfg"].body_names = (G1_29DOF_AMP_ANCHOR_NAME,)
        self.observations.amp_state.body_ori_b.params["body_cfg"].body_names = G1_29DOF_AMP_BODY_NAMES
        self.observations.amp_state.body_lin_vel_b.params["anchor_cfg"].body_names = (G1_29DOF_AMP_ANCHOR_NAME,)
        self.observations.amp_state.body_lin_vel_b.params["body_cfg"].body_names = G1_29DOF_AMP_BODY_NAMES
        self.observations.amp_state.body_ang_vel_b.params["anchor_cfg"].body_names = (G1_29DOF_AMP_ANCHOR_NAME,)
        self.observations.amp_state.body_ang_vel_b.params["body_cfg"].body_names = G1_29DOF_AMP_BODY_NAMES
        self.disable_zero_weight_rewards()
