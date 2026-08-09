from __future__ import annotations

from isaaclab.utils import configclass

from frog_lab.tasks.locomotion.config.g1_29dof.flat_env_cfg import G1_29DOFFlatEnvCfg

from .base_env_cfg import AMPObservationsCfg
from .common import G1_29DOF_AMP_ANCHOR_NAME, G1_29DOF_AMP_BODY_NAMES


@configclass
class G1_29DOFAmpFlatEnvCfg(G1_29DOFFlatEnvCfg):
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
