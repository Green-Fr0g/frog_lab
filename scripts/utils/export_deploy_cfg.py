"""Export deployment parameters to ``deploy.yaml`` for the C++ deploy side.

The exported file lives next to ``params/env.yaml`` in the training log
directory and, together with the exported ``policy.onnx``, forms a complete
deployment package: the yaml carries every environment-side numeric parameter
(observation scaling/clipping, action scaling/offset/clipping, control rate,
joint mapping, PD gains) needed to run the bare policy on real hardware.

The mechanism is ported from ``unitree_rl_lab`` with one simplification: there
is no SDK joint remapping, ``joint_ids_map`` lists the actual joint names in
policy (i.e. IsaacLab) order.
"""

import os

import numpy as np
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv


def format_value(x):
    """Round floats to 3 significant digits for a compact yaml file."""
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    """Export the environment-side parameters of ``env`` to ``log_dir/params/deploy.yaml``.

    Only the ``policy`` observation group is exported; critic/privileged groups
    are not needed on the deploy side. Call this once before training starts.
    """
    asset: Articulation = env.scene["robot"]

    cfg = {}  # noqa: SIM904

    # --- joint mapping ---
    # No SDK remapping: the actual joint names in policy order. The i-th action
    # dimension drives the i-th entry of this list.
    cfg["joint_ids_map"] = list(asset.data.joint_names)

    # --- control rate ---
    cfg["sim_dt"] = env.cfg.sim.dt
    cfg["decimation"] = env.cfg.decimation

    # --- actuator gains and default joint state (in joint_ids_map order) ---
    cfg["stiffness"] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["damping"] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
