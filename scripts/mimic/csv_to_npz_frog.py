"""Replay a CSV motion into Isaac Sim and export it as an NPZ.

This version is config-driven:
- motion interpretation comes from ``motion_data/config/g1.yaml``
- robot asset selection comes from a small registry in this script

Example:
    python csv_to_npz_frog.py \
        --config motion_data/config/g1.yaml \
        --output_name /tmp/g1_motion.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--config", type=str, default="motion_data/config/g1.yaml", help="Motion config yaml.")
parser.add_argument(
    "--csv_path",
    type=str,
    default=None,
    help="Optional override for motion_data.csv_path. Used by the batch converter.",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "Frame range: START END (both inclusive). The frame index starts from 1. "
        "If not provided, all frames will be loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

#asset registry
from frog_lab.assets.g1_23dof import G1_23DOF_CFG
from frog_lab.assets.g1_mimic import G1_CYLINDER_CFG


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml structure in: {path}")
    return data


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

def _root_quat_to_wxyz(quat: torch.Tensor, order: str) -> torch.Tensor:
    if order == "wxyz":
        return quat
    if order == "xyzw":
        return quat[:, [3, 0, 1, 2]]
    raise ValueError(f"Unsupported root quaternion order: {order}")


def _get_robot_asset_cfg(robot_name: str) -> ArticulationCfg:
    if robot_name == "g1":
        return G1_CYLINDER_CFG
    if robot_name == "g1_23":
        return G1_23DOF_CFG
    raise KeyError(f"Unknown robot_name '{robot_name}'. Supported: g1, g1_23dof")


def _build_scene_cfg(robot_cfg: ArticulationCfg) -> type[InteractiveSceneCfg]:
    @configclass
    class ReplayMotionSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionSceneCfg


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        root_quat_order: str,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.root_quat_order = root_quat_order
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = _root_quat_to_wxyz(motion[:, 3:7], self.root_quat_order)
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.clamp(index_0 + 1, max=self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag

def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    motion: MotionLoader,
    csv_joint_names: list[str],
    output_name: str,
    root_link_name: str,
    robot_name: str,
):
    robot = scene["robot"]
    if root_link_name not in robot.body_names:
        raise ValueError(f"root_link_name '{root_link_name}' not found in robot bodies: {robot.body_names}")

    robot_joint_indexes = robot.find_joints(csv_joint_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],

        "robot_name": [robot_name],
        "joint_names": np.array(robot.joint_names),
        "body_names": np.array(robot.body_names),
        "root_link_name": [root_link_name],

        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [], 
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[key] = np.stack(log[key], axis=0)
            np.savez(output_name, **log)
            print(f"[INFO]: Motion saved to: {output_name}")


def main():
    config_path = Path(args_cli.config).resolve()
    config = _load_yaml(config_path)
    motion_cfg = config["motion_data"]

    robot_name = str(motion_cfg["robot_name"])
    csv_path_value = args_cli.csv_path if args_cli.csv_path is not None else motion_cfg["csv_path"]
    if not isinstance(csv_path_value, str):
        raise TypeError(
            "csv_path must be a string. For multiple paths, use scripts/mimic/batch_csv_to_npz_frog.py."
        )
    csv_path = _resolve_path(config_path.parent, csv_path_value)
    input_fps = int(motion_cfg.get("csv_fps", 30))
    root_quat_order = str(motion_cfg.get("root_quat_order", "xyzw"))
    root_link_name = str(motion_cfg["root_link_name"])
    csv_joint_names = list(motion_cfg["csv_joint_names"])

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = _build_scene_cfg(_get_robot_asset_cfg(robot_name))
    scene = InteractiveScene(scene_cfg(num_envs=1, env_spacing=2.0))

    sim.reset()
    print("[INFO]: Setup complete...")

    motion = MotionLoader(
        motion_file=str(csv_path),
        input_fps=input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
        root_quat_order=root_quat_order,
    )
    
    run_simulator(sim, scene, motion, csv_joint_names, args_cli.output_name, root_link_name, robot_name)


if __name__ == "__main__":
    main()
    simulation_app.close()
