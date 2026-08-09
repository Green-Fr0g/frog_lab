#!/usr/bin/env python3
"""Replay G1 29-DOF AMP CSV motions in IsaacLab and save body-state NPZ assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
G1_29DOF_AMP_RAW_MOTION_DIR = REPO_ROOT / "motion_data" / "amp" / "g1"
G1_29DOF_AMP_MOTION_DIR = (
    REPO_ROOT / "source" / "frog_lab" / "frog_lab" / "assets" / "motions" / "g1" / "amp" / "locomotion"
)
G1_29DOF_AMP_ALL_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=None, help="Single raw G1 AMP CSV file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=G1_29DOF_AMP_RAW_MOTION_DIR,
        help="Directory containing raw G1 AMP CSV files.",
    )
    parser.add_argument("--output-name", type=str, default=None, help="Output NPZ filename for --input-file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=G1_29DOF_AMP_MOTION_DIR,
        help="Directory where converted NPZ files will be written.",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="1-based inclusive CSV frame range. If omitted, all frames are used.",
    )
    parser.add_argument("--input-fps", type=float, default=120.0, help="Raw CSV frame rate.")
    parser.add_argument("--output-fps", type=float, default=50.0, help="Converted NPZ frame rate.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files.")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not update the viewer camera during replay.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from frog_lab.assets.g1_29dof import G1_29DOF_CFG


@configclass
class ReplayG1AMPSceneCfg(InteractiveSceneCfg):
    """Minimal scene used to replay G1 AMP motions."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class CSVReplayMotion:
    """Load a root-plus-joint CSV and provide output-FPS replay states."""

    def __init__(
        self,
        *,
        motion_file: Path,
        input_fps: float,
        output_fps: float,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ) -> None:
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.frame_range = frame_range
        self.current_idx = 0
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        if self.frame_range is None:
            motion_np = np.loadtxt(self.motion_file, delimiter=",", dtype=np.float32)
        else:
            start, end = self.frame_range
            motion_np = np.loadtxt(
                self.motion_file,
                delimiter=",",
                dtype=np.float32,
                skiprows=start - 1,
                max_rows=end - start + 1,
            )
        if motion_np.ndim != 2 or motion_np.shape[1] != 36:
            raise ValueError(f"{self.motion_file} must have shape (frames, 36), got {motion_np.shape}")

        motion = torch.from_numpy(motion_np).to(device=self.device, dtype=torch.float32)
        self.root_pos_input = motion[:, 0:3]
        self.root_quat_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.joint_pos_input = motion[:, 7:36]
        self.input_frames = motion.shape[0]
        if self.input_frames < 2:
            raise ValueError(f"{self.motion_file} must contain at least two frames.")
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self) -> None:
        times = torch.arange(0.0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.root_pos = self._lerp(self.root_pos_input[index_0], self.root_pos_input[index_1], blend.unsqueeze(1))
        self.root_quat = self._slerp(self.root_quat_input[index_0], self.root_quat_input[index_1], blend)
        self.joint_pos = self._lerp(self.joint_pos_input[index_0], self.joint_pos_input[index_1], blend.unsqueeze(1))

    def _compute_velocities(self) -> None:
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.output_dt, dim=0)[0]
        self.joint_vel = torch.gradient(self.joint_pos, spacing=self.output_dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_quat, self.output_dt)

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for idx in range(a.shape[0]):
            out[idx] = quat_slerp(a[idx], b[idx], blend[idx])
        return out

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]:
        state = (
            self.root_pos[self.current_idx : self.current_idx + 1],
            self.root_quat[self.current_idx : self.current_idx + 1],
            self.root_lin_vel[self.current_idx : self.current_idx + 1],
            self.root_ang_vel[self.current_idx : self.current_idx + 1],
            self.joint_pos[self.current_idx : self.current_idx + 1],
            self.joint_vel[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset = self.current_idx >= self.output_frames
        if reset:
            self.current_idx = 0
        return state, reset


def resolve_file_pairs() -> list[tuple[Path, str]]:
    if args_cli.input_file is not None:
        output_name = args_cli.output_name or args_cli.input_file.with_suffix(".npz").name
        return [(args_cli.input_file, output_name)]

    csv_paths = sorted(args_cli.input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args_cli.input_dir}")
    return [(path, path.with_suffix(".npz").name) for path in csv_paths]


def replay_one_motion(
    *,
    sim: SimulationContext,
    scene: InteractiveScene,
    csv_path: Path,
    output_path: Path,
    joint_indices: torch.Tensor,
    body_indices: torch.Tensor,
) -> None:
    motion = CSVReplayMotion(
        motion_file=csv_path,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
    )
    robot = scene["robot"]
    scene.reset()

    log: dict[str, list[np.ndarray] | np.ndarray | str] = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    frame_count = 0
    while simulation_app.is_running():
        (
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            joint_pos_motion,
            joint_vel_motion,
        ), reset = motion.get_next_state()

        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = root_pos
        root_state[:, 0:2] += scene.env_origins[:, 0:2]
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_lin_vel
        root_state[:, 10:13] = root_ang_vel
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices] = joint_pos_motion
        joint_vel[:, joint_indices] = joint_vel_motion
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        if not args_cli.no_camera:
            pos_lookat = root_state[0, 0:3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, joint_indices].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, joint_indices].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, body_indices].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, body_indices].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, body_indices].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, body_indices].cpu().numpy().copy())

        frame_count += 1
        if reset:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fps=np.asarray(args_cli.output_fps, dtype=np.float32),
        body_names=np.asarray(G1_29DOF_AMP_ALL_BODY_NAMES),
        joint_names=np.asarray(G1_29DOF_JOINT_NAMES),
        joint_pos=np.stack(log["joint_pos"], axis=0).astype(np.float32),
        joint_vel=np.stack(log["joint_vel"], axis=0).astype(np.float32),
        body_pos_w=np.stack(log["body_pos_w"], axis=0).astype(np.float32),
        body_quat_w=np.stack(log["body_quat_w"], axis=0).astype(np.float32),
        body_lin_vel_w=np.stack(log["body_lin_vel_w"], axis=0).astype(np.float32),
        body_ang_vel_w=np.stack(log["body_ang_vel_w"], axis=0).astype(np.float32),
        source_file=str(csv_path),
        input_fps=np.asarray(args_cli.input_fps, dtype=np.float32),
        conversion="isaaclab_replay",
    )
    print(f"converted {csv_path.name} -> {output_path} ({frame_count} frames)")


def main() -> None:
    file_pairs = resolve_file_pairs()
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayG1AMPSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]

    joint_indices = robot.find_joints(list(G1_29DOF_JOINT_NAMES), preserve_order=True)[0]
    body_indices = robot.find_bodies(list(G1_29DOF_AMP_ALL_BODY_NAMES), preserve_order=True)[0]

    converted = 0
    skipped = 0
    for csv_path, output_name in file_pairs:
        output_path = args_cli.output_dir / output_name
        if output_path.exists() and not args_cli.overwrite:
            skipped += 1
            continue
        replay_one_motion(
            sim=sim,
            scene=scene,
            csv_path=csv_path,
            output_path=output_path,
            joint_indices=joint_indices,
            body_indices=body_indices,
        )
        converted += 1

    print(f"done: converted={converted}, skipped={skipped}, output_dir={args_cli.output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
