"""Play and export AMP/WASABI checkpoints with frog_rl."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

try:
    from . import cli_args
except ImportError:  # pragma: no cover - supports direct script execution
    import cli_args

parser = argparse.ArgumentParser(description="Play an agent with frog_rl.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--real-time", action="store_true", default=False)
control_group = parser.add_mutually_exclusive_group()
control_group.add_argument("--keyboard", action="store_true", help="Use the keyboard to control velocity commands.")
control_group.add_argument("--joy", action="store_true", help="Use a gamepad to control velocity commands.")
cli_args.add_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from frog_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg, Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from camera_follow import CameraFollower

import isaaclab_tasks  # noqa: F401
import frog_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    camera_follower = CameraFollower() if args_cli.keyboard or args_cli.joy else None
    agent_cfg = cli_args.update_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.keyboard or args_cli.joy:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False

        controller_cfg_cls = Se2KeyboardCfg if args_cli.keyboard else Se2GamepadCfg
        controller_cls = Se2Keyboard if args_cli.keyboard else Se2Gamepad
        controller = controller_cls(
            controller_cfg_cls(
                v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
            )
        )
        print(f"[INFO] {controller}")
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device),
        )

    log_root_path = os.path.abspath(os.path.join("logs", "frog_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(resume_path), "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_type = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}.get(agent_cfg.class_name)
    if runner_type is None:
        raise ValueError(f"Unsupported frog_rl runner class: {agent_cfg.class_name}")
    runner = runner_type(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(export_dir, "policy.pt")
    runner.export_policy_to_onnx(export_dir, "policy.onnx")

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        if camera_follower is not None:
            if torch.any(dones).item():
                camera_follower.reset()
            camera_follower.update(env)
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        sleep_time = dt - (time.time() - start)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
