"""Camera utilities for interactive Isaac Lab playback."""

from __future__ import annotations

from collections import deque

import torch

import isaaclab.utils.math as math_utils


class CameraFollower:
    """Smoothly follow the first robot in the first environment."""

    def __init__(self, window_size: int = 50):
        self._camera_positions: deque[torch.Tensor] = deque(maxlen=window_size)
        self._initialized_env = None

    def reset(self) -> None:
        """Clear smoothing state after an environment reset."""
        self._camera_positions.clear()

    def update(self, env) -> None:
        """Update the viewport camera using the robot's current world pose."""
        unwrapped_env = env.unwrapped
        camera_controller = getattr(unwrapped_env, "viewport_camera_controller", None)
        if camera_controller is None:
            return

        robot = unwrapped_env.scene["robot"]
        robot_pos = robot.data.root_pos_w[0]
        robot_quat = robot.data.root_quat_w[0]
        camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=robot_pos.dtype, device=robot_pos.device)
        camera_pos = math_utils.transform_points(
            camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
        ).squeeze(0)

        env_origin = unwrapped_env.scene.env_origins[0].to(device=robot_pos.device, dtype=robot_pos.dtype)
        relative_camera_pos = camera_pos - env_origin
        relative_robot_pos = robot_pos - env_origin
        self._camera_positions.append(relative_camera_pos)
        smooth_camera_pos = torch.stack(tuple(self._camera_positions)).mean(dim=0)

        if self._initialized_env != id(unwrapped_env):
            camera_controller.set_view_env_index(env_index=0)
            camera_controller.update_view_to_env()
            self._initialized_env = id(unwrapped_env)

        camera_controller.update_view_location(
            eye=smooth_camera_pos.detach().cpu().numpy(),
            lookat=relative_robot_pos.detach().cpu().numpy(),
        )
