from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import frog_lab.tasks.locomotion.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DiscreteCommand(CommandTerm):
    """Command generator that assigns a discrete integer command to each environment.

    The command is sampled uniformly from a user-specified list of integers
    (e.g. ``[1, 2, 3]``). Each environment receives one of these values, which is
    used as the target command until the command is resampled.

    This is analogous to :class:`UniformVelocityCommand`, except that the command
    is a single discrete value drawn from :attr:`cfg.available_commands` instead of
    a continuous velocity in SE(2). The command tensor has shape ``(num_envs, 1)``.
    """

    cfg: DiscreteCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: DiscreteCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If :attr:`cfg.available_commands` is empty or contains
                non-integer values.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # sanity checks on the configuration
        if not self.cfg.available_commands:
            raise ValueError("The 'available_commands' list cannot be empty.")
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements of 'available_commands' must be integers.")

        # store the available commands as a tensor for efficient indexing
        self._available_commands = torch.tensor(
            self.cfg.available_commands, dtype=torch.int32, device=self.device
        )

        # create buffer to store the command: shape (num_envs, 1)
        self.command_b = torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        return (
            "DiscreteCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tAvailable commands: {self.cfg.available_commands}"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired discrete command. Shape is (num_envs, 1)."""
        return self.command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """No metrics are tracked for the discrete command."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new command uniformly from :attr:`cfg.available_commands`."""
        indices = torch.randint(len(self._available_commands), (len(env_ids),), device=self.device)
        self.command_b[env_ids] = self._available_commands[indices].unsqueeze(1)

    def _update_command(self):
        """The command does not require any post-processing."""
        pass


@configclass
class DiscreteCommandCfg(CommandTermCfg):
    """Configuration for the discrete command generator."""

    class_type: type = DiscreteCommand

    available_commands: list[int] = []
    """List of discrete integer values to sample from.

    Each environment is assigned one of these values as its command.
    Example: [1, 2, 3]
    """


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
