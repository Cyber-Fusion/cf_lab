"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg_quad import UniformGaitCommandCfgQuad


class GaitCommandQuad(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: UniformGaitCommandCfgQuad
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformGaitCommandCfgQuad, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        # command format: [freq, duration, offset2, offset3, offset4, feet_h, base_h, pitch, roll]
        self.gait_command = torch.zeros(self.num_envs, 9, device=self.device)
        # incremental gait phase state
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        self.dt = env.step_dt
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def reset(self, env_ids=None):
        if env_ids is None:
            self.gait_indices[:] = 0.0
        else:
            self.gait_indices[env_ids] = 0.0
        return super().reset(env_ids)

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 9)."""
        return self.gait_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        # -- contact duration
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.durations)
        # -- phase offset2
        self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.offsets2)
        # -- phase offset3
        self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.offsets3)
        # -- phase offset4
        self.gait_command[env_ids, 4] = r.uniform_(*self.cfg.ranges.offsets4)
        # -- feet height
        self.gait_command[env_ids, 5] = r.uniform_(*self.cfg.ranges.feet_height)
        # -- base height
        self.gait_command[env_ids, 6] = r.uniform_(*self.cfg.ranges.base_height)
        # -- body pitch
        self.gait_command[env_ids, 7] = r.uniform_(*self.cfg.ranges.body_pitch)
        # -- body roll
        self.gait_command[env_ids, 8] = r.uniform_(*self.cfg.ranges.body_roll)


    def _update_command(self):
        """Incrementally advance gait phase and compute per-foot indices."""
        frequencies = self.gait_command[:, 0]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        offsets2 = self.gait_command[:, 2]
        offsets3 = self.gait_command[:, 3]
        offsets4 = self.gait_command[:, 4]
        self.foot_indices = torch.remainder(
            torch.stack([
                self.gait_indices,
                self.gait_indices + offsets2 + 1,
                self.gait_indices + offsets3 + 1,
                self.gait_indices + offsets4 + 1,
            ], dim=1),
            1.0,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
