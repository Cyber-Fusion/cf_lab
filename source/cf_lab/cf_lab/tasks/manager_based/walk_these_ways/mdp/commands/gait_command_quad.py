# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for Walk These Ways 9D behavior vector (quadruped)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg_quad import UniformGaitCommandCfgQuad

# Canonical quadruped gaits: (off2/RF, off3/LH, off4/RH)
CANONICAL_GAITS = torch.tensor([
    [0.5, 0.5, 0.0],  # Trot
    [0.5, 0.0, 0.5],  # Pace
    [0.0, 0.5, 0.5],  # Bound
    [0.0, 0.0, 0.0],  # Pronk
])


class GaitCommandQuad(CommandTerm):
    """Command generator for the WTW 9D behavior vector.

    Command layout (9D):
        [0] frequency    - gait stepping frequency [Hz]
        [1] duration     - stance duty cycle [0-1]
        [2] off2         - RF foot phase offset
        [3] off3         - LH foot phase offset
        [4] off4         - RH foot phase offset
        [5] feet_height  - foot swing height command [m]
        [6] base_height  - body height command [m]
        [7] body_pitch   - body pitch command [rad]
        [8] body_roll    - body roll command [rad]
    """

    cfg: UniformGaitCommandCfgQuad

    def __init__(self, cfg: UniformGaitCommandCfgQuad, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.gait_command = torch.zeros(self.num_envs, 9, device=self.device)
        self._canonical_gaits = CANONICAL_GAITS.to(self.device)
        self.metrics = {}

    def __str__(self) -> str:
        msg = "GaitCommandQuad:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tMulti-gait: {self.cfg.multi_gait}, Binary phases: {self.cfg.binary_phases}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The 9D gait command. Shape is (num_envs, 9)."""
        return self.gait_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        n = len(env_ids)
        if n == 0:
            return

        r = torch.empty(n, device=self.device)

        # Frequency
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        # Duration (duty cycle)
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.durations)

        # Phase offsets
        if self.cfg.multi_gait:
            gait_idx = torch.randint(0, len(self._canonical_gaits), (n,), device=self.device)
            offsets = self._canonical_gaits[gait_idx].clone()

            if not self.cfg.binary_phases:
                jitter = (torch.rand(n, 3, device=self.device) - 0.5) * 2 * self.cfg.gait_phase_jitter
                offsets = torch.remainder(offsets + jitter, 1.0)

            self.gait_command[env_ids, 2] = offsets[:, 0]  # off2 (RF)
            self.gait_command[env_ids, 3] = offsets[:, 1]  # off3 (LH)
            self.gait_command[env_ids, 4] = offsets[:, 2]  # off4 (RH)
        else:
            self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.offsets2)
            self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.offsets3)
            self.gait_command[env_ids, 4] = r.uniform_(*self.cfg.ranges.offsets4)

        # Remaining parameters
        self.gait_command[env_ids, 5] = r.uniform_(*self.cfg.ranges.feet_height)
        self.gait_command[env_ids, 6] = r.uniform_(*self.cfg.ranges.base_height)
        self.gait_command[env_ids, 7] = r.uniform_(*self.cfg.ranges.body_pitch)
        self.gait_command[env_ids, 8] = r.uniform_(*self.cfg.ranges.body_roll)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
