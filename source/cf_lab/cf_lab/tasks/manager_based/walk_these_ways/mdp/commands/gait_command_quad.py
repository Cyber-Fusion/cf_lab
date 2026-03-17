# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for Walk These Ways 8D behavior vector (quadruped)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg_quad import UniformGaitCommandCfgQuad

# Canonical quadruped gaits: (theta1, theta2, theta3)
# Canonical quadruped gaits: trot=(0.5,0,0), pronk=(0,0,0), bound=(0,0.5,0), pace=(0,0,0.5)
CANONICAL_GAITS = torch.tensor([
    [0.5, 0.0, 0.0],  # Trot
    [0.0, 0.0, 0.0],  # Pronk
    [0.0, 0.5, 0.0],  # Bound
    [0.0, 0.0, 0.5],  # Pace
])


class GaitCommandQuad(CommandTerm):
    """Command generator for the WTW 8D behavior vector.

    Command layout (8D):
        [0] theta1      - phase offset parameter 1
        [1] theta2      - phase offset parameter 2
        [2] theta3      - phase offset parameter 3
        [3] frequency   - gait stepping frequency [Hz]
        [4] base_height - body height command [m]
        [5] body_pitch  - body pitch command [rad]
        [6] stance_width - foot stance width command [m]
        [7] footswing_height - footswing height command [m]
    """

    cfg: UniformGaitCommandCfgQuad

    def __init__(self, cfg: UniformGaitCommandCfgQuad, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.gait_command = torch.zeros(self.num_envs, 8, device=self.device)
        # Move canonical gaits to device
        self._canonical_gaits = CANONICAL_GAITS.to(self.device)
        self.metrics = {}

    def __str__(self) -> str:
        msg = "GaitCommandQuad:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tCanonical gait probability: {self.cfg.canonical_gait_probability}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The 8D gait command. Shape is (num_envs, 8)."""
        return self.gait_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments.

        Phase offsets (theta1/2/3) are sampled either from a Gaussian centered on
        a canonical gait or uniformly, controlled by canonical_gait_probability.
        Remaining parameters are sampled uniformly from their ranges.
        """
        n = len(env_ids)
        if n == 0:
            return

        # --- Sample theta1, theta2, theta3 ---
        use_canonical = torch.rand(n, device=self.device) < self.cfg.canonical_gait_probability

        # Canonical path: pick a random gait center, add Gaussian noise, wrap to [0, 1)
        gait_idx = torch.randint(0, len(self._canonical_gaits), (n,), device=self.device)
        centers = self._canonical_gaits[gait_idx]  # (n, 3)
        noise = torch.randn(n, 3, device=self.device) * self.cfg.canonical_gait_std
        canonical_theta = torch.remainder(centers + noise, 1.0)

        # Uniform path: sample uniformly from configured ranges
        uniform_theta = torch.zeros(n, 3, device=self.device)
        uniform_theta[:, 0].uniform_(*self.cfg.ranges.theta1)
        uniform_theta[:, 1].uniform_(*self.cfg.ranges.theta2)
        uniform_theta[:, 2].uniform_(*self.cfg.ranges.theta3)

        # Select based on probability
        use_canonical_3d = use_canonical.unsqueeze(1).expand(-1, 3)
        theta = torch.where(use_canonical_3d, canonical_theta, uniform_theta)

        self.gait_command[env_ids, 0] = theta[:, 0]
        self.gait_command[env_ids, 1] = theta[:, 1]
        self.gait_command[env_ids, 2] = theta[:, 2]

        # --- Sample remaining parameters uniformly ---
        r = torch.empty(n, device=self.device)
        self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.frequency)
        self.gait_command[env_ids, 4] = r.uniform_(*self.cfg.ranges.base_height)
        self.gait_command[env_ids, 5] = r.uniform_(*self.cfg.ranges.body_pitch)
        self.gait_command[env_ids, 6] = r.uniform_(*self.cfg.ranges.stance_width)
        self.gait_command[env_ids, 7] = r.uniform_(*self.cfg.ranges.footswing_height)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
