# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command terms for the AYG velocity task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DualPoolUniformVelocityCommand(UniformVelocityCommand):
    """Velocity command that splits envs into a forward pool and a non-forward pool.

    The forward pool gets `lin_vel_x >= 0`; the non-forward pool gets `lin_vel_x <= 0`.
    `lin_vel_y` and `ang_vel_z` are sampled symmetrically for both pools, so the
    non-forward pool covers backward, lateral, and yaw motion (and any combination).
    """

    cfg: "DualPoolUniformVelocityCommandCfg"

    def __init__(self, cfg: "DualPoolUniformVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        num_fwd = int(self.num_envs * cfg.forward_env_fraction)
        self.forward_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.forward_mask[:num_fwd] = True

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        fwd_local = self.forward_mask[env_ids_t]
        fwd_ids = env_ids_t[fwd_local]
        nonfwd_ids = env_ids_t[~fwd_local]
        self.vel_command_b[fwd_ids, 0] = self.vel_command_b[fwd_ids, 0].abs()
        self.vel_command_b[nonfwd_ids, 0] = -self.vel_command_b[nonfwd_ids, 0].abs()


@configclass
class DualPoolUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`DualPoolUniformVelocityCommand`."""

    class_type: type = DualPoolUniformVelocityCommand

    forward_env_fraction: float = 0.5
    """Fraction of envs (from the lowest indices) assigned to the forward pool. Defaults to 0.5."""
