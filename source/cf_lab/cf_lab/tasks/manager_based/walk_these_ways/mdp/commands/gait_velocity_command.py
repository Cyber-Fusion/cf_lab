# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-gait velocity command with a vx-only curriculum."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from .scaling import vx_scale

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .gait_velocity_command_cfg import MultiGaitVelocityCommandCfg


class MultiGaitVelocityCommand(UniformVelocityCommand):
    """Velocity command that samples from a per-gait range.

    Reads the canonical gait assignment for each env from a sibling :class:`GaitCommandQuad`
    term (via its ``gait_ids`` tensor) and samples ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z``
    and ``heading`` from the matching :class:`UniformVelocityCommandCfg.Ranges`.

    A linear curriculum (driven externally via :attr:`curriculum_progress`) ramps the
    ``lin_vel_x`` bound from ``(-initial_max_lin_vel_x, +initial_max_lin_vel_x)`` at progress 0
    to the per-gait range at progress 1. ``lin_vel_y`` and ``ang_vel_z`` are sampled directly
    from their (small) per-gait ranges from step 0 without any curriculum ramp.

    When :attr:`MultiGaitVelocityCommandCfg.couple_to_vx` is True, ``vy`` and ``ang_vel_z`` are
    multiplied by ``vx_scale(vx)`` so the policy is not asked to track aggressive lateral / yaw
    objectives while running fast forward.
    """

    cfg: MultiGaitVelocityCommandCfg

    def __init__(self, cfg: MultiGaitVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # progress in [0, 1] driven by the curriculum term; resample sees the latest value
        self.curriculum_progress: float = 0.0

        # Stack per-gait ranges into device tensors of shape (num_gaits, 2) for vectorised lookup
        ranges_pg = cfg.ranges_per_gait
        self._range_lin_vel_x = torch.tensor(
            [r.lin_vel_x for r in ranges_pg], dtype=torch.float, device=self.device
        )
        self._range_lin_vel_y = torch.tensor(
            [r.lin_vel_y for r in ranges_pg], dtype=torch.float, device=self.device
        )
        self._range_ang_vel_z = torch.tensor(
            [r.ang_vel_z for r in ranges_pg], dtype=torch.float, device=self.device
        )
        # heading range is optional on the parent dataclass (only used when heading_command=True)
        heading_rows = [
            r.heading if r.heading is not None else (-torch.pi, torch.pi) for r in ranges_pg
        ]
        self._range_heading = torch.tensor(heading_rows, dtype=torch.float, device=self.device)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # Pull the gait assignment for each resampling env from the sibling gait term.
        gait_term = self._env.command_manager.get_term(self.cfg.gait_command_name)
        gait_ids = gait_term.gait_ids[env_ids]  # (n,)
        n = gait_ids.numel()
        p = float(self.curriculum_progress)

        # vx: curriculum-blended bounds (ramp from (-initial, +initial) up to per-gait final)
        vx_final = self._range_lin_vel_x[gait_ids]  # (n, 2)
        vx_min = (1.0 - p) * (-self.cfg.initial_max_lin_vel_x) + p * vx_final[:, 0]
        vx_max = (1.0 - p) * self.cfg.initial_max_lin_vel_x + p * vx_final[:, 1]
        self.vel_command_b[env_ids, 0] = vx_min + torch.rand(n, device=self.device) * (vx_max - vx_min)

        # vy, omega: direct per-gait sampling (no curriculum, ranges are already kept small)
        for col, per_gait in ((1, self._range_lin_vel_y), (2, self._range_ang_vel_z)):
            final = per_gait[gait_ids]  # (n, 2)
            self.vel_command_b[env_ids, col] = final[:, 0] + torch.rand(n, device=self.device) * (
                final[:, 1] - final[:, 0]
            )

        # heading: sample uniformly per env from the per-gait heading range; no curriculum on heading
        if self.cfg.heading_command:
            heading_per_env = self._range_heading[gait_ids]  # (n, 2)
            u = torch.rand(n, device=self.device)
            self.heading_target[env_ids] = heading_per_env[:, 0] + u * (
                heading_per_env[:, 1] - heading_per_env[:, 0]
            )
            r = torch.empty(n, device=self.device)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # standing envs (same per-env Bernoulli as parent)
        r = torch.empty(n, device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        # Parent applies heading-control on vel_command_b[:, 2] for heading envs (clamped to
        # cfg.ranges.ang_vel_z) and zeroes vel_command_b for standing envs.
        super()._update_command()

        # Uniform vx-based shrink on the secondary commands. Below |vx|=1 nothing changes; above,
        # vy and omega decay as 1/(|vx|/knee). The matching scaling on pitch / roll / base-height
        # lives in the sibling GaitCommandQuad term.
        if self.cfg.couple_to_vx:
            scale = vx_scale(self.vel_command_b[:, 0])
            self.vel_command_b[:, 1] *= scale  # vy
            self.vel_command_b[:, 2] *= scale  # omega
