# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum logging terms for the AYG velocity task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_fwd_pool_mean(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Mean terrain level over the forward pool (logging-only)."""
    terrain: TerrainImporter = env.scene.terrain
    forward_mask = env.command_manager.get_term(command_name).forward_mask
    levels = terrain.terrain_levels[forward_mask].float()
    return levels.mean() if levels.numel() > 0 else torch.zeros((), device=terrain.device)


def terrain_levels_nonfwd_pool_mean(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Mean terrain level over the non-forward pool (logging-only)."""
    terrain: TerrainImporter = env.scene.terrain
    forward_mask = env.command_manager.get_term(command_name).forward_mask
    levels = terrain.terrain_levels[~forward_mask].float()
    return levels.mean() if levels.numel() > 0 else torch.zeros((), device=terrain.device)
