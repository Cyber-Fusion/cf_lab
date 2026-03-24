# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    lin_vel_step: float = 0.5,
    ang_vel_step: float = 0.5,
    max_lin_vel_x: float = 3.0,
    max_lin_vel_y: float = 1.0,
    max_ang_vel_z: float = 5.0,
    reward_threshold_lin: float = 0.8,
    reward_threshold_ang: float = 0.7,
) -> float:
    """Velocity command curriculum.

    Progressively widens the velocity command ranges as the agent's velocity tracking
    performance improves. Linear and angular velocity ranges are expanded independently
    when the corresponding episodic tracking reward exceeds the threshold.

    The curriculum checks the average episodic reward for the resetting environments
    (normalized by episode length) and expands the range by one bin step if above threshold.

    Returns:
        The current max linear velocity (for logging).
    """
    if len(env_ids) == 0:
        return 0.0

    cmd_term = env.command_manager.get_term(command_name)

    # Get per-env episodic reward for velocity tracking terms
    # (episode_sums are still available here — curriculum runs before reward_manager.reset)
    ep_len = env.episode_length_buf[env_ids].float().clamp(min=1.0) * env.step_dt

    lin_sum = env.reward_manager._episode_sums.get("track_lin_vel_xy_exp", None)
    ang_sum = env.reward_manager._episode_sums.get("track_ang_vel_z_exp", None)

    if lin_sum is not None:
        avg_lin_reward = (lin_sum[env_ids] / ep_len).mean().item()
    else:
        avg_lin_reward = 0.0

    if ang_sum is not None:
        avg_ang_reward = (ang_sum[env_ids] / ep_len).mean().item()
    else:
        avg_ang_reward = 0.0

    # Expand linear velocity range
    if avg_lin_reward > reward_threshold_lin:
        cur_max_x = cmd_term.cfg.ranges.lin_vel_x[1]
        new_max_x = min(cur_max_x + lin_vel_step, max_lin_vel_x)
        cmd_term.cfg.ranges.lin_vel_x = (-new_max_x, new_max_x)
        cur_max_y = cmd_term.cfg.ranges.lin_vel_y[1]
        new_max_y = min(cur_max_y + lin_vel_step, max_lin_vel_y)
        cmd_term.cfg.ranges.lin_vel_y = (-new_max_y, new_max_y)

    # Expand angular velocity range
    if avg_ang_reward > reward_threshold_ang:
        cur_max_z = cmd_term.cfg.ranges.ang_vel_z[1]
        new_max_z = min(cur_max_z + ang_vel_step, max_ang_vel_z)
        cmd_term.cfg.ranges.ang_vel_z = (-new_max_z, new_max_z)

    return cmd_term.cfg.ranges.lin_vel_x[1]


def anneal_sigma_exp_neg(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    sigma_min: float = 1.0,
    sigma_max: float = 20.0,
    anneal_steps: int = 24000,
) -> float:
    """Quadratically anneal the exp-negative sigma coefficient.

    sigma = sigma_min + (sigma_max - sigma_min) * min((step/anneal_steps)^2, 1.0)

    Early training: sigma is low, exp gate ≈ 1, policy focuses on velocity tracking.
    Late training: sigma is high, exp gate suppresses reward when behavior is poor.

    Args:
        env: The learning environment.
        env_ids: Not used directly, but required by curriculum interface.
        sigma_min: Starting sigma value.
        sigma_max: Final sigma value.
        anneal_steps: Number of env steps over which to anneal.

    Returns:
        The current sigma value (for logging).
    """
    progress = min(env.common_step_counter / anneal_steps, 1.0)
    new_val = sigma_min + (sigma_max - sigma_min) * (progress ** 2)
    env.reward_manager.sigma = new_val
    return new_val
