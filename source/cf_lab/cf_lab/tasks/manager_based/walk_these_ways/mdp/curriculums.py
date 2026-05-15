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


def anneal_sigma_exp_neg(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    sigma_min: float = 1.0,
    sigma_max: float = 20.0,
    anneal_steps: int = 1000,
) -> float:
    """Quadratically anneal the exp-negative sigma coefficient.

    sigma = sigma_min + (sigma_max - sigma_min) * min((step/anneal_steps)^2, 1.0)

    Early training: sigma is low, exp gate ~ 1, policy focuses on velocity tracking.
    Late training: sigma is high, exp gate suppresses reward when behavior is poor.

    Returns:
        The current sigma value (for logging).
    """
    progress = min(env.common_step_counter / (anneal_steps), 1.0)
    new_val = sigma_min + (sigma_max - sigma_min) * (progress**2)
    env.reward_manager.sigma = new_val
    return new_val


def per_gait_progress_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    gait_id: int,
    progress_min: float = 0.0,
    progress_max: float = 1.0,
    max_fall_rate: float = 0.10,
    min_track_quality: float = 0.5,
    step_size: float = 0.0125,
    min_steps_between_updates: int = 24,
    ema_alpha: float = 0.05,
    track_term_name: str = "track_lin_vel_xy_exp",
    term_name: str = "gait",
) -> float:
    """Monotonic, rate-limited per-gait progress curriculum gated on fall rate AND tracking quality.

    Progress is a unitless fraction in [0, 1] stored in
    ``GaitRewardQuad.per_gait_progress``. The gait reward computes its per-env scale
    as ``gait_full_magnitude * progress``, so progress=0 means "no gait pressure"
    and progress=1 means "full gait pressure". ``progress_min`` > 0 seeds some
    pressure from the start.

    Progress is moved one ``step_size`` toward ``progress_max`` whenever both
    conditions are met for the resetting envs running this gait (smoothed via EMA):
      - failure-induced reset rate <= ``max_fall_rate`` (time-out resets are not failures)
      - mean linear-velocity tracking reward (raw, pre-weight) >= ``min_track_quality``

    The curriculum is **monotonic**: once progress advances toward ``progress_max``,
    it never regresses. The motivation is to stop the back-and-forth oscillation
    seen with bidirectional updates.

    Per-gait state (EMAs, progress value, last-update step) lives on the gait
    reward instance. The function is invoked by Isaac Lab's CurriculumManager on
    every env step where any env resets; updates are rate-limited to at most one
    (+step_size) per ``min_steps_between_updates`` env steps per gait. With
    rsl_rl ``num_steps_per_env=24``, the default of 24 env steps == 1 PPO iteration,
    so progress changes by at most ``step_size`` per iteration.

    Tracking-quality measurement uses the episode-mean of the raw
    ``track_term_name`` reward function value (i.e. ``_episode_sums[track_term_name]
    / (episode_length * weight * dt)``). For ``base_linear_velocity_reward`` this
    saturates at 1.0 at low commanded speeds, so ``min_track_quality=0.5`` means
    "raw reward >= 0.5 on average across the episode".

    Args:
        gait_id: 0=trot, 1=pace, 2=bound, 3=pronk (matches GaitCommandQuad.CANONICAL_GAITS).
        progress_min: starting progress fraction; assigned at first call.
        progress_max: upper clamp on progress; progress is monotonically moved toward it.
        max_fall_rate: gate threshold for the failure-reset EMA.
        min_track_quality: gate threshold for the episode-mean tracking-reward EMA.
        step_size: progress increment per allowed update.
        min_steps_between_updates: minimum env steps between consecutive updates per gait.
        ema_alpha: per-sample EMA decay for the gating signals.
        track_term_name: name of the linear-velocity tracking reward term.
        term_name: name of the gait reward term carrying ``per_gait_progress``.

    Returns:
        The current progress for this gait (for logging).
    """
    rm = env.reward_manager
    reward_inst = rm.get_term_cfg(term_name).func  # GaitRewardQuad instance

    # First-call initialization: seed all four slots to progress_min and set up EMA state.
    # `_per_gait_fall_rate_ema` starts at 1.0 (assume failure) and `_per_gait_track_quality_ema`
    # at 0.0 so the gate cannot fire until enough samples have accumulated.
    if not getattr(reward_inst, "_per_gait_curriculum_initialized", False):
        reward_inst.per_gait_progress[:] = progress_min
        reward_inst._per_gait_curriculum_initialized = True
        reward_inst._per_gait_last_update_step = [-min_steps_between_updates] * 4
        reward_inst._per_gait_fall_rate_ema = [1.0] * 4
        reward_inst._per_gait_track_quality_ema = [0.0] * 4

    if len(env_ids) == 0:
        return float(reward_inst.per_gait_progress[gait_id].item())

    gait_cmd = env.command_manager.get_term("gait_command")
    env_ids_t = torch.as_tensor(env_ids, device=gait_cmd.current_gait_ids.device, dtype=torch.long)
    mask = gait_cmd.current_gait_ids[env_ids_t] == gait_id
    if not mask.any():
        return float(reward_inst.per_gait_progress[gait_id].item())
    matching = env_ids_t[mask]
    n = int(matching.numel())

    # ---- EMA updates from this batch of resetting envs ----

    # Failure rate: `terminated` is True for non-time-out terminations (falls / crashes).
    terminated = env.termination_manager.terminated[matching]
    batch_fall_rate = float(terminated.float().mean().item())

    # Tracking quality: episode-mean raw reward value of the linear-velocity tracking term.
    # _episode_sums[name] = sum_t(func_t * weight * dt). Mean per-step func = sum / (steps * weight * dt).
    track_term_cfg = rm.get_term_cfg(track_term_name)
    track_weight = float(track_term_cfg.weight)
    if track_weight == 0.0:
        # Tracking term disabled — skip the gate entirely (will never advance).
        return float(reward_inst.per_gait_progress[gait_id].item())
    dt = env.step_dt
    ep_len = env.episode_length_buf[matching].float().clamp(min=1.0)
    track_sum = rm._episode_sums[track_term_name][matching]
    track_quality = (track_sum / (ep_len * track_weight * dt)).clamp(min=0.0)
    batch_track_quality = float(track_quality.mean().item())

    # Per-sample-equivalent decay so a batch of `n` samples decays the EMA like n independent updates.
    alpha_n = 1.0 - (1.0 - ema_alpha) ** n
    reward_inst._per_gait_fall_rate_ema[gait_id] = (
        (1.0 - alpha_n) * reward_inst._per_gait_fall_rate_ema[gait_id] + alpha_n * batch_fall_rate
    )
    reward_inst._per_gait_track_quality_ema[gait_id] = (
        (1.0 - alpha_n) * reward_inst._per_gait_track_quality_ema[gait_id] + alpha_n * batch_track_quality
    )

    # ---- Rate-limit + monotonic update ----
    cur_step = env.common_step_counter
    if cur_step - reward_inst._per_gait_last_update_step[gait_id] < min_steps_between_updates:
        return float(reward_inst.per_gait_progress[gait_id].item())

    fall_rate_ema = reward_inst._per_gait_fall_rate_ema[gait_id]
    track_quality_ema = reward_inst._per_gait_track_quality_ema[gait_id]
    if not (fall_rate_ema <= max_fall_rate and track_quality_ema >= min_track_quality):
        return float(reward_inst.per_gait_progress[gait_id].item())

    cur = float(reward_inst.per_gait_progress[gait_id].item())
    new_progress = min(cur + abs(step_size), progress_max)
    reward_inst.per_gait_progress[gait_id] = new_progress
    reward_inst._per_gait_last_update_step[gait_id] = cur_step

    return float(reward_inst.per_gait_progress[gait_id].item())
