# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


# ---------------------------------------------------------------------------
# WTW command index constants (9D behavior vector)
# [0]=frequency, [1]=duration, [2]=off2(RF), [3]=off3(LH), [4]=off4(RH),
# [5]=feet_height, [6]=base_height, [7]=body_pitch, [8]=body_roll
# ---------------------------------------------------------------------------
IDX_FREQUENCY = 0
IDX_DURATION = 1
IDX_OFFSET2 = 2   # RF foot phase offset
IDX_OFFSET3 = 3   # LH foot phase offset
IDX_OFFSET4 = 4   # RH foot phase offset
IDX_FEET_HEIGHT = 5
IDX_BASE_HEIGHT = 6
IDX_BODY_PITCH = 7
IDX_BODY_ROLL = 8


def compute_per_foot_timings(
    off2: torch.Tensor, off3: torch.Tensor, off4: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-foot timing from direct offset parameterization.

    LF is the reference foot (offset = 0).
    off2 = RF offset, off3 = LH offset, off4 = RH offset.

    Returns:
        (t_LF, t_RF, t_LH, t_RH) each shape (num_envs,), wrapped to [0, 1).
    """
    t_LF = torch.remainder(t, 1.0)
    t_RF = torch.remainder(t + off2, 1.0)
    t_LH = torch.remainder(t + off3, 1.0)
    t_RH = torch.remainder(t + off4, 1.0)
    return t_LF, t_RF, t_LH, t_RH


def periodic_contact_schedule(foot_timings: torch.Tensor, kappa: float) -> torch.Tensor:
    """Compute desired contact states using a periodic smooth square wave.

    Uses tanh(sin(2*pi*t) / sigma) which is naturally periodic (no remainder
    discontinuity) and approximates a square wave with 50% duty cycle.
    Stance for t in [0, 0.5), swing for t in [0.5, 1.0).

    Args:
        foot_timings: Per-foot phase values, shape (N, num_feet).
        kappa: Sharpness parameter (smaller = sharper transitions).
               Maps to sigma = kappa * 2 * pi for comparable transition width
               to the Normal CDF formulation.

    Returns:
        Desired contact states near 1.0 for stance, near 0.0 for swing. Shape (N, num_feet).
    """
    sigma = kappa * 2 * math.pi
    return 0.5 * (1.0 + torch.tanh(torch.sin(2.0 * math.pi * foot_timings) / sigma))


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding."""
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


class GaitRewardQuad(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.env = env

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward: swing phase force tracking
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward: stance phase velocity tracking
        foot_velocities = torch.norm(
            self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2], dim=-1
        )
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combined as a single reward; weight applied by reward manager.
        total_reward = force_reward + velocity_reward

        cmd_not_null = env.command_manager.get_command("base_velocity").norm(p=1, dim=1) > 0.05

        total_reward = total_reward * cmd_not_null
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states using a periodic smooth square wave.

        Per-foot timing is computed from direct offset parameterization, then the contact
        schedule uses a periodic tanh-sin formulation that avoids the remainder
        discontinuity of the Normal CDF approximation.
        Returns values near 1.0 for stance and near 0.0 for swing.
        """
        off2 = gait_params[:, IDX_OFFSET2]
        off3 = gait_params[:, IDX_OFFSET3]
        off4 = gait_params[:, IDX_OFFSET4]
        frequency = gait_params[:, IDX_FREQUENCY]

        # Global timing variable: advances by f^cmd / f_pi each control step
        t = torch.remainder(self._env.episode_length_buf * self.dt * frequency, 1.0)

        # Per-foot timing (AYG URDF order: LF, RF, LH, RH)
        t_LF, t_RF, t_LH, t_RH = compute_per_foot_timings(off2, off3, off4, t)
        foot_timings = torch.stack([t_LF, t_RF, t_LH, t_RH], dim=1)  # (N, 4)

        # Periodic contact schedule (no remainder discontinuity)
        desired_contact_states = periodic_contact_schedule(foot_timings, self.kappa_gait_probs)

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute swing phase force tracking.

        Σ_foot (1 - C_foot) * exp(-|f^foot|² / σ_cf)
        """
        swing_mask = 1 - desired_contacts  # (N, num_feet)
        force_exp = torch.exp(-forces ** 2 / self.force_sigma)  # (N, num_feet)
        return torch.sum(swing_mask * force_exp, dim=1)

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute stance phase velocity tracking.

        Σ_foot C_foot * exp(-|v^foot|² / σ_cv)
        """
        vel_exp = torch.exp(-velocities ** 2 / self.vel_sigma)  # (N, num_feet)
        return torch.sum(desired_contacts * vel_exp, dim=1)


class FootSwingHeightQuad(GaitRewardQuad):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.height_scanner_cfg = cfg.params.get("height_scanner_cfg", None)

    def _get_ground_height(self) -> torch.Tensor:
        """Get the mean ground height beneath the robot from height scanner.

        Returns:
            Ground height per env, shape (N,). Returns 0 if no scanner.
        """
        if self.height_scanner_cfg is not None:
            height_scanner: RayCaster = self.env.scene[self.height_scanner_cfg.name]
            return torch.mean(height_scanner.data.ray_hits_w[..., 2], dim=1)
        return torch.zeros(self.num_envs, device=self.asset.device)

    def compute_footswing_height(self, desired_contacts):
        commands = self.env.command_manager.get_command("gait_command")
        cmd_height = commands[:, IDX_FEET_HEIGHT].unsqueeze(1)  # (N, 1)

        # Foot heights relative to ground (not absolute world z)
        ground_height = self._get_ground_height().unsqueeze(1)  # (N, 1)
        feet_heights = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2] - ground_height  # (N, 4)

        cmd_not_null = self.env.command_manager.get_command("base_velocity").norm(p=1, dim=1) > 0.05

        # Penalize during SWING phase (1 - C), not stance
        return torch.sum(
            torch.square(feet_heights - cmd_height) * (1 - desired_contacts[:, :]),
            dim=1
        ) * cmd_not_null

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        asset_cfg,
        sensor_cfg,
        height_scanner_cfg=None,
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)
        desired_contact_states = self.compute_contact_targets(gait_params)
        return self.compute_footswing_height(desired_contact_states)


class FootClearanceCmdLinearQuad(GaitRewardQuad):
    """Foot clearance reward using phase-modulated triangular height profile.

    During swing phase, the target foot height follows a triangular profile:
        - Start of swing: 0
        - Mid-swing: cmd_feet_height
        - End of swing: 0

    Penalizes only swing feet, weighted by swing mask from the contact schedule.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.foot_radius = cfg.params.get("foot_radius", 0.02)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        asset_cfg,
        sensor_cfg,
        foot_radius=0.02,
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Per-foot timing for triangular profile
        frequency = gait_params[:, IDX_FREQUENCY]
        duration = gait_params[:, IDX_DURATION]
        off2 = gait_params[:, IDX_OFFSET2]
        off3 = gait_params[:, IDX_OFFSET3]
        off4 = gait_params[:, IDX_OFFSET4]

        t = torch.remainder(self._env.episode_length_buf * self.dt * frequency, 1.0)
        t_LF, t_RF, t_LH, t_RH = compute_per_foot_timings(off2, off3, off4, t)
        foot_timings = torch.stack([t_LF, t_RF, t_LH, t_RH], dim=1)  # (N, 4)

        # Triangular target height profile during swing phase
        dur = duration.unsqueeze(1)  # (N, 1)
        swing_duration = (1.0 - dur).clamp(min=0.01)

        # Swing fraction: how far through swing [0, 1]
        swing_frac = ((foot_timings - dur) / swing_duration).clamp(0.0, 1.0)

        # Triangular: ramps up to 1.0 at mid-swing, back to 0
        triangle = 2.0 * torch.min(swing_frac, 1.0 - swing_frac)

        # Target height = cmd_feet_height * triangle
        cmd_feet_height = gait_params[:, IDX_FEET_HEIGHT].unsqueeze(1)  # (N, 1)
        target_height = cmd_feet_height * triangle

        # Current foot heights (corrected by foot radius)
        feet_heights = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2] - self.foot_radius

        # Swing mask from contact schedule
        swing_mask = 1 - desired_contact_states

        # Height error during swing only
        height_error = torch.square(feet_heights - target_height) * swing_mask

        cmd_not_null = env.command_manager.get_command("base_velocity").norm(p=1, dim=1) > 0.05

        return torch.sum(height_error, dim=1) * cmd_not_null


class ActionSmoothnessPenalty(ManagerTermBase):
    """Penalize large instantaneous changes in the network action output."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = None
        self.prev_action = None

    def reset(self, env_ids=None):
        if self.prev_action is not None and env_ids is not None:
            self.prev_action[env_ids] = 0.0
        if self.prev_prev_action is not None and env_ids is not None:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current_action = env.action_manager.action.clone()

        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        return penalty


def base_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_base_height = commands[:, IDX_BASE_HEIGHT]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        adjusted_target_height = cmd_base_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        adjusted_target_height = cmd_base_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking of commanded base height using exp kernel.

    Returns exp(-error^2 / std^2), giving 1.0 at perfect tracking and
    decaying smoothly toward 0.0 as the height error grows.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_base_height = commands[:, IDX_BASE_HEIGHT]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        target = cmd_base_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        target = cmd_base_height

    error = asset.data.root_pos_w[:, 2] - target
    return torch.exp(-torch.square(error) / (std**2))


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def stand_when_zero_command(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cmd_null = env.command_manager.get_command("base_velocity").norm(dim=1, p=1) < 0.05
    return torch.norm(diff_angle, p=1, dim=1) * cmd_null


def stand_still_when_zero_command(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities when no command."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    cmd_null = env.command_manager.get_command("base_velocity").norm(dim=1, p=1) < 0.05
    return torch.norm(joint_vel, p=1, dim=1) * cmd_null


# ---------------------------------------------------------------------------
# Augmented auxiliary rewards (behavior-dependent)
# ---------------------------------------------------------------------------


def orientation_control(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from commanded pitch and roll.

    Extracts current pitch and roll from projected gravity vector in body frame
    and compares to commanded values from the gait command.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_pitch = commands[:, IDX_BODY_PITCH]
    cmd_roll = commands[:, IDX_BODY_ROLL]

    grav = asset.data.projected_gravity_b  # (N, 3)
    current_pitch = torch.atan2(grav[:, 0], -grav[:, 2])
    current_roll = torch.atan2(grav[:, 1], -grav[:, 2])

    return torch.square(current_pitch - cmd_pitch) + torch.square(current_roll - cmd_roll)


def body_pitch_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of body pitch from commanded pitch.

    Pitch is extracted from the projected gravity vector in body frame.
    When upright, projected_gravity_b ≈ [0, 0, -1].
    For pitch angle θ (nose-up positive): projected_gravity_b = [sin(θ), 0, -cos(θ)].
    So: pitch = atan2(g_x, -g_z) = θ.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_pitch = commands[:, IDX_BODY_PITCH]

    grav = asset.data.projected_gravity_b  # (N, 3)
    # Fix: atan2(g_x, -g_z) gives positive pitch for nose-up
    current_pitch = torch.atan2(grav[:, 0], -grav[:, 2])

    return torch.square(current_pitch - cmd_pitch)


def body_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body roll (rotation around forward axis).

    Roll is extracted from the projected gravity vector in body frame.
    When upright, projected_gravity_b ≈ [0, 0, -1].
    Roll angle φ: roll = atan2(g_y, -g_z).
    Target is always zero (upright).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b  # (N, 3)
    current_roll = torch.atan2(grav[:, 1], -grav[:, 2])
    return torch.square(current_roll)


class RaibertHeuristicFootswing(ManagerTermBase):
    """Penalize foot position deviation from the Raibert Heuristic target.

    The Raibert Heuristic computes desired foot landing positions as an adjustment
    to the nominal stance based on velocity command and fixed desired stance width.
    The penalty is applied during the swing phase only.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.dt = env.step_dt
        self.desired_stance_width = cfg.params.get("desired_stance_width", 0.3)
        self.desired_stance_length = cfg.params.get("desired_stance_length", 0.55)

        # Capture default foot positions in body frame at init (robot is in default pose)
        foot_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]  # (N, 4, 3)
        root_pos = self.asset.data.default_root_state[:, :3]  # (N, 3)
        self.default_foot_pos_b = (foot_pos_w - root_pos.unsqueeze(1)).clone()

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg, desired_stance_width=0.3, desired_stance_length=0.55) -> torch.Tensor:
        gait_params = env.command_manager.get_command("gait_command")
        vel_cmd = env.command_manager.get_command("base_velocity")
        vx_cmd = vel_cmd[:, 0]
        frequency = gait_params[:, IDX_FREQUENCY]

        # Get current foot positions in body frame
        base_quat = self.asset.data.root_quat_w
        base_pos = self.asset.data.root_pos_w
        foot_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]  # (N, 4, 3)

        foot_pos_rel = foot_pos_w - base_pos.unsqueeze(1)
        foot_pos_b = torch.zeros_like(foot_pos_rel)
        for i in range(4):
            foot_pos_b[:, i, :] = quat_rotate_inverse(base_quat, foot_pos_rel[:, i, :])

        # Raibert heuristic: desired x = nominal + v_cmd * T_stance/2
        t_stance = 0.5 / torch.clamp(frequency, min=0.5)

        default_pos = self.default_foot_pos_b  # (N, 4, 3)

        # Desired foot x position: nominal + velocity-proportional offset
        p_des_x = default_pos[:, :, 0] + vx_cmd.unsqueeze(1) * t_stance.unsqueeze(1) * 0.5

        # Desired foot y position: use fixed desired stance width
        default_y = default_pos[:, :, 1]  # (N, 4)
        default_half_width = (torch.abs(default_y[:, 0]) + torch.abs(default_y[:, 1])) * 0.5
        desired_half_width = self.desired_stance_width * 0.5
        scale = desired_half_width / torch.clamp(default_half_width, min=0.01)
        p_des_y = default_y * scale.unsqueeze(1)

        error = torch.square(foot_pos_b[:, :, 0] - p_des_x) + torch.square(foot_pos_b[:, :, 1] - p_des_y)

        # Compute swing mask using periodic contact schedule
        off2 = gait_params[:, IDX_OFFSET2]
        off3 = gait_params[:, IDX_OFFSET3]
        off4 = gait_params[:, IDX_OFFSET4]
        t = torch.remainder(self._env.episode_length_buf * self.dt * frequency, 1.0)
        t_LF, t_RF, t_LH, t_RH = compute_per_foot_timings(off2, off3, off4, t)
        foot_timings = torch.stack([t_LF, t_RF, t_LH, t_RH], dim=1)

        contact_states = periodic_contact_schedule(foot_timings, 0.05)
        swing_mask = 1 - contact_states

        cmd_not_null = vel_cmd.norm(p=1, dim=1) > 0.05

        return torch.sum(error * swing_mask, dim=1) * cmd_not_null
