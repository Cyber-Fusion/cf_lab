# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _locomotion_gate(env: ManagerBasedRLEnv, cmd_threshold: float = 0.05, vel_threshold: float = 0.3) -> torch.Tensor:
    """Return a per-env boolean mask that is True when the robot should be locomoting.

    The mask is True when **either**:
    - the velocity command (lin_x, lin_y, ang_z) L1 norm exceeds *cmd_threshold*, OR
    - the actual base planar linear + yaw velocity L1 norm exceeds *vel_threshold*.
    """
    cmd = env.command_manager.get_command("base_velocity")
    asset: RigidObject = env.scene["robot"]
    cmd_active = cmd.norm(p=1, dim=1) > cmd_threshold
    body_vel = torch.cat([asset.data.root_lin_vel_w[:, :2], asset.data.root_ang_vel_w[:, 2:3]], dim=1)
    vel_active = body_vel.norm(p=1, dim=1) > vel_threshold
    return cmd_active | vel_active


def _zero_cmd_mask(env: ManagerBasedRLEnv, cmd_threshold: float = 0.05) -> torch.Tensor:
    """True per env when the velocity command is ~zero (regardless of body velocity)."""
    cmd = env.command_manager.get_command("base_velocity")
    return cmd.norm(p=1, dim=1) <= cmd_threshold


def track_lin_vel_xy_exp_speed_adaptive(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_low: float = 0.25,
    sigma_high: float = 0.5,
    sigma_low_vel: float = 1.0,
    sigma_high_vel: float = 2.0,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) with a speed-adaptive sigma and linear ramp.

    Sigma of the squared-error exp kernel is linearly interpolated from ``sigma_low`` to
    ``sigma_high`` over the commanded ``|v_xy|`` range ``[sigma_low_vel, sigma_high_vel]`` (clamped
    outside). The reward is then scaled by ``max(1.0, 1.0 + ramp_rate * (|v_cmd| - ramp_at_vel))``,
    mirroring the spot-like ``base_linear_velocity_reward`` ramp.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel_cmd_magnitude = torch.linalg.norm(cmd, dim=1)

    t = torch.clamp((vel_cmd_magnitude - sigma_low_vel) / (sigma_high_vel - sigma_low_vel), min=0.0, max=1.0)
    sigma = sigma_low + t * (sigma_high - sigma_low)

    lin_vel_error = torch.sum(
        torch.square(cmd - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / sigma**2)

    velocity_scaling = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return reward * velocity_scaling


def track_base_height_exp(
    env: ManagerBasedRLEnv, std: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_base_height = commands[:, 6]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = cmd_base_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = cmd_base_height

    base_height_error = asset.data.root_pos_w[:, 2] - adjusted_target_height

    return torch.exp(-torch.square(base_height_error) / std**2)


def orientation_control(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation of body orientation from commanded pitch/roll."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    commands = env.command_manager.get_command("gait_command")
    cmd_pitch, cmd_roll = commands[:, 7], commands[:, 8]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(num_envs, 3)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(num_envs, 3)
    quat_r = quat_from_angle_axis(-cmd_roll, x_axis)
    quat_p = quat_from_angle_axis(-cmd_pitch, y_axis)
    desired_quat = quat_mul(quat_r, quat_p)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(num_envs, 3)
    desired_grav = quat_apply_inverse(desired_quat, gravity)

    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2] - desired_grav[:, :2]), dim=1)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


class GaitRewardQuad(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.env = env

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
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
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )  # (num_envs, num_feet)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        # body_lin_vel_w (num_envs, num_feet, 3)
        foot_velocities = torch.norm(
            self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2], dim=-1
        )  # (num_envs, num_feet)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = -(force_reward + velocity_reward)

        return total_reward * _locomotion_gate(env)

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        durations = torch.cat(
            [
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        command_term = self._env.command_manager.get_term("gait_command")
        foot_indices = command_term.foot_indices.clone()

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = torch.distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-(forces[:, i] ** 2) / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-(forces[:, i] ** 2) / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-(velocities[:, i] ** 2) / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-(velocities[:, i] ** 2) / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class ContactWhenWantedLowSpeed(GaitRewardQuad):
    """Reward foot contact force during desired-stance phases, faded out at high |vx|.

    GaitRewardQuad only *penalizes* the wrong quadrants (force in swing, velocity in stance).
    At low |vx| the implicit pressure to plant the foot is weak (body velocity is small, so
    an airborne foot during desired stance doesn't accrue much velocity penalty either), and
    policies tend to drift toward low-duty-cycle hovering trots. This term explicitly rewards
    contact force when contact is wanted, and is faded to zero above ``vx_high`` so it does
    not interfere with the flying-trot regime at high speed.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.vx_low = float(cfg.params["vx_low"])
        self.vx_high = float(cfg.params["vx_high"])
        self.contact_force_sigma = float(cfg.params["contact_force_sigma"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        vx_low: float,
        vx_high: float,
        contact_force_sigma: float,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)
        desired_contact_states = self.compute_contact_targets(gait_params)

        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )  # (num_envs, num_feet)

        # Reward contact force when desired_contact = 1: (1 - exp(-F^2/sigma)) → 0 with no
        # force, → 1 with strong contact. Averaged across feet.
        per_foot = desired_contact_states * (1.0 - torch.exp(-foot_forces**2 / self.contact_force_sigma))
        contact_reward = per_foot.mean(dim=1)

        # Fade with |vx|: 1.0 at |vx|<=vx_low, 0.0 at |vx|>=vx_high, linear in between.
        vx_abs = env.command_manager.get_command("base_velocity")[:, 0].abs()
        fade = 1.0 - torch.clamp((vx_abs - self.vx_low) / max(self.vx_high - self.vx_low, 1e-6), 0.0, 1.0)

        return contact_reward * fade * _locomotion_gate(env)


class FootSwingHeightQuad(GaitRewardQuad):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.env = env

    def compute_footswing_height(self, desired_contacts):
        commands = self.env.command_manager.get_command("gait_command")
        cmd_height = commands[:, 5].unsqueeze(1)
        adjusted_target = cmd_height + self.terrain_height.unsqueeze(1)

        feet_heights = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]

        return torch.sum(
            torch.square(feet_heights - adjusted_target) * (1 - desired_contacts[:, :]), dim=1
        ) * _locomotion_gate(self.env)

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

        if height_scanner_cfg is not None:
            height_scanner: RayCaster = env.scene[height_scanner_cfg.name]
            self.terrain_height = torch.mean(height_scanner.data.ray_hits_w[..., 2], dim=1)
        else:
            self.terrain_height = torch.zeros(self.num_envs, device=self.asset.device)

        desired_contact_states = self.compute_contact_targets(gait_params)

        return self.compute_footswing_height(desired_contact_states)


class FootClearanceCmdLinearQuad(GaitRewardQuad):
    """Foot clearance reward matching walk-these-ways _reward_feet_clearance_cmd_linear.

    Uses a phase-modulated target height (triangular profile peaking at mid-swing)
    and masks by desired contact states so only swing feet are penalized.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.foot_radius = cfg.params.get("foot_radius", 0.02)

    def _compute_unwarped_foot_indices(self, gait_params):
        """Compute per-foot gait indices without stance/swing warping."""
        command_term = self._env.command_manager.get_term("gait_command")
        return command_term.foot_indices.clone()

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
        foot_radius=0.02,
        height_scanner_cfg=None,
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)

        # Per-foot gait indices and commanded stance duration (duty cycle).
        foot_indices = self._compute_unwarped_foot_indices(gait_params)
        durations = gait_params[:, 1].unsqueeze(1)  # (num_envs, 1), broadcasts across feet

        # Triangular phase peaking at mid-swing for ANY duty cycle.
        # swing_pos: 0 at start of swing (foot_index = duration), 1 at end (foot_index = 1).
        swing_pos = torch.clip((foot_indices - durations) / (1.0 - durations), 0.0, 1.0)
        phases = 1.0 - torch.abs(2.0 * swing_pos - 1.0)

        # Desired contact states from warped indices (for swing masking)
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Terrain height offset from height scanner
        if height_scanner_cfg is not None:
            height_scanner: RayCaster = env.scene[height_scanner_cfg.name]
            terrain_height = torch.mean(height_scanner.data.ray_hits_w[..., 2], dim=1).unsqueeze(1)
        else:
            terrain_height = 0.0

        # Target height modulated by phase + foot radius offset + terrain height
        cmd_height = env.command_manager.get_command("gait_command")[:, 5].unsqueeze(1)
        target_height = cmd_height * phases + self.foot_radius + terrain_height

        # Foot heights in world frame
        foot_height = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]

        # Penalize only swing feet
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1) * _locomotion_gate(env)


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
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
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )  # (num_envs, num_feet)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        # body_lin_vel_w (num_envs, num_feet, 3)
        foot_velocities = torch.norm(
            self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2], dim=-1
        )  # (num_envs, num_feet)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = torch.distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-(forces[:, i] ** 2) / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-(forces[:, i] ** 2) / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-(velocities[:, i] ** 2) / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-(velocities[:, i] ** 2) / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None

    def reset(self, env_ids=None):
        if self.prev_action is not None and env_ids is not None:
            self.prev_action[env_ids] = 0.0
        if self.prev_prev_action is not None and env_ids is not None:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    return torch.exp(-feet_height_error / std**2)


def handstand_feet_on_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.all(first_air, dim=1).float()
    return reward


def handstand_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward


def handstand_orientation_l2(
    env: ManagerBasedRLEnv, target_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Define the target gravity direction for an upright posture in the base frame
    target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
    # Penalize deviation of the projected gravity vector from the target
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact


def base_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_base_height = commands[:, 6]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = cmd_base_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = cmd_base_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_feet_height: float = 0.35,
) -> torch.Tensor:
    asset_feet: Articulation = env.scene[asset_feet_cfg.name]
    asset_base: Articulation = env.scene[asset_base_cfg.name]

    feet_positions = asset_feet.data.body_pos_w[:, asset_feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    feet_vels = asset_feet.data.body_lin_vel_w[:, asset_feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)

    base_rotation = asset_base.data.root_link_quat_w[:, :]  # (num_envs, 4)
    base_positions = asset_base.data.root_link_pos_w[:, :]  # (num_envs, 3)
    base_vels = asset_base.data.root_link_lin_vel_w[:, :]  # (num_envs, 3)

    num_envs = feet_positions.shape[0]
    num_feet = feet_positions.shape[1]
    cur_footpos_translated = feet_positions - base_positions.unsqueeze(1)
    footpos_in_body_frame = torch.zeros(num_envs, num_feet, 3, device="cuda")
    cur_footvel_translated = feet_vels - base_vels.unsqueeze(1)
    footvel_in_body_frame = torch.zeros(num_envs, num_feet, 3, device="cuda")
    for i in range(num_feet):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(base_rotation, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(base_rotation, cur_footvel_translated[:, i, :])

    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_feet_height).view(num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(num_envs, -1)

    return torch.sum(height_error * foot_leteral_vel, dim=1)


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_height = commands[:, 5]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = cmd_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = cmd_height

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - adjusted_target_height)
    reward = foot_z_target_error * torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
        p=0.5,
        dim=-1,
    )
    return torch.sum(reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    commands = env.command_manager.get_command("gait_command")
    cmd_height = commands[:, 5].unsqueeze(1)

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - cmd_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh

    cmd_not_null = env.command_manager.get_command("base_velocity").norm(p=1, dim=1) > 0.05

    return torch.exp(-torch.sum(reward, dim=1) / std) * cmd_not_null


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
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


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # Penalize feet stumbling
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_norm = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1)
    vertical_contacts = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    reward = torch.any(contacts_norm > 5 * vertical_contacts, dim=1)
    return reward


def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    asset: Articulation = env.scene[asset_cfg.name]

    feet_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


def feet_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize if the distance between feet is below a minimum threshold."""

    asset: Articulation = env.scene[asset_cfg.name]

    feet_position_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, 0:2]  # (num_envs, num_feet, 2)

    if feet_position_xy is None:
        return torch.zeros(env.num_envs)

    # feet distance on x-y plane
    feet_distance = torch.norm(feet_position_xy[:, 0, :2] - feet_position_xy[:, 1, :2], dim=-1)

    return torch.clamp(0.1 - feet_distance, min=0.0)


def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()


def stand_still(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


def stand_when_zero_command(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_cmd_only_gate: bool = False,
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command.

    When ``use_cmd_only_gate`` is False (default), the gate is the inverse of the locomotion
    gate (which also looks at body velocity). When True, gate only on a near-zero velocity
    command — the reward fires even if the body still has residual velocity.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    mask = _zero_cmd_mask(env) if use_cmd_only_gate else ~_locomotion_gate(env)
    return torch.norm(diff_angle, p=1, dim=1) * mask


def stand_still_when_zero_command(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_cmd_only_gate: bool = False,
) -> torch.Tensor:
    """Penalize joint velocities when no command. See :func:`stand_when_zero_command` for ``use_cmd_only_gate``."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    mask = _zero_cmd_mask(env) if use_cmd_only_gate else ~_locomotion_gate(env)
    return torch.norm(joint_vel, p=1, dim=1) * mask


def zero_vel_when_zero_command(
    env,
    command_name: str,
    cmd_threshold: float = 0.05,
    yaw_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base linear and angular velocity when the velocity command is zero.

    Returns ``||v_xy|| + yaw_weight * |yaw_rate|`` masked to only apply when the command L1 norm
    is below *cmd_threshold*. Use with a negative weight.
    """
    cmd = env.command_manager.get_command(command_name)
    zero_cmd_mask = cmd.norm(p=1, dim=1) <= cmd_threshold

    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_xy = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_ang_vel_w[:, 2])

    return (lin_vel_xy + yaw_weight * yaw_rate) * zero_cmd_mask


def zero_ang_vel_when_zero_command(
    env,
    command_name: str,
    cmd_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base angular velocity when the velocity command is zero.

    Returns ``||omega||`` masked to only apply when the command L1 norm is below *cmd_threshold*.
    Use with a negative weight.
    """
    cmd = env.command_manager.get_command(command_name)
    zero_cmd_mask = cmd.norm(p=1, dim=1) <= cmd_threshold

    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = torch.norm(asset.data.root_ang_vel_w, dim=1)

    return ang_vel * zero_cmd_mask


class RaibertHeuristicReward(ManagerTermBase):
    """Penalize deviation of foot positions from Raibert heuristic desired positions."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg = cfg.params["asset_cfg"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt
        self.desired_stance_width = cfg.params["desired_stance_width"]
        self.desired_stance_length = cfg.params["desired_stance_length"]

        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name,
        asset_cfg,
        desired_stance_width,
        desired_stance_length,
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)
        frequencies = gait_params[:, 0]

        # Read per-foot gait indices from the command term (incremental, no phase jumps)
        command_term = self._env.command_manager.get_term("gait_command")
        foot_indices = command_term.foot_indices.clone()

        # Transform foot positions to yaw-only body frame
        foot_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]  # (N, 4, 3)
        base_pos_w = self.asset.data.root_link_pos_w  # (N, 3)
        base_quat_w = self.asset.data.root_link_quat_w  # (N, 4)

        translated = foot_pos_w - base_pos_w.unsqueeze(1)
        yaw_q = yaw_quat(base_quat_w)

        body_frame = torch.zeros(self.num_envs, 4, 3, device=self.asset.device)
        for i in range(4):
            body_frame[:, i, :] = quat_apply_inverse(yaw_q, translated[:, i, :])

        # Nominal stance positions for cf_lab foot order [LF, RF, LH, RH]
        W = self.desired_stance_width
        L = self.desired_stance_length
        xs_nom = torch.tensor([L / 2, L / 2, -L / 2, -L / 2], device=self.asset.device).unsqueeze(0)
        ys_nom = torch.tensor([W / 2, -W / 2, W / 2, -W / 2], device=self.asset.device).unsqueeze(0)

        # Asymmetric triangle: +1 -> -1 across stance, -1 -> +1 across swing.
        # Reduces to the canonical symmetric triangle when duration = 0.5.
        durations = gait_params[:, 1].unsqueeze(1)  # (num_envs, 1), broadcasts across feet
        stance_mask = foot_indices < durations
        phases = torch.where(
            stance_mask,
            1.0 - 2.0 * foot_indices / durations,
            -1.0 + 2.0 * (foot_indices - durations) / (1.0 - durations),
        )

        x_vel_des = env.command_manager.get_command("base_velocity")[:, 0:1]
        yaw_vel_des = env.command_manager.get_command("base_velocity")[:, 2:3]
        y_vel_des = yaw_vel_des * self.desired_stance_length / 2

        # Peak offset = v_des * T_stance / 2 = v_des * duration / (2 * freq).
        stance_half_period = durations / (2.0 * frequencies.unsqueeze(1))
        xs_offset = phases * x_vel_des * stance_half_period
        ys_offset = phases * y_vel_des * stance_half_period
        ys_offset[:, 2:4] *= -1  # flip sign for hind legs (LH, RH)

        desired_xs = xs_nom + xs_offset
        desired_ys = ys_nom + ys_offset
        desired = torch.cat((desired_xs.unsqueeze(2), desired_ys.unsqueeze(2)), dim=2)  # (N, 4, 2)

        err = torch.abs(desired - body_frame[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2)) * _locomotion_gate(env)


def feet_regulation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    desired_body_height: float = 0.65,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    feet_positions_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet, 1)

    feet_velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0:2]  # (num_envs, num_feet, 2)
    vel_norms_xy = torch.norm(feet_velocity_xy, dim=-1)  # (num_envs, num_feet)

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_desired_body_height = desired_body_height + sensor.data.pos_w[:, 2]  # (num_envs, 1)
        adjusted_desired_body_height = adjusted_desired_body_height.unsqueeze(1).repeat(1, 2)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_desired_body_height = desired_body_height

    exp_term = torch.exp(-feet_positions_z / (0.025 * adjusted_desired_body_height))  # (num_envs, num_feet)
    exp_term = torch.clamp(exp_term, min=0.001, max=10.0)
    r_fr = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

    return r_fr
