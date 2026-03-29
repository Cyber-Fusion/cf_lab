from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .ayg_env_cfg import AygFlatEnvCfg, AygRoughEnvCfg

# Observation noise ranges (matching manager-based env)
NOISE_RANGES = {
    "base_lin_vel": 0.1,
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
    "joint_pos": 0.01,
    "joint_vel": 1.5,
    "height_scan": 0.1,
}


class AygEnv(DirectRLEnv):
    cfg: AygFlatEnvCfg | AygRoughEnvCfg

    def __init__(self, cfg: AygFlatEnvCfg | AygRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Heading target for heading-based yaw control
        self._heading_target = torch.zeros(self.num_envs, device=self.device)
        # Command resampling timer (in steps)
        self._command_resample_steps = int(self.cfg.command_resample_time / (self.cfg.sim.dt * self.cfg.decimation))
        self._command_time_left = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "feet_regulation",
                "foot_clearance",
                "base_height",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies(["Base", ".*_Hip"])
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_Foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*_Shank", ".*_Thigh"])
        self._feet_body_ids, _ = self._robot.find_bodies(".*_Foot")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AygRoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        # Resample commands for envs whose timer has expired
        self._command_time_left -= 1
        resample_ids = (self._command_time_left <= 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Compute heading-based yaw command
        root_quat = self._robot.data.root_quat_w
        _, _, yaw = math_utils.euler_xyz_from_quat(root_quat)
        heading_error = torch.atan2(
            torch.sin(self._heading_target - yaw), torch.cos(self._heading_target - yaw)
        )
        self._commands[:, 2] = torch.clamp(
            self.cfg.heading_control_stiffness * heading_error,
            min=-1.0,
            max=1.0,
        )

        # Build observation with noise
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        if self.cfg.enable_obs_noise:
            base_lin_vel = base_lin_vel + torch.empty_like(base_lin_vel).uniform_(
                -NOISE_RANGES["base_lin_vel"], NOISE_RANGES["base_lin_vel"]
            )
            base_ang_vel = base_ang_vel + torch.empty_like(base_ang_vel).uniform_(
                -NOISE_RANGES["base_ang_vel"], NOISE_RANGES["base_ang_vel"]
            )
            projected_gravity = projected_gravity + torch.empty_like(projected_gravity).uniform_(
                -NOISE_RANGES["projected_gravity"], NOISE_RANGES["projected_gravity"]
            )
            joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(
                -NOISE_RANGES["joint_pos"], NOISE_RANGES["joint_pos"]
            )
            joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(
                -NOISE_RANGES["joint_vel"], NOISE_RANGES["joint_vel"]
            )

        height_data = None
        if isinstance(self.cfg, AygRoughEnvCfg):
            nominal_standing_height = 0.35
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self._height_scanner.data.ray_hits_w[..., 2]
                - nominal_standing_height
            ).clip(-1.0, 1.0)
            if self.cfg.enable_obs_noise:
                height_data = height_data + torch.empty_like(height_data).uniform_(
                    -NOISE_RANGES["height_scan"], NOISE_RANGES["height_scan"]
                )

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    base_lin_vel,
                    base_ang_vel,
                    projected_gravity,
                    self._commands,
                    self._heading_target.unsqueeze(1),
                    joint_pos,
                    joint_vel,
                    self._actions,
                    height_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum(
            (last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1
        ) * (torch.norm(self._commands[:, :2], dim=1) > 0.1)
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # feet regulation
        feet_pos_z = self._robot.data.body_pos_w[:, self._feet_body_ids, 2]
        feet_vel_xy = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, 0:2]
        vel_norms_xy = torch.norm(feet_vel_xy, dim=-1)
        exp_term = torch.exp(-feet_pos_z / (0.025 * self.cfg.base_height_target))
        exp_term = torch.clamp(exp_term, min=0.001, max=10.0)
        feet_reg = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

        # foot clearance: reward feet for achieving target height during swing phase
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        is_in_swing = current_air_time > 0.0
        # Reward: exp(-|feet_z - target|/sigma) for swing feet, only when moving
        clearance_error = torch.square(feet_pos_z - self.cfg.foot_clearance_target)
        foot_clearance = torch.sum(torch.exp(-clearance_error / 0.005) * is_in_swing.float(), dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        # base height
        base_height = self._robot.data.root_pos_w[:, 2]
        base_height_error = torch.square(base_height - self.cfg.base_height_target)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "feet_regulation": feet_reg * self.cfg.feet_regulation_reward_scale * self.step_dt,
            "foot_clearance": foot_clearance * self.cfg.foot_clearance_reward_scale * self.step_dt,
            "base_height": base_height_error * self.cfg.base_height_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._resample_commands(env_ids)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Randomize base pose (x, y, yaw) matching manager-based env
        default_root_state[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        default_root_state[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-3.14, 3.14)
        quat_yaw = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        default_root_state[:, 3:7] = math_utils.quat_mul(quat_yaw, default_root_state[:, 3:7])
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample velocity commands for given environment indices."""
        n = len(env_ids)
        # Sample linear velocity commands
        self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(-1.0, 1.0)
        self._commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(-1.0, 1.0)
        # Sample heading target (yaw command is computed from heading error in _get_observations)
        self._heading_target[env_ids] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        # Set standing-still fraction: 2% of envs get zero commands
        standing_mask = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
        self._commands[env_ids[standing_mask], :] = 0.0
        self._heading_target[env_ids[standing_mask]] = 0.0
        # Reset resample timer
        self._command_time_left[env_ids] = self._command_resample_steps
