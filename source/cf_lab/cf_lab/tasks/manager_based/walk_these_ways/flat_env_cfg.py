# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import AygRoughWTWEnvCfg


@configclass
class AygFlatWTWEnvCfg(AygRoughWTWEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ====================================================================
        # Terrain: flat plane
        # ====================================================================
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ====================================================================
        # Terminations (flat-specific overrides)
        # ====================================================================
        # Disable shank/thigh termination — too aggressive for gait learning.
        # Paper only penalizes these contacts (via undesired_contacts reward).
        # Keeping the penalty (-8.0 in exp-neg gate) is sufficient.
        self.terminations.shank_thigh_contact = None

        # ====================================================================
        # Events (flat-specific overrides)
        # ====================================================================
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        # Friction DR — covers Isaac Lab nominal (0.8/0.6) and Gazebo ground (1.0/1.0)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        # Push robot for perturbation recovery training (Phase 2)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.events.push_robot.interval_range_s = (15.0, 25.0)

        # ====================================================================
        # Commands (flat-specific)
        # ====================================================================
        # Velocity command: heading mode
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.heading_control_stiffness = 0.5
        self.commands.base_velocity.rel_standing_envs = 0.3  # 30% zero-command exposure (was 20%)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Gait command ranges
        self.commands.gait_command.ranges.frequencies = (1.5, 3.0)
        self.commands.gait_command.ranges.durations = (0.5, 0.5)
        self.commands.gait_command.ranges.feet_height = (0.05, 0.2)
        self.commands.gait_command.ranges.base_height = (0.25, 0.35)
        self.commands.gait_command.ranges.body_pitch = (-0.4, 0.4)
        self.commands.gait_command.ranges.body_roll = (-0.2, 0.2)

        # ====================================================================
        # Rewards (flat-specific weights — ADDITIVE terms)
        # ====================================================================
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_base_height_exp.weight = 1.0
        self.rewards.track_base_height_exp.params["sensor_cfg"] = None  # flat terrain

        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.joint_vel_l2.weight = -1.0e-3
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -2.0e-4

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.03  # increased to dampen base shaking during standing

        # ====================================================================
        # Rewards (flat-specific weights — EXP_NEGATIVE terms)
        # ====================================================================
        self.rewards.gait.weight = 10.0
        self.rewards.gait.params["gait_force_sigma"] = 50.0
        self.rewards.gait.params["gait_vel_sigma"] = 1.0
        self.rewards.gait.params["kappa_gait_probs"] = 0.07

        self.rewards.orientation_control.weight = -40.0

        self.rewards.base_height_l2.weight = -50.0  # strengthened to enforce height against dominant gait reward
        self.rewards.base_height_l2.params["sensor_cfg"] = None  # flat terrain

        self.rewards.feet_slip.weight = -0.04

        self.rewards.undesired_contacts.weight = -8.0

        self.rewards.foot_clearance.weight = -30.0
        self.rewards.foot_clearance.params["foot_radius"] = 0.02
        self.rewards.foot_clearance.params["gait_force_sigma"] = 50.0
        self.rewards.foot_clearance.params["gait_vel_sigma"] = 1.0
        self.rewards.foot_clearance.params["kappa_gait_probs"] = 0.07

        # ====================================================================
        # Curriculum (flat-specific)
        # ====================================================================
        # Slower sigma annealing over ~2000 iterations to max.
        self.curriculum.sigma_exp_neg_anneal.params["anneal_steps"] = 72000
        # Cap sigma_max at 5.0 (default 20.0). With positive gait reward in
        # the exp-neg gate, high sigma causes astronomical reward amplification
        # that drowns out velocity tracking. sigma=5 enforces good behavior
        # while keeping tracking viable. Tuning confirmed: sigma=2-5 gives
        # best velocity tracking (error_vel_xy=0.28 at sigma=2.2 vs 0.52 at sigma=20).
        self.curriculum.sigma_exp_neg_anneal.params["sigma_max"] = 3.0

        # Velocity curriculum caps — AYG can't reach Go1 speeds
        self.curriculum.velocity_curriculum.params["max_lin_vel_x"] = 2.0
        self.curriculum.velocity_curriculum.params["max_lin_vel_y"] = 1.0
        self.curriculum.velocity_curriculum.params["max_ang_vel_z"] = 3.0

        # ====================================================================
        # Disabled terms (override rough-specific weights back to zero)
        # ====================================================================
        self.rewards.footswing_height.weight = 0.0  # disabled; foot_clearance (-30 exp-neg) provides correct triangular trajectory
        self.rewards.footswing_height.params["height_scanner_cfg"] = None
        self.rewards.body_pitch_tracking.weight = 0.0
        self.rewards.body_roll_l2.weight = 0.0
        # HAA-only: anchor lateral leg position to default at zero cmd.
        # HFE/KFE stay free for height/pitch/roll tracking — no conflict.
        self.rewards.stand_when_zero_command.weight = -0.5
        self.rewards.stand_when_zero_command.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*HAA"])
        # HAA-only: penalize HAA joint velocities at zero cmd to prevent lateral drift/oscillation.
        # HFE/KFE stay free for posture tracking — no conflict.
        self.rewards.stand_still_when_zero_command.weight = -0.5
        self.rewards.stand_still_when_zero_command.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*HAA"])
        self.rewards.stand_still_base_vel.weight = -0.5  # ADDITIVE: directly penalize base drift at zero command
        self.rewards.track_zero_vel_exp.weight = 8.0  # ADDITIVE: exp reward for zero velocity at zero command
        self.rewards.track_zero_vel_exp.params["std"] = 0.2  # wide enough to provide gradient at typical drift speeds
        self.rewards.joint_deviation_l1.weight = -0.03  # gentle regularization to prevent HAA splay


class AygFlatWTWEnvCfg_PLAY(AygFlatWTWEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
