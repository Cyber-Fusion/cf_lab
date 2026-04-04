# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from cf_lab.tasks.manager_based.walk_these_ways.wtw_params import WalkTheseWaysParams as Params

from cf_lab.tasks.manager_based.walk_these_ways.wtw_env_cfg import LocomotionWalkTheseWaysRoughEnvCfg

##
# Pre-defined configs
##
from cf_lab.assets.ayg import AYG_CFG  # isort: skip


@configclass
class AygRoughWTWEnvCfg(LocomotionWalkTheseWaysRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to ayg and rename stuff
        self.scene.robot = AYG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # per-joint action scale: HAA smaller to prevent spider stance
        self.actions.joint_pos.scale = {".*HAA": 0.125, ".*HFE": 0.3, ".*KFE": 0.3}

        # event — push_robot inherited from base config (interval, ±0.5 m/s) for robustness
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards — velocity tracking
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # penalties — motion quality
        self.rewards.lin_vel_z_l2.weight = -2.0e-2
        self.rewards.ang_vel_xy_l2.weight = -2.0e-2
        self.rewards.flat_orientation_l2.weight = 0.0  # disabled — conflicts with body_pitch command

        self.rewards.joint_deviation_l1.weight = -0.05
        self.rewards.joint_vel_l2.weight = -2.0e-5
        self.rewards.joint_acc_l2.weight = -5.0e-9
        self.rewards.joint_torques_l2.weight = -2.0e-5

        self.rewards.base_height_l2.weight = -30.0  # was -0.2; massively increased to prevent crawling
        self.rewards.feet_slip.weight = -8.0e-4

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01

        self.rewards.feet_air_time.weight = -0.0
        self.rewards.undesired_contacts.weight = -2.0  # was -1.0; stronger anti-cheat

        # gait tracking
        self.rewards.gait.weight = 4.0  # was 0.5; much stronger gait signal
        self.rewards.gait.params["gait_vel_sigma"] = 1.25
        self.rewards.gait.params["kappa_gait_probs"] = 0.05
        self.rewards.footswing_height.weight = -2.0
        self.rewards.footswing_height.params["gait_vel_sigma"] = 1.25
        self.rewards.footswing_height.params["kappa_gait_probs"] = 0.05
        self.rewards.foot_clearance.weight = 0.0

        # orientation — separate pitch tracking and roll penalty (replacing flat_orientation_l2)
        self.rewards.body_pitch_tracking.weight = -3.0  # was -0.1; tracks commanded pitch
        self.rewards.body_roll_l2.weight = -3.0  # new: penalizes roll (keep upright)
        self.rewards.raibert_heuristic.weight = 0.0  # disabled

        self.rewards.stand_when_zero_command.weight = -0.01
        self.rewards.stand_still_when_zero_command.weight = -0.01

        # Commands — rough uses non-heading mode
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Gait command ranges tuned for AYG (heavier, slower joints than Go1)
        self.commands.gait_command.ranges.frequencies = (1.5, 3.0)
        self.commands.gait_command.ranges.base_height = (0.28, 0.38)

        # Velocity curriculum caps (AYG can't reach Go1 speeds)
        self.curriculum.velocity_curriculum.params["max_lin_vel_x"] = 1.5
        self.curriculum.velocity_curriculum.params["max_lin_vel_y"] = 0.5

@configclass
class AygRoughWTWEnvCfg_PLAY(AygRoughWTWEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
