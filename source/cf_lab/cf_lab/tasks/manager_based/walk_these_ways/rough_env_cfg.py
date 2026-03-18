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

        # reduce action scale
        self.actions.joint_pos.scale = {".*HAA": 0.125, ".*HFE": 0.25, ".*KFE": 0.25}

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
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

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.rewards.lin_vel_z_l2.weight = -0.02
        self.rewards.ang_vel_xy_l2.weight = -0.001
        self.rewards.orientation_control.weight = -5.0

        self.rewards.joint_vel_l2.weight = -1.0e-3
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -1.0e-3 / 10

        self.rewards.base_height_l2.weight = -10.0
        self.rewards.feet_slip.weight = -0.04

        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.action_smoothness_l2.weight = -0.1 / 2

        self.rewards.feet_air_time.weight = -0.0
        self.rewards.undesired_contacts.weight = -1.0 * 5
        
        self.rewards.joint_deviation_l1.weight = -1.0

        self.rewards.raibert_heuristic.weight = -10.0 / 2
        self.rewards.gait.weight = 4.0 * 2
        self.rewards.footswing_height.weight = -0.0
        self.rewards.foot_clearance.weight = -30.0 * 2

        self.rewards.stand_when_zero_command.weight = -0.0
        self.rewards.stand_still_when_zero_command.weight = -0.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.gait_command.multi_gait = True
        self.commands.gait_command.binary_phases = True
        self.commands.gait_command.ranges.frequencies = (2.0, 4.0)
        self.commands.gait_command.ranges.body_pitch = (-0.4, 0.4)
        self.commands.gait_command.ranges.body_roll = (-0.2, 0.2)

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
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
