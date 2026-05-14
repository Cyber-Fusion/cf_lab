# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import mdp as ayg_mdp

##
# Pre-defined configs
##
from cf_lab.assets.ayg import AYG_CFG  # isort: skip


@configclass
class AygRewardsCfg(RewardsCfg):
    """Extended reward config with AYG-specific reward terms."""

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={"target_height": 0.35, "asset_cfg": SceneEntityCfg("robot", body_names="Base")},
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    feet_regulation = RewTerm(
        func=ayg_mdp.feet_regulation,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot")},
    )

    foot_clearance = RewTerm(
        func=ayg_mdp.foot_clearance_swing,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
            "target_height": 0.10,
            "sigma": 0.005,
        },
    )

    flight_phase = RewTerm(
        func=ayg_mdp.flight_phase_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot")},
    )


@configclass
class AygRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: AygRewardsCfg = AygRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to ayg and rename stuff
        self.scene.robot = AYG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Base"
        # Randomization body name overrides (parent uses lowercase "base", AYG body is "Base")
        # and AYG-appropriate mass range (~10 kg robot, ±1.5 kg ≈ 15%).
        self.events.add_base_mass.params["asset_cfg"].body_names = "Base"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 1.5)
        self.events.base_com.params["asset_cfg"].body_names = "Base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "Base"
        # Rename the joints in the rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_Foot"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_Shank", ".*_Thigh"]
        # Rename the joints in the terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Base", ".*_Hip"]

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.physics_material.params["static_friction_range"] = (0.4, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 2.0)
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
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -5e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.feet_air_time.weight = 0.125
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_pos_limits.weight = -0.01
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = "Base"
        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.joint_deviation_l1.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*HAA")
        self.rewards.feet_regulation.weight = 0.0
        self.rewards.feet_regulation.params["asset_cfg"].body_names = ".*_Foot"
        self.rewards.feet_regulation.params["desired_body_height"] = 0.35
        self.rewards.foot_clearance.weight = 0.25
        self.rewards.foot_clearance.params["asset_cfg"].body_names = ".*_Foot"
        self.rewards.foot_clearance.params["sensor_cfg"].body_names = ".*_Foot"
        self.rewards.foot_clearance.params["sigma"] = 0.01
        self.rewards.flight_phase.weight = -1.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

@configclass
class AygRoughEnvCfg_PLAY(AygRoughEnvCfg):
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
        self.events.base_external_force_torque = None
        self.events.push_robot = None
