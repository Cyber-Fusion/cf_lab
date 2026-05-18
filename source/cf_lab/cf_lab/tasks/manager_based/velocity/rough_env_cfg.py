# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
)

from . import mdp as ayg_mdp

##
# Pre-defined configs
##
from cf_lab.assets.ayg import AYG_CFG  # isort: skip


##
# Scene with an added forward-facing depth raycaster (used by the distillation student)
##


@configclass
class AygSceneCfg(MySceneCfg):
    """Scene config that augments the upstream rough scene with a sparse forward depth camera."""

    # ROS-convention rotation that aligns the camera optical axis (+Z) with the robot's forward (+X).
    # Quaternion in (w, x, y, z): rotates camera frame {X=right, Y=down, Z=forward} into
    # robot/base frame {X=forward, Y=left, Z=up}.
    front_depth_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Base",
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=5.0,
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            width=8,
            height=6,
        ),
    )


##
# Observations: keep the original 8-term `policy` (= teacher input) and add a `student` group
# with proprio (no base_lin_vel) + a noisy sparse forward depth point cloud.
##


@configclass
class AygObservationsCfg(ObservationsCfg):
    """Observations with both the original teacher policy group and a new student group."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher input — reproduces the scaling/clipping/noise the loaded PPO
        checkpoint was trained with (see
        ``logs/rsl_rl/ayg_rough/2026-04-24_23-34-06/params/env.yaml``). The upstream
        ``LocomotionVelocityRoughEnvCfg`` dropped these scales/clips in a later
        IsaacLab version, so without this override the frozen teacher would see
        ``base_ang_vel`` 5x and ``joint_vel`` 20x larger than at training time.
        """

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentCfg(ObsGroup):
        """Proprioception (without base_lin_vel) + sparse forward depth point cloud."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        front_depth_pointcloud = ObsTerm(
            func=ayg_mdp.front_depth_pointcloud,
            params={
                "sensor_cfg": SceneEntityCfg("front_depth_camera"),
                "min_range": 0.2,
                "max_range": 5.0,
                "noise_std": 0.02,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    student: StudentCfg = StudentCfg()


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
    scene: AygSceneCfg = AygSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AygObservationsCfg = AygObservationsCfg()
    rewards: AygRewardsCfg = AygRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to ayg and rename stuff
        self.scene.robot = AYG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Base"
        # Tick the depth raycaster at the same rate as the existing height_scanner.
        self.scene.front_depth_camera.update_period = self.decimation * self.sim.dt
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
        self.observations.student.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AygRoughDualCurrEnvCfg(AygRoughEnvCfg):
    """Rough variant that splits envs 50/50 into a forward and a non-forward pool.

    Both pools use the stock terrain_levels_vel curriculum; per-pool mean levels are
    logged separately so the two skill curves are visible in TensorBoard.
    """

    def __post_init__(self):
        super().__post_init__()
        base_cmd = self.commands.base_velocity
        self.commands.base_velocity = ayg_mdp.DualPoolUniformVelocityCommandCfg(
            asset_name=base_cmd.asset_name,
            resampling_time_range=base_cmd.resampling_time_range,
            rel_standing_envs=base_cmd.rel_standing_envs,
            rel_heading_envs=base_cmd.rel_heading_envs,
            heading_command=base_cmd.heading_command,
            heading_control_stiffness=base_cmd.heading_control_stiffness,
            debug_vis=base_cmd.debug_vis,
            ranges=base_cmd.ranges,
            forward_env_fraction=0.5,
        )
        self.curriculum.terrain_levels_fwd = CurrTerm(func=ayg_mdp.terrain_levels_fwd_pool_mean)
        self.curriculum.terrain_levels_nonfwd = CurrTerm(func=ayg_mdp.terrain_levels_nonfwd_pool_mean)


@configclass
class AygRoughDualCurrEnvCfg_PLAY(AygRoughDualCurrEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.observations.student.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
