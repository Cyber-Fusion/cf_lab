# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import AygRoughWTWEnvCfg


@configclass
class AygFlatWTWEnvCfg(AygRoughWTWEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # null out height_scanner refs in reward params for flat terrain
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.footswing_height.params["height_scanner_cfg"] = None
        self.rewards.foot_clearance.params["height_scanner_cfg"] = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ============================== Rewards ============================== #
        # Flat-specific reward weights (kernel/params come from the base config now).
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Reward weights that differ from the rough config (which is being tuned on a branch).
        self.rewards.base_height_l2.weight = -200.0
        self.rewards.stand_when_zero_command.weight = -1.0
        self.rewards.stand_still_when_zero_command.weight = -0.0
        # The trained policy did not use zero_ang_vel_when_zero_command.
        self.rewards.zero_ang_vel_when_zero_command.weight = 0.0

        # =============================== Trot only ============================== #
        # Disable canonical-gait sampling and pin offsets to trot (LF/RH in phase, RF/LH
        # half-cycle out). With `multi_gait=False`, `gait_ids` stays at 0 for every env so
        # `ranges_per_gait[0]` (the trot slot below) is what gets used by the velocity term.
        self.commands.gait_command.multi_gait = False
        self.commands.gait_command.ranges.offsets2 = (0.5, 0.5)  # RF
        self.commands.gait_command.ranges.offsets3 = (0.5, 0.5)  # LH
        self.commands.gait_command.ranges.offsets4 = (0.0, 0.0)  # RH

        # =================== Command ranges (per canonical gait) ================= #
        # Order matches GaitCommandQuad.CANONICAL_GAITS: [trot, pace, bound, pronk]. vy and omega
        # are capped at |1.0|; vx keeps its high per-gait envelope. Lower per-gait values can be
        # set here individually for gaits that struggle with lateral / yaw motion.
        gait_ranges = self.commands.base_velocity.ranges_per_gait
        # trot
        gait_ranges[0].lin_vel_x = (-4.0, 4.0)
        gait_ranges[0].lin_vel_y = (-1.0, 1.0)
        gait_ranges[0].ang_vel_z = (-1.0, 1.0)
        # pace
        gait_ranges[1].lin_vel_x = (-3.0, 3.0)
        gait_ranges[1].lin_vel_y = (-1.0, 1.0)
        gait_ranges[1].ang_vel_z = (-1.0, 1.0)
        # bound
        gait_ranges[2].lin_vel_x = (-2.0, 2.0)
        gait_ranges[2].lin_vel_y = (-1.0, 1.0)
        gait_ranges[2].ang_vel_z = (-1.0, 1.0)
        # pronk
        gait_ranges[3].lin_vel_x = (-2.0, 2.0)
        gait_ranges[3].lin_vel_y = (-1.0, 1.0)
        gait_ranges[3].ang_vel_z = (-1.0, 1.0)

        # Outer clamp used by the parent's heading-control output; max per-gait |omega| is 1.0.
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_heading_envs = 1.0


@configclass
class AygFlatWTWEnvCfg_PLAY(AygFlatWTWEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # # remove random pushing event
        # self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Skip the velocity curriculum so play sees the full per-gait ranges immediately.
        self.curriculum.gait_velocity_curriculum.params["anneal_steps"] = 1
