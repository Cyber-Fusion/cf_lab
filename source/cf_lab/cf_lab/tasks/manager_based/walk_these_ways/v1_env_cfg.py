# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walk-These-Ways v1 (additive rewards).

The v0 task uses the multiplicative reward formulation via ``WTWManagerBasedRLEnv``
(``ExpNegativeRewardManager`` combines additive and exp-negative terms as
``sum(add) * exp(sum(neg) * sigma)``). v1 uses the standard
``isaaclab.envs:ManagerBasedRLEnv``, summing every term. Gait pressure is
provided by ``GaitRewardQuad``'s internal per-env scale
``gait_full_magnitude * per_gait_progress[gait_id]``, advanced by four
``per_gait_progress_curriculum`` terms.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from cf_lab.tasks.manager_based.walk_these_ways import mdp
from cf_lab.tasks.manager_based.walk_these_ways.rough_env_cfg import AygRoughWTWEnvCfg
from cf_lab.tasks.manager_based.walk_these_ways.wtw_env_cfg import RewardsCfg


@configclass
class RewardsCfgV1(RewardsCfg):
    """Additive-mode reward terms.

    Inherits all v0 term definitions; only swaps the stand-still reward functions
    to use the looser command-only gating (no velocity check). Weights are set
    per-env-cfg in :class:`AygRoughWTWEnvCfgV1`.
    """

    def __post_init__(self):
        # Use the command-only gating variants for v1 (the v0 functions key off
        # _locomotion_gate, which also fires when actual body velocity is high).
        self.stand_when_zero_command.func = mdp.stand_when_zero_cmd_only
        self.stand_still_when_zero_command.func = mdp.stand_still_when_zero_cmd_only


@configclass
class CurriculumCfgV1:
    """v1 curriculum: terrain levels + four independent per-gait progress curriculums.

    Replaces v0's ``sigma_exp_neg_anneal`` (which is irrelevant under additive
    rewards). Each per-gait term advances ``GaitRewardQuad.per_gait_progress``
    independently for trot/pace/bound/pronk.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    gait_progress_trot = CurrTerm(
        func=mdp.per_gait_progress_curriculum,
        params={
            "gait_id": 0,
            "progress_min": 0.0625,
            "progress_max": 1.0,
            "max_fall_rate": 0.40,
            "min_track_quality": 0.2,
            "step_size": 0.0125,
        },
    )
    gait_progress_pace = CurrTerm(
        func=mdp.per_gait_progress_curriculum,
        params={
            "gait_id": 1,
            "progress_min": 0.0625,
            "progress_max": 1.0,
            "max_fall_rate": 0.40,
            "min_track_quality": 0.2,
            "step_size": 0.0125,
        },
    )
    gait_progress_bound = CurrTerm(
        func=mdp.per_gait_progress_curriculum,
        params={
            "gait_id": 2,
            "progress_min": 0.0625,
            "progress_max": 1.0,
            "max_fall_rate": 0.40,
            "min_track_quality": 0.2,
            "step_size": 0.0125,
        },
    )
    gait_progress_pronk = CurrTerm(
        func=mdp.per_gait_progress_curriculum,
        params={
            "gait_id": 3,
            "progress_min": 0.0625,
            "progress_max": 1.0,
            "max_fall_rate": 0.40,
            "min_track_quality": 0.2,
            "step_size": 0.0125,
        },
    )


@configclass
class AygRoughWTWEnvCfgV1(AygRoughWTWEnvCfg):
    """v1 rough env: inherits v0 rough setup, then swaps rewards/curriculum and
    recalibrates weights for additive summation."""

    def __post_init__(self):
        # v0 rough setup first (robot, actions, commands, v0 reward weights — all
        # overwritten below; cost is one __post_init__ pass).
        super().__post_init__()

        # ----- swap reward & curriculum managers -----
        self.rewards = RewardsCfgV1()
        self.curriculum = CurriculumCfgV1()

        # ----- v1 observation tweaks -----
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.history_length = None
        self.observations.policy.flatten_history_dim = False
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_vel.scale = 0.05

        # ----- v1 reward weights (additive recalibration) -----
        # Multiplicative-to-additive scaling factor: sigma_final * r_add (per step)
        # ~= 20 * 0.034 ~= 0.68. Applied to former exp-negative penalties.
        additive_scale = 0.2

        # Gait reward: per-env scale = gait_full_magnitude * per_gait_progress[gait_id].
        # `gait_full_magnitude` is decoupled from `additive_scale` so penalty scaling
        # doesn't disturb gait pressure.
        gait_full_magnitude = -4.0
        self.rewards.gait.weight = 1.0
        self.rewards.gait.params["gait_full_magnitude"] = gait_full_magnitude

        # Task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Former exp-negative terms (now additive penalties)
        self.rewards.footswing_height.weight = -0.0
        self.rewards.foot_clearance.weight = -150.0 * additive_scale
        self.rewards.base_height_l2.weight = -150.0 * additive_scale
        self.rewards.raibert_heuristic.weight = -0.0
        self.rewards.feet_slip.weight = -0.04 * additive_scale
        self.rewards.undesired_contacts.weight = -10.0 * additive_scale
        self.rewards.stand_when_zero_command.weight = -1.0 * additive_scale
        self.rewards.stand_still_when_zero_command.weight = -0.1 * additive_scale
        self.rewards.zero_vel_when_zero_command.weight = -0.0 * additive_scale
        self.rewards.zero_ang_vel_when_zero_command.weight = -0.0 * additive_scale
        self.rewards.orientation_control.weight = -40.0 * additive_scale

        # Native additive penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.joint_deviation_l1.weight = -0.0
        self.rewards.joint_deviation_l2.weight = -0.3
        self.rewards.joint_vel_l2.weight = -1.0e-3
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -2.0e-4
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01


@configclass
class AygRoughWTWEnvCfgV1_PLAY(AygRoughWTWEnvCfgV1):
    def __post_init__(self):
        super().__post_init__()

        # smaller scene for play
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


@configclass
class AygFlatWTWEnvCfgV1(AygRoughWTWEnvCfgV1):
    def __post_init__(self):
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
        # no terrain curriculum on flat ground
        self.curriculum.terrain_levels = None


@configclass
class AygFlatWTWEnvCfgV1_PLAY(AygFlatWTWEnvCfgV1):
    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
