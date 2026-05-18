# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Blind student baseline: proprio + last_action only, no depth camera in the scene.

Mirrors :class:`AygRoughStudentEnvCfg` for the teacher observation group (so DAgger
supervision still works against the locked Phase 1 teacher), but the student `policy`
group drops both privileged signals (base_lin_vel, height_scan) and the depth term.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from .rough_env_cfg import AygRoughEnvCfg, AygRoughEnvCfg_PLAY


@configclass
class BlindStudentObservationsCfg(ObservationsCfg):
    """Two-group obs for blind-student DAgger.

    `policy` is what the student network consumes at training and deploy time: proprio +
    last action only — no base_lin_vel (real robot can't measure cleanly), no height_scan
    (privileged), no depth.

    `teacher` mirrors what the locked Phase 1 teacher was trained on and is used only at
    training time to generate supervision targets via the frozen teacher.
    """

    @configclass
    class TeacherCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


def _apply_blind_student_overlay(cfg) -> None:
    # Strip privileged terms from the student policy group; do NOT add a depth camera.
    # All remaining terms are 1-D vectors → concatenate_terms=True yields a single
    # tensor that RSL-RL's StudentTeacher policy consumes directly. (Previously
    # False to match the custom DAgger trainer's Dict-obs contract.)
    cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.height_scan = None
    cfg.observations.policy.concatenate_terms = True


@configclass
class AygRoughStudentBlindEnvCfg(AygRoughEnvCfg):
    observations: BlindStudentObservationsCfg = BlindStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_blind_student_overlay(self)


@configclass
class AygRoughStudentBlindEnvCfg_PLAY(AygRoughEnvCfg_PLAY):
    observations: BlindStudentObservationsCfg = BlindStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_blind_student_overlay(self)
