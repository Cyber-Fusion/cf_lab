# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL distillation runner configs for the AYG student environments.

These configs feed Isaac Lab's stock `scripts/rsl_rl/train.py` via the
`rsl_rl_distillation_cfg_entry_point` registry key. RSL-RL's
``StudentTeacher.load_state_dict`` auto-detects an ``ActorCritic`` checkpoint
(``actor.*`` keys) and extracts only the actor weights into the teacher
network, so the locked Phase 1 teacher (``model_9999.pt``) loads directly
without any conversion.

Importing this module also imports ``cf_lab.learning`` which has the side
effect of registering ``StudentTeacherVision`` in
``rsl_rl.runners.distillation_runner``'s namespace so the runner's
``eval(class_name)`` lookup resolves it.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

import cf_lab.learning  # noqa: F401  — side-effect: registers StudentTeacherVision


@configclass
class _VisionStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Extends the stock cfg with the depth/ego shape constants the CNN needs.

    Extra fields propagate to ``StudentTeacherVision.__init__`` via the
    runner's ``**self.policy_cfg`` splat.
    """

    # Layout of the concatenated student obs (see rough_student_env_cfg.py for
    # the term order pinning the slice boundary):
    #   ego_dim = base_ang_vel + projected_gravity + velocity_commands
    #           + joint_pos + joint_vel + actions = 3+3+3+12+12+12 = 45
    #   depth = history_length * height * width = 10 * 45 * 80 = 36000
    ego_dim: int = 45
    depth_t: int = 10
    depth_h: int = 45
    depth_w: int = 80
    depth_latent_dim: int = 64
    ego_latent_dim: int = 128
    head_hidden_dims: list[int] = [256, 128]


@configclass
class AygRoughStudentVisionDistillationCfg(RslRlDistillationRunnerCfg):
    """Distillation runner for the vision student (D555-equivalent depth, 30° pitched)."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    # Share the teacher's experiment folder so ``get_checkpoint_path`` can find
    # ``logs/rsl_rl/ayg_rough/Teacher(baseline)/model_9999.pt`` directly via the
    # ``--load_run`` / ``--checkpoint`` CLI args.
    experiment_name = "ayg_rough"
    # Map RSL-RL's two policy roles onto the env's two obs groups:
    # student net consumes env.observations.policy (45-dim ego + 10-frame depth stack),
    # teacher net consumes env.observations.teacher (235-dim ego + height_scan).
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = _VisionStudentTeacherCfg(
        class_name="StudentTeacherVision",
        init_noise_std=0.1,
        noise_std_type="scalar",
        # Phase 1 teacher was trained with empirical_normalization=False — match it.
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        # Teacher hidden dims MUST match the locked checkpoint
        # (AygRoughPPORunnerCfg.policy.actor_hidden_dims).
        teacher_hidden_dims=[512, 256, 128],
        # student_hidden_dims is required by the parent cfg but the
        # VisionStudentNet ignores it (we use head_hidden_dims for the fused MLP head).
        student_hidden_dims=[256, 256, 256],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        loss_type="mse",
        optimizer="adam",
    )
