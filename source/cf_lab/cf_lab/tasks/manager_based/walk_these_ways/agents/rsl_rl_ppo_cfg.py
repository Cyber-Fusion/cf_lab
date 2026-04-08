# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class AygRoughWTWPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 21
    max_iterations = 30000
    save_interval = 500
    experiment_name = "ayg_wtw_rough"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,  # was 0.005; reduced to control noise growth over long runs
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # was 0.008; relaxed to prevent LR collapse
        max_grad_norm=1.0,
    )


@configclass
class AygFlatWTWPPORunnerCfg(AygRoughWTWPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.num_steps_per_env = 24
        self.save_interval = 500
        self.experiment_name = "ayg_wtw_flat"
        self.empirical_normalization = False

        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=False,
                use_mirror_loss=False,
                data_augmentation_func="cf_lab.tasks.manager_based.walk_these_ways.mdp.symmetry:bilateral_symmetry_augmentation",
                mirror_loss_coeff=1.0,
            ),
        )
