# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticEstimatorCfg(RslRlPpoActorCriticCfg):
    """PPO actor-critic config extended with state estimator fields."""

    class_name: str = "ActorCriticEstimator"
    estimator_hidden_dims: list[int] = [256, 128]
    estimator_output_dim: int = 3  # [vx, vy, vz]
    estimator_loss_coef: float = 1.0


@configclass
class RslRlPpoEstimatorAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config that uses PPOEstimator (with supervised estimator loss)."""

    class_name: str = "PPOEstimator"


@configclass
class AygRoughWTWPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 21
    max_iterations = 30000
    save_interval = 500
    experiment_name = "ayg_wtw_rough"
    empirical_normalization = True
    policy = RslRlPpoActorCriticEstimatorCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoEstimatorAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AygFlatWTWPPORunnerCfg(AygRoughWTWPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "ayg_wtw_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
