# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with concurrent state estimator.

The state estimator is an MLP [256, 128] with ELU activations that predicts
body velocity from the observation history. Its output is concatenated to the
actor's input. The estimator is trained with both:
  1. Policy gradient (backprop through actor)
  2. Supervised MSE loss against ground truth velocity
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticEstimator(ActorCritic):
    """ActorCritic with a state estimator head.

    The estimator predicts privileged information (body velocity) from policy
    observations. The prediction is concatenated to the actor's input.
    Gradients flow through the estimator from both the policy loss and
    a supervised MSE loss against ground truth.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Estimator config
        estimator_hidden_dims: list[int] = [256, 128],
        estimator_output_dim: int = 3,
        estimator_loss_coef: float = 1.0,
        # Standard ActorCritic params
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        num_policy_obs = 0
        for obs_group in obs_groups["policy"]:
            num_policy_obs += obs[obs_group].shape[-1]

        nn.Module.__init__(self)

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.estimator_loss_coef = estimator_loss_coef

        num_actor_obs = num_policy_obs + estimator_output_dim
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]

        # Estimator: policy_obs -> predicted [vx, vy, vz]
        self.estimator = MLP(num_policy_obs, estimator_output_dim, estimator_hidden_dims, activation)
        print(f"Estimator MLP: {self.estimator}")

        # Actor: (policy_obs + estimated) -> actions
        if state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if noise_std_type == "scalar":
                nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif noise_std_type == "log":
                nn.init.constant_(self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
        else:
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _get_policy_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[g] for g in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def _run_estimator(self, policy_obs: torch.Tensor) -> torch.Tensor:
        estimated = self.estimator(policy_obs)
        return torch.cat([policy_obs, estimated], dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self._get_policy_obs(obs)
        return self._run_estimator(policy_obs)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        augmented_obs = self.get_actor_obs(obs)
        augmented_obs = self.actor_obs_normalizer(augmented_obs)
        self._update_distribution(augmented_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        augmented_obs = self.get_actor_obs(obs)
        augmented_obs = self.actor_obs_normalizer(augmented_obs)
        if self.state_dependent_std:
            return self.actor(augmented_obs)[..., 0, :]
        return self.actor(augmented_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def compute_estimator_loss(self, obs: TensorDict) -> torch.Tensor:
        """Compute supervised MSE loss between estimator output and ground truth.

        Args:
            obs: TensorDict containing all observation groups including 'estimator_gt'.

        Returns:
            Scalar MSE loss tensor.
        """
        policy_obs = self._get_policy_obs(obs)
        estimated = self.estimator(policy_obs)

        gt_groups = self.obs_groups.get("estimator_gt", [])
        if not gt_groups:
            return torch.tensor(0.0, device=estimated.device)
        gt = torch.cat([obs[g] for g in gt_groups], dim=-1)

        return nn.functional.mse_loss(estimated, gt)


# Register into rsl_rl namespaces so eval("ActorCriticEstimator") works
import rsl_rl.modules as _rsl_modules  # noqa: E402
import rsl_rl.runners.on_policy_runner as _runner_module  # noqa: E402

_rsl_modules.ActorCriticEstimator = ActorCriticEstimator
_runner_module.ActorCriticEstimator = ActorCriticEstimator
