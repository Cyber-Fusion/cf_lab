# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO algorithm extended with supervised estimator loss.

This module extends rsl_rl's PPO to add an MSE loss that trains the state
estimator to predict body velocity from observation history. The estimator
loss is added to the standard PPO loss (surrogate + value + entropy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class PPOEstimator(PPO):
    """PPO with supervised estimator loss for the WTW state estimator.

    During each update step, in addition to the standard PPO losses, computes
    an MSE loss between the estimator's velocity prediction and the ground truth
    from the 'estimator_gt' observation group.
    """

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_estimator_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            # Standard PPO forward pass
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL divergence
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = kl.mean()

                    if hasattr(self, "writer") and self.writer is not None:
                        pass
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Estimator supervised loss
            estimator_loss = torch.tensor(0.0, device=value_loss.device)
            if hasattr(self.policy, "compute_estimator_loss") and hasattr(self.policy, "estimator_loss_coef"):
                estimator_loss = self.policy.compute_estimator_loss(obs_batch)

            # Combined loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.policy.estimator_loss_coef * estimator_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += estimator_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator": mean_estimator_loss,
        }
        return loss_dict


# Register into rsl_rl namespaces
import rsl_rl.algorithms as _rsl_alg  # noqa: E402
import rsl_rl.runners.on_policy_runner as _runner_module  # noqa: E402

_rsl_alg.PPOEstimator = PPOEstimator
_runner_module.PPOEstimator = PPOEstimator
