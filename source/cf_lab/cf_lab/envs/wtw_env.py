# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ManagerBasedRLEnv with exp_negative reward computation support."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager

from cf_lab.tasks.manager_based.walk_these_ways.wtw_env_cfg import RewardType


class ExpNegativeRewardManager(RewardManager):
    """RewardManager with exp_negative reward type support.

    When reward_type is "exp_negative", the total reward is computed as:
        R = sum(additive_terms * dt) * exp(sum(exp_negative_terms * dt) * sigma)

    sigma is annealed externally via curriculum (anneal_sigma_exp_neg).
    Early training: sigma is low, exp gate ~ 1, policy focuses on velocity tracking.
    Late training: sigma is high, exp gate suppresses reward when behavior is poor.
    """

    def __init__(self, cfg, env, sigma=1.0):
        super().__init__(cfg, env)
        self.sigma = sigma

    def compute(self, dt: float) -> torch.Tensor:
        # R = sum(additive * dt) * exp(sum(exp_negative * dt) * sigma)
        self._reward_buf[:] = 0.0
        negative_reward = torch.zeros_like(self._reward_buf)

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

            # route based on per-term reward type
            reward_type = getattr(term_cfg, "reward_type", None)
            if reward_type == RewardType.EXP_NEGATIVE:
                negative_reward += value
            else:
                self._reward_buf += value

            # update episodic sum
            self._episode_sums[name] += value

            # update step reward
            self._step_reward[:, term_idx] = value / dt

        # apply exponential negative scaling
        self._reward_buf *= torch.exp(negative_reward * self.sigma)
        return self._reward_buf


class WTWManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with exp_negative reward support.

    This env subclass replaces the default RewardManager with ExpNegativeRewardManager
    after initialization, enabling the exp_negative reward computation mode used by
    Walk-These-Ways environments.
    """

    def load_managers(self):
        super().load_managers()
        # Replace reward manager with custom one that supports exp_negative
        del self.reward_manager
        self.reward_manager = ExpNegativeRewardManager(
            self.cfg.rewards,
            self,
            sigma=getattr(self.cfg, "sigma", 1.0),
        )
