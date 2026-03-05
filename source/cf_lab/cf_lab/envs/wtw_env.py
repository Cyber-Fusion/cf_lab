# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ManagerBasedRLEnv with exp_negative reward computation support."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


class ExpNegativeRewardManager(RewardManager):
    """RewardManager with exp_negative reward type support.

    When reward_type is "exp_negative", the total reward is computed as:
        R = sum(positive_terms) * exp(sum(negative_terms) / negative_reward_scale)

    This encourages the agent to maximize positive rewards while minimizing negative ones,
    with the exponential term smoothly scaling down positive rewards when negatives are large.
    """

    reward_type: str = "exp_negative"
    negative_reward_scale: float = 0.02

    def compute(self, dt: float) -> torch.Tensor:
        if self.reward_type != "exp_negative":
            return super().compute(dt)

        # exp_negative: R = sum(positive) * exp(sum(negative) / scale)
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

            # separate positive and negative components
            positive_reward = torch.clip(value, min=0.0)
            negative_reward += torch.clip(value, max=0.0)
            self._reward_buf += positive_reward

            # update episodic sum
            self._episode_sums[name] += value

            # update step reward
            self._step_reward[:, term_idx] = value / dt

        # apply exponential negative scaling
        self._reward_buf *= torch.exp(negative_reward / self.negative_reward_scale)
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
        self.reward_manager = ExpNegativeRewardManager(self.cfg.rewards, self)
