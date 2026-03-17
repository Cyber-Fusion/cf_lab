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
        R = sum(positive_terms * dt) * exp(sum(negative_terms) / sigma)

    Positive terms are scaled by dt (standard RL reward accumulation).
    Negative terms go into the exponential WITHOUT dt, so their magnitude
    is independent of the simulation timestep. sigma is annealed from
    sigma_start to sigma_end over total_anneal_steps to allow gentle
    penalties early in training and aggressive enforcement later.
    """

    reward_type: str = "exp_negative"
    sigma_min = 1
    sigma_max = 5
    total_anneal_steps: int = 1000

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._global_step = 0

    def compute(self, dt: float) -> torch.Tensor:
        if self.reward_type != "exp_negative":
            return super().compute(dt)

        self._global_step += 1./24
        progress = min((self._global_step / (0.8*self.total_anneal_steps))**3, 1.0)
        sigma_exp_neg = self.sigma_min + (self.sigma_max - self.sigma_min) * progress

        # exp_negative: R = sum(positive * dt) * exp(sum(negative) / sigma)
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
        self._reward_buf *= torch.exp(negative_reward * sigma_exp_neg)
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
