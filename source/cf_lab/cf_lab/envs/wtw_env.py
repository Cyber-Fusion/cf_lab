# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ManagerBasedRLEnv with exp-negative WTW reward composition.

Reward formula:
    R = sum(additive_terms * dt) * exp(sum(exp_negative_terms * dt) * sigma)

- additive_terms: velocity tracking, joint/action penalties — direct weighted sum.
- exp_negative_terms: gait tracking, orientation, height, contacts — modulate the
  additive reward through an exponential gate.
- sigma: anneals from sigma_min to sigma_max over training, making the exp gate
  progressively stricter.

Early training (sigma low): exp gate ≈ 1, policy focuses on velocity tracking.
Late training (sigma high): exp gate suppresses reward when behavior is poor.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


# Terms that go through the exp-negative pathway
_EXP_NEGATIVE_TERMS = {
    "gait",
    "orientation_control",
    "base_height_l2",
    "feet_slip",
    "undesired_contacts",
    "foot_clearance",
    "raibert_heuristic",
}

# Default sigma starting value
_SIGMA_DEFAULT = 1.0


class ExpNegativeRewardManager(RewardManager):
    """RewardManager implementing the exp-negative reward formula.

    R = sum(additive_terms * dt) * exp(sum(exp_negative_terms * dt) * sigma)

    sigma is annealed via curriculum from sigma_min to sigma_max.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sigma = _SIGMA_DEFAULT

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        r_additive = torch.zeros_like(self._reward_buf)
        r_exp_neg = torch.zeros_like(self._reward_buf)

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

            if name in _EXP_NEGATIVE_TERMS:
                r_exp_neg += value
            else:
                r_additive += value

            # Update episodic sum and per-step reward (for logging)
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

        # Exp-negative formula: additive base * exp(exp_neg_sum * sigma)
        self._reward_buf = r_additive * torch.exp(r_exp_neg * self.sigma)
        return self._reward_buf


class WTWManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with exp-negative WTW reward composition.

    Replaces the default RewardManager with ExpNegativeRewardManager after initialization.
    """

    def load_managers(self):
        super().load_managers()
        # Replace reward manager with exp-negative version
        del self.reward_manager
        self.reward_manager = ExpNegativeRewardManager(self.cfg.rewards, self)
