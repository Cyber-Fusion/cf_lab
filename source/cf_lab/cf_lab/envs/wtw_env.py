# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ManagerBasedRLEnv with hybrid WTW reward composition.

Hybrid reward formula:
    r_total = r_additive + r_gait_task * exp(c_aux * r_gait_aux)

- r_additive: classical weighted sum of velocity tracking, action penalties,
  orientation, joint penalties, etc. Direct gradients, no dampening.
- r_gait_task: gait contact tracking reward (the "gait" term).
- r_gait_aux: gait-specific penalties (footswing_height) that modulate gait quality.
- c_aux: exponential coefficient, annealed via curriculum.

This gives velocity tracking a direct, strong gradient signal while using the
exponential formulation only for gait-specific shaping.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


# Gait task terms — go through exp formulation
_GAIT_TASK_TERMS = {"gait"}

# Gait auxiliary terms — penalties that modulate gait quality through exp
_GAIT_AUX_TERMS = {"footswing_height"}

# Everything else is additive (velocity tracking, action penalties, orientation, etc.)

# Default c_aux starting value (annealed via curriculum)
_C_AUX_DEFAULT = 0.02


class WTWRewardManager(RewardManager):
    """RewardManager implementing hybrid WTW reward composition.

    Rewards are split into three categories:
      - Additive terms (velocity tracking, penalties) → direct weighted sum
      - Gait task terms ("gait") → r_gait_task
      - Gait aux terms ("footswing_height") → r_gait_aux

    Total: r_additive + r_gait_task * exp(c_aux * r_gait_aux)

    The c_aux coefficient can be annealed via curriculum (start low → end high)
    to let the policy learn basic locomotion first, then refine gait quality.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.c_aux = _C_AUX_DEFAULT

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        r_additive = torch.zeros_like(self._reward_buf)
        r_gait_task = torch.zeros_like(self._reward_buf)
        r_gait_aux = torch.zeros_like(self._reward_buf)

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

            # Classify into three categories
            if name in _GAIT_TASK_TERMS:
                r_gait_task += value
            elif name in _GAIT_AUX_TERMS:
                r_gait_aux += value
            else:
                r_additive += value

            # Update episodic sum and per-step reward (for logging)
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

        # Hybrid formula: additive base + gait with exp modulation
        self._reward_buf = r_additive + r_gait_task * torch.exp(self.c_aux * r_gait_aux)
        return self._reward_buf


class WTWManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with WTW reward composition.

    Replaces the default RewardManager with WTWRewardManager after initialization.
    """

    def load_managers(self):
        super().load_managers()
        # Replace reward manager with WTW version
        del self.reward_manager
        self.reward_manager = WTWRewardManager(self.cfg.rewards, self)
