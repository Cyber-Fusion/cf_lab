# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ManagerBasedRLEnv with WTW reward composition.

Total reward is computed as:
    r_total = r_task * exp(c_aux * r_aux)

where r_task is the sum of task (positive-weight) reward terms and r_aux is the
sum of auxiliary (negative-weight) penalty terms. c_aux = 0.02.

This ensures the agent is always rewarded for task progress, with auxiliary
penalties only modulating the magnitude — penalties can never overwhelm task reward.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


# Task reward term names (positive task rewards)
_TASK_TERM_NAMES = {"track_lin_vel_xy_exp", "track_ang_vel_z_exp"}

# c_aux = 0.02
_C_AUX = 0.02


class WTWRewardManager(RewardManager):
    """RewardManager implementing the WTW reward composition.

    Instead of a linear sum, rewards are split by term name:
      - Task terms (velocity tracking) → r_task (summed)
      - Auxiliary terms (all others) → r_aux (summed, typically negative)

    Total: r_task * exp(c_aux * r_aux)
    """

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        r_task = torch.zeros_like(self._reward_buf)
        r_aux = torch.zeros_like(self._reward_buf)

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

            # Classify by term name, not by value sign
            if name in _TASK_TERM_NAMES:
                r_task += value
            else:
                r_aux += value

            # Update episodic sum and per-step reward (for logging)
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

        # WTW formula: r_task * exp(c_aux * r_aux)
        self._reward_buf = r_task * torch.exp(_C_AUX * r_aux)
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
