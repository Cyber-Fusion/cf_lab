# Copyright (c) 2022-2026, The cf_lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RSL-RL wrapper that filters out zero-weight reward terms from logging."""

import torch

from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class RslRlVecEnvWrapperFiltered(RslRlVecEnvWrapper):
    """Extends RslRlVecEnvWrapper to filter zero-weight reward terms from extras["log"].

    This prevents reward terms with weight=0.0 from cluttering the training logs.
    """

    def __init__(self, env, clip_actions: float | None = None):
        super().__init__(env, clip_actions=clip_actions)
        # Cache the set of zero-weight reward term names
        self._zero_weight_keys: set[str] = set()
        if hasattr(self.unwrapped, "reward_manager"):
            rm = self.unwrapped.reward_manager
            for name, cfg in zip(rm._term_names, rm._term_cfgs):
                if cfg.weight == 0.0:
                    self._zero_weight_keys.add(f"Episode_Reward/{name}")

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)
        if "log" in extras and self._zero_weight_keys:
            extras["log"] = {k: v for k, v in extras["log"].items() if k not in self._zero_weight_keys}
        return obs, rew, dones, extras
