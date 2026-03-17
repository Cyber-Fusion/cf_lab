# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def simulation_crashed(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.5
) -> torch.Tensor:
    """Terminate when the simulation crashed.

    The simulation is considered to be crashed if the root position of the actor is too far away from
    the initial position.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.logical_or(
        torch.isnan(asset.data.root_link_lin_vel_w[:, 0]),
        torch.isinf(asset.data.root_link_lin_vel_w[:, 0])
    )
