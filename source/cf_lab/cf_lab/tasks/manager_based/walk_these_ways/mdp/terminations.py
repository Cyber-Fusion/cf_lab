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


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the robot tilts too far from upright.

    Checks the z-component of the projected gravity vector in body frame.
    When upright, projected_gravity_b ≈ [0, 0, -1], so gz ≈ -1.
    When tilted by angle θ, gz = -cos(θ).
    Terminate if cos(θ) < cos(limit_angle), i.e. gz > -cos(limit_angle).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 2] is the z-component: -1 when upright, 0 when sideways, +1 when flipped
    gz = asset.data.projected_gravity_b[:, 2]
    import math
    return gz > -math.cos(limit_angle)


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
