# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Spot-like locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCasterCamera
from isaaclab.utils.math import unproject_depth

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_heights(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the z-position of each foot body in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2]


def foot_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the current air time for each foot."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Return binary contact state for each foot (1.0 = in contact)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return contact.float()


def foot_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the net contact force magnitude for each foot."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.norm(forces, dim=-1)


def front_depth_pointcloud(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_range: float = 0.2,
    max_range: float = 5.0,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """Sparse, noisy forward depth point cloud in the sensor frame with a per-point validity mask.

    Reads a low-resolution depth image from a :class:`RayCasterCamera` and
    unprojects it to 3D points using the camera intrinsics.

    A ray is valid only if its depth is finite **and strictly inside**
    ``(min_range, max_range)``. The strict upper bound matters: with
    ``depth_clipping_behavior="max"`` the underlying ray caster reports
    ``depth == max_range`` for rays that hit nothing within range, and treating
    those as valid would paint a phantom wall at exactly ``max_range``.

    Per-point output is ``(x, y, z, mask)`` with ``mask = 1`` for valid rays and
    ``mask = 0`` otherwise; invalid points have ``(x, y, z) = (0, 0, 0)`` so the
    network can distinguish "no measurement" from a real near-origin hit. Noise
    is added on points before re-zeroing invalids so it never leaks into masked
    entries.

    The output is flattened to ``(num_envs, H * W * 4)``.
    """
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    # (N, H, W, 1) -> (N, H, W)
    depth = sensor.data.output["distance_to_image_plane"].squeeze(-1)
    intrinsics = sensor.data.intrinsic_matrices

    valid = torch.isfinite(depth) & (depth >= min_range) & (depth < max_range)
    depth_clean = torch.where(valid, depth, torch.zeros_like(depth))

    # (N, H*W, 3) sensor-frame points
    points = unproject_depth(depth_clean, intrinsics, is_ortho=True)

    valid_flat = valid.reshape(valid.shape[0], -1, 1)
    if noise_std > 0.0:
        points = points + noise_std * torch.randn_like(points)
    points = torch.where(valid_flat, points, torch.zeros_like(points))

    # Append per-point validity mask -> (N, H*W, 4)
    mask = valid_flat.to(points.dtype)
    points = torch.cat([points, mask], dim=-1)

    return points.reshape(points.shape[0], -1)
