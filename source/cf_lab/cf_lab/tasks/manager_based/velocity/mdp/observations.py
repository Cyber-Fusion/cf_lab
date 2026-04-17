# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Spot-like locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


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


def friction_coefficients(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Per-env mean static and dynamic friction across all shapes. Shape: (num_envs, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # material_properties shape: (num_envs, num_shapes, 3) -> (static, dynamic, restitution)
    materials = asset.root_physx_view.get_material_properties()
    return materials[:, :, :2].mean(dim=1).to(env.device)


def body_mass(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Current mass of the selected bodies (reflects mass randomization).

    Returns shape (num_envs, len(body_ids)). When ``asset_cfg.body_names`` matches
    a single body (e.g., the base), the observation is (num_envs, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)
    return masses[:, asset_cfg.body_ids]


def base_external_force_torque(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """External force + torque applied to the selected body (typically the base).

    Returns the concatenated force (3) and torque (3) vectors per selected body.
    Shape: (num_envs, 6 * len(body_ids)).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :]
    torques = asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids, :]
    return torch.cat([forces.reshape(env.num_envs, -1), torques.reshape(env.num_envs, -1)], dim=-1)
