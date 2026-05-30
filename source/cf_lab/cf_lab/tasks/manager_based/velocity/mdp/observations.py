# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Spot-like locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


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


class DepthHistoryStrided(ManagerTermBase):
    """Strided temporal stack of depth frames from a single camera.

    Maintains an internal circular buffer of the last ``(num_frames - 1) * stride + 1``
    raw depth frames (kept on-device, NOT placed in the rollout storage) and returns
    only ``num_frames`` frames sampled every ``stride`` control steps, **newest first**.

    Rationale (see ``rough_student_env_cfg.py``): the env steps at 50 Hz while the D555
    refreshes at 30 Hz, so a *contiguous* stack contains duplicate frames and spans only
    a fraction of a second. With ``stride=4`` (12.5 Hz < 30 Hz) every returned frame is a
    distinct camera image and a 10-frame window spans ~0.72 s — matching what the real
    robot's inference node will reproduce by buffering depth at the control rate.

    The ring advances exactly once per control step (gated on ``common_step_counter``),
    so extra ``ObservationManager.compute()`` calls (recording, reset re-computation)
    cannot corrupt the stride timing.

    Output shape: ``(num_envs, num_frames * H * W)`` — depth in meters with NaN/Inf zeroed
    and values clamped to ``[0, far_clip]``. The vision student reshapes this back to
    ``(N, num_frames, H, W)``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._num_frames = int(cfg.params.get("num_frames", 10))
        self._stride = int(cfg.params.get("stride", 4))
        self._far_clip = float(cfg.params.get("far_clip", 9.0))
        self._ring_len = (self._num_frames - 1) * self._stride + 1
        self._buffer: torch.Tensor | None = None
        self._pos = 0
        self._last_step = -1

    def reset(self, env_ids: torch.Tensor | None = None):
        if self._buffer is None:
            return
        if env_ids is None:
            self._buffer.zero_()
        else:
            self._buffer[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        num_frames: int = 10,
        stride: int = 4,
        far_clip: float = 9.0,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        img = sensor.data.output[data_type]
        if img.dim() == 4:  # (N, H, W, C) -> drop the singleton channel
            img = img[..., 0]
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, self._far_clip)
        num_envs, height, width = img.shape

        if self._buffer is None:
            self._buffer = torch.zeros(num_envs, self._ring_len, height, width, device=img.device)
            self._pos = 0

        # Advance the ring head once per control step; otherwise just refresh the head.
        step = int(env.common_step_counter)
        if step != self._last_step:
            self._last_step = step
            self._pos = (self._pos - 1) % self._ring_len
        self._buffer[:, self._pos] = img

        offsets = torch.arange(self._num_frames, device=img.device) * self._stride
        idx = (self._pos + offsets) % self._ring_len
        frames = self._buffer.index_select(1, idx)  # (N, num_frames, H, W), newest first
        return frames.reshape(num_envs, -1)


class ProprioHistoryStrided(ManagerTermBase):
    """Strided temporal stack of a compact proprioceptive vector.

    Aligned 1:1 (same ``num_frames`` / ``stride`` / ordering) with
    :class:`DepthHistoryStrided` so the vision student can infer inter-frame ego-motion
    (how the body moved between depth frames) — the signal needed to register a moving
    camera into a usable spatial estimate.

    Per-frame vector (21 dims):
        base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_vel_rel(12)

    Output shape: ``(num_envs, num_frames * 21)``, newest first.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._num_frames = int(cfg.params.get("num_frames", 10))
        self._stride = int(cfg.params.get("stride", 4))
        self._command_name = cfg.params.get("command_name", "base_velocity")
        self._ring_len = (self._num_frames - 1) * self._stride + 1
        self._buffer: torch.Tensor | None = None
        self._pos = 0
        self._last_step = -1

    def reset(self, env_ids: torch.Tensor | None = None):
        if self._buffer is None:
            return
        if env_ids is None:
            self._buffer.zero_()
        else:
            self._buffer[env_ids] = 0.0

    def _proprio(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.cat(
            [
                base_mdp.base_ang_vel(env),
                base_mdp.projected_gravity(env),
                base_mdp.generated_commands(env, command_name=self._command_name),
                base_mdp.joint_vel_rel(env),
            ],
            dim=-1,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        num_frames: int = 10,
        stride: int = 4,
        command_name: str = "base_velocity",
    ) -> torch.Tensor:
        vec = self._proprio(env)  # (N, 21)
        num_envs, dim = vec.shape

        if self._buffer is None:
            self._buffer = torch.zeros(num_envs, self._ring_len, dim, device=vec.device)
            self._pos = 0

        step = int(env.common_step_counter)
        if step != self._last_step:
            self._last_step = step
            self._pos = (self._pos - 1) % self._ring_len
        self._buffer[:, self._pos] = vec

        offsets = torch.arange(self._num_frames, device=vec.device) * self._stride
        idx = (self._pos + offsets) % self._ring_len
        frames = self._buffer.index_select(1, idx)  # (N, num_frames, 21), newest first
        return frames.reshape(num_envs, -1)
