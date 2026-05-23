# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bilateral (left-right) symmetry augmentation for AYG quadruped WTW environment.

Implements PPOaug data augmentation following:
    "Leveraging Symmetry in RL-based Legged Locomotion Control"
    Zhi Su et al., IROS 2024, arXiv:2403.17320

The AYG robot has sagittal (left-right) symmetry under the reflection group
C2 = {e, g_s} where g_s is the sagittal mirror (y -> -y). This module
provides the group representations rho_S (observation) and rho_A (action)
needed for PPOaug data augmentation (Section IV.A of the paper).

Derivation from AYG URDF (ayg_description/urdf/ayg.urdf):

  Joint ordering (Isaac Sim, matches URDF declaration order):
      [0] LF_HAA  [1] LF_HFE  [2] LF_KFE    (left front)
      [3] RF_HAA  [4] RF_HFE  [5] RF_KFE    (right front)
      [6] LH_HAA  [7] LH_HFE  [8] LH_KFE    (left hind)
      [9] RH_HAA  [10] RH_HFE [11] RH_KFE   (right hind)

  Axis directions from URDF:
      HAA front: (1, 0, 0)    HAA hind: (-1, 0, 0)   -> same within L-R pairs -> NEGATE on swap
      HFE all:   (0, -1, 0)                            -> same across all legs  -> KEEP sign
      KFE all:   (0, -1, 0)                            -> same across all legs  -> KEEP sign

  Per-timestep observation layout (61 dims, flat WTW without height_scan):
      [0:3]   base_lin_vel       -> [vx, -vy, vz]
      [3:6]   base_ang_vel       -> [-wx, wy, -wz]
      [6:9]   projected_gravity  -> [gx, -gy, gz]
      [9:12]  velocity_commands  -> [vx, -vy, -wz]
      [12:16] gait_phase (4D)   -> swap LF<->RF, LH<->RH
      [16:25] gait_command (9D) -> only negate roll (idx 24)
      [25:37] joint_pos (12)    -> swap legs + negate HAA
      [37:49] joint_vel (12)    -> swap legs + negate HAA
      [49:61] actions (12)      -> swap legs + negate HAA

  Gait command offsets are self-symmetric for canonical gaits with
  binary_phases=True (trot/pace/bound/pronk all satisfy g_s . cmd = cmd).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# ============================================================================
# Mirror index tables — derived from AYG URDF + paper Section II.A
# ============================================================================

_OBS_STEP_SIZE = 61

# Joint mirror: swap LF<->RF and LH<->RH
_JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
_JOINT_SIGN = [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]

# Full observation permutation: obs_mirror[i] = _SIGN[i] * obs[_PERM[i]]
_PERM = (
    # base_lin_vel [0:3] — no reordering
    [0, 1, 2]
    # base_ang_vel [3:6] — no reordering
    + [3, 4, 5]
    # projected_gravity [6:9] — no reordering
    + [6, 7, 8]
    # velocity_commands [9:12] — no reordering
    + [9, 10, 11]
    # gait_phase [12:16] — swap LF(12)<->RF(13), LH(14)<->RH(15)
    + [13, 12, 15, 14]
    # gait_command [16:25] — no reordering (canonical gaits self-symmetric)
    + [16, 17, 18, 19, 20, 21, 22, 23, 24]
    # joint_pos [25:37] — swap LF<->RF, LH<->RH (JOINT_PERM + offset 25)
    + [28, 29, 30, 25, 26, 27, 34, 35, 36, 31, 32, 33]
    # joint_vel [37:49] — same swap (JOINT_PERM + offset 37)
    + [40, 41, 42, 37, 38, 39, 46, 47, 48, 43, 44, 45]
    # actions [49:61] — same swap (JOINT_PERM + offset 49)
    + [52, 53, 54, 49, 50, 51, 58, 59, 60, 55, 56, 57]
)

_SIGN = (
    # base_lin_vel: [vx, -vy, vz] — lateral velocity flips
    [1.0, -1.0, 1.0]
    # base_ang_vel: [-wx, wy, -wz] — roll rate and yaw rate flip
    + [-1.0, 1.0, -1.0]
    # projected_gravity: [gx, -gy, gz] — lateral gravity component flips
    + [1.0, -1.0, 1.0]
    # velocity_commands: [vx, -vy, -wz] — lateral vel and yaw cmd flip
    + [1.0, -1.0, -1.0]
    # gait_phase: all +1 (swap handled by permutation, sign preserved)
    + [1.0, 1.0, 1.0, 1.0]
    # gait_command: only roll (idx 8 within gait_cmd = idx 24 global) negates
    + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
    # joint_pos: [-HAA, +HFE, +KFE] x 4 legs
    + [-1.0, 1.0, 1.0] * 4
    # joint_vel: same pattern
    + [-1.0, 1.0, 1.0] * 4
    # actions: same pattern
    + [-1.0, 1.0, 1.0] * 4
)

# Estimator ground-truth: base_lin_vel only (3 dims)
_EST_SIGN = [1.0, -1.0, 1.0]

# ============================================================================
# Tensor cache (lazily initialized per device)
# ============================================================================

_cache: dict[str, torch.Tensor] = {}


def _get_cached(name: str, values: list, device: torch.device) -> torch.Tensor:
    key = f"{name}_{device}"
    if key not in _cache:
        dtype = torch.long if "perm" in name else torch.float32
        _cache[key] = torch.tensor(values, dtype=dtype, device=device)
    return _cache[key]


# ============================================================================
# Mirror functions
# ============================================================================


def _mirror_obs_flat(obs_flat: torch.Tensor) -> torch.Tensor:
    """Mirror a flattened observation tensor (handles history stacking).

    With history_length=N and flatten_history_dim=True, the observation is
    (batch, N * step_size). We reshape to (batch, N, step_size), apply the
    per-step mirror, and reshape back.

    Args:
        obs_flat: shape (batch, total_obs_dim) where total_obs_dim = step_size * num_history.

    Returns:
        Mirrored observation with same shape.
    """
    device = obs_flat.device
    batch, total_dim = obs_flat.shape
    num_steps = total_dim // _OBS_STEP_SIZE

    perm = _get_cached("obs_perm", _PERM, device)
    sign = _get_cached("obs_sign", _SIGN, device)

    obs = obs_flat.view(batch, num_steps, _OBS_STEP_SIZE)
    mirrored = obs[:, :, perm] * sign
    return mirrored.view(batch, total_dim)


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror joint-space actions (12 dims)."""
    perm = _get_cached("joint_perm", _JOINT_PERM, actions.device)
    sign = _get_cached("joint_sign", _JOINT_SIGN, actions.device)
    return actions[:, perm] * sign


def _mirror_estimator_gt(est: torch.Tensor) -> torch.Tensor:
    """Mirror estimator ground truth (base_lin_vel: [vx, -vy, vz])."""
    sign = _get_cached("est_sign", _EST_SIGN, est.device)
    return est * sign


# ============================================================================
# Public API — RSL-RL symmetry_cfg interface
# ============================================================================


@torch.no_grad()
def bilateral_symmetry_augmentation(
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    env,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Bilateral (left-right) symmetry augmentation for AYG WTW environment.

    Implements PPOaug (Section IV.A of the paper): concatenates original and
    mirrored data along the batch dimension, doubling the effective batch size.

    The mirrored transitions reuse the same reward r because all WTW reward
    terms are G-invariant (verified: squared norms, symmetric foot sums).

    Compatible with RSL-RL's RslRlSymmetryCfg interface. Called during
    PPO.update() for each mini-batch.

    Args:
        obs: Observation TensorDict with keys "policy", "critic", "estimator_gt".
             Can be None when only mirroring actions (mirror loss path).
        actions: Action tensor of shape (batch, 12). Can be None.
        env: The vectorized environment (unused, required by RSL-RL interface).

    Returns:
        (augmented_obs, augmented_actions) with batch dimension doubled.
        Either can be None if the corresponding input was None.
    """
    aug_obs = None
    aug_actions = None

    if obs is not None:
        mirror_dict = {}
        for key in obs.keys():
            tensor = obs[key]
            if key in ("policy", "critic"):
                mirrored = _mirror_obs_flat(tensor)
            elif key == "estimator_gt":
                mirrored = _mirror_estimator_gt(tensor)
            else:
                mirrored = tensor.clone()
            mirror_dict[key] = torch.cat([tensor, mirrored], dim=0)
        aug_obs = TensorDict(mirror_dict, batch_size=[tensor.shape[0] * 2])

    if actions is not None:
        aug_actions = torch.cat([actions, _mirror_actions(actions)], dim=0)

    return aug_obs, aug_actions
