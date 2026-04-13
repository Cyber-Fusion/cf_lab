# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right (sagittal) symmetry augmentation for AYG direct velocity environments.

AYG joint ordering in Isaac Sim (grouped by type, from articulation):

    [0] LF_HAA  [1] LH_HAA  [2] RF_HAA  [3] RH_HAA
    [4] LF_HFE  [5] LH_HFE  [6] RF_HFE  [7] RH_HFE
    [8] LF_KFE  [9] LH_KFE  [10] RF_KFE [11] RH_KFE

Direct env observation layout (Layout C):
    [0:3]   base_lin_vel
    [3:6]   base_ang_vel
    [6:9]   projected_gravity
    [9:12]  velocity_commands
    [12]    heading_target (scalar)
    [13:25] joint_pos  (12)
    [25:37] joint_vel  (12)
    [37:49] actions    (12)
    [49:236] height_scan (187, rough only)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["compute_symmetric_states_direct"]

# ---------------------------------------------------------------------------
# AYG joint permutation and sign arrays for left-right mirror
# ---------------------------------------------------------------------------

# Expected AYG joint ordering in Isaac Sim (type-grouped, from articulation).
_EXPECTED_JOINT_ORDER = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]

# Swap LF<->RF (0<->2, 4<->6, 8<->10) and LH<->RH (1<->3, 5<->7, 9<->11)
_JOINT_PERM = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
# Negate HAA joints (indices 0-3), keep HFE and KFE
_JOINT_SIGN: list[float] = [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]

_ordering_verified = False


def _verify_joint_ordering(env) -> None:
    """Assert that AYG's Isaac Sim joint order matches the hardcoded permutation arrays."""
    global _ordering_verified
    if _ordering_verified:
        return
    robot = env.unwrapped.scene.articulations["robot"]
    actual = list(robot.joint_names)
    if actual != _EXPECTED_JOINT_ORDER:
        raise RuntimeError(
            "AYG joint ordering in Isaac Sim does not match the expected leg-grouped order.\n"
            f"  Expected: {_EXPECTED_JOINT_ORDER}\n"
            f"  Actual:   {actual}\n"
            "Update _JOINT_PERM, _JOINT_SIGN, and _EXPECTED_JOINT_ORDER in\n"
            f"  {__file__}"
        )
    _ordering_verified = True


def _switch_ayg_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror to 12-dim joint data."""
    return joint_data[..., _JOINT_PERM] * torch.tensor(_JOINT_SIGN, device=joint_data.device, dtype=joint_data.dtype)


# Flat obs has 49 dims, rough adds 187 height scan rays = 236
_FLAT_OBS_DIM = 49


def _transform_policy_obs_direct(obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for Layout C (direct env, with optional height scan)."""
    obs = obs.clone()
    device = obs.device
    # base linear velocity: [vx, vy, vz] -> [vx, -vy, vz]
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)
    # base angular velocity: [wx, wy, wz] -> [-wx, wy, -wz]
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity: [gx, gy, gz] -> [gx, -gy, gz]
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity commands: [vx, vy, wz] -> [vx, -vy, -wz]
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # heading target (scalar yaw): negate
    obs[:, 12] = -obs[:, 12]
    # joint pos / joint vel / actions
    obs[:, 13:25] = _switch_ayg_joints_left_right(obs[:, 13:25])
    obs[:, 25:37] = _switch_ayg_joints_left_right(obs[:, 25:37])
    obs[:, 37:49] = _switch_ayg_joints_left_right(obs[:, 37:49])
    # height scan (if present): 187 rays from GridPatternCfg(size=[1.6, 1.0], resolution=0.1)
    # Grid is 17 (x) x 11 (y). Flip along y-axis.
    if obs.shape[1] > _FLAT_OBS_DIM:
        obs[:, 49:236] = obs[:, 49:236].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)
    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for the 12-dim action tensor."""
    return _switch_ayg_joints_left_right(actions.clone())


@torch.no_grad()
def compute_symmetric_states_direct(
    env,  # VecEnv wrapper — not used directly, but required by RSL-RL API
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augment observations and actions with left-right mirror (2x batch).

    For use with **Layout C** tasks:
        - Isaac-Velocity-Flat-Ayg-Direct-v0
        - Isaac-Velocity-Rough-Ayg-Direct-v0

    Args:
        env: The VecEnv wrapper (passed by RSL-RL, used for API compatibility).
        obs: Observation TensorDict with ``"policy"`` key containing the flat obs tensor.
        actions: Action tensor of shape ``(N, 12)``.

    Returns:
        ``(obs_aug, actions_aug)`` with batch size doubled.  Either may be ``None``
        if the corresponding input was ``None``.
    """
    _verify_joint_ordering(env)

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][batch_size:] = _transform_policy_obs_direct(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
