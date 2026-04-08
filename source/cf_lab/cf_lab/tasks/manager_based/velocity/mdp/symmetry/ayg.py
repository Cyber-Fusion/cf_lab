# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right (sagittal) symmetry augmentation for the AYG quadruped.

AYG joint ordering in Isaac Sim (grouped by leg, from URDF):

    [0] LF_HAA  [1] LF_HFE  [2] LF_KFE
    [3] RF_HAA  [4] RF_HFE  [5] RF_KFE
    [6] LH_HAA  [7] LH_HFE  [8] LH_KFE
    [9] RH_HAA  [10] RH_HFE [11] RH_KFE

Under left-right mirror (sagittal reflection):
    - LF (0,1,2) <-> RF (3,4,5)
    - LH (6,7,8) <-> RH (9,10,11)
    - HAA joints (indices 0,3,6,9) get sign-flipped

Provides two public functions matching different observation layouts:
    - ``compute_symmetric_states`` — Layout A (LocomotionVelocityRoughEnvCfg obs):
      [lin_vel(3), ang_vel(3), grav(3), cmd(3), jpos(12), jvel(12), act(12), height_scan?]
    - ``compute_symmetric_states_spot_inspired`` — Layout B (AygObservationsCfg):
      [ang_vel(3), grav(3), cmd(3), jpos(12), jvel(12), act(12)]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "compute_symmetric_states_spot_inspired"]

# ---------------------------------------------------------------------------
# AYG joint permutation and sign arrays for left-right mirror
# ---------------------------------------------------------------------------

# Expected AYG joint ordering in Isaac Sim (leg-grouped, from URDF declaration order).
_EXPECTED_JOINT_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]

# Swap LF<->RF and LH<->RH
_JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# Negate HAA joints only (indices 0,3,6,9 in the original ordering map to
# positions 0,3,6,9 in the permuted result as well)
_JOINT_SIGN: list[float] = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]

# Foot body ordering from URDF (matched by ".*_Foot"):
#   [0] LF_Foot  [1] RF_Foot  [2] LH_Foot  [3] RH_Foot
# Swap LF<->RF and LH<->RH:
_FOOT_PERM = [1, 0, 3, 2]

_ordering_verified = False


def _verify_joint_ordering(env: ManagerBasedRLEnv) -> None:
    """Assert that AYG's Isaac Sim joint order matches the hardcoded permutation arrays.

    Called once on the first augmentation call.  Raises ``RuntimeError`` with the
    actual ordering if there is a mismatch so the user knows exactly what to fix.
    """
    global _ordering_verified
    if _ordering_verified:
        return
    robot = env.scene["robot"]
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
    """Apply left-right mirror to 12-dim joint data (pos, vel, or actions).

    Permutes LF<->RF and LH<->RH, then negates HAA joints.
    """
    return joint_data[..., _JOINT_PERM] * torch.tensor(_JOINT_SIGN, device=joint_data.device, dtype=joint_data.dtype)


def _switch_ayg_feet_left_right(foot_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror to 4-dim foot data (heights, air time, contact, forces).

    Permutes LF<->RF and LH<->RH.  Foot quantities are scalars per foot so
    no sign flip is needed.
    """
    return foot_data[..., _FOOT_PERM]


# ---------------------------------------------------------------------------
# Layout A: LocomotionVelocityRoughEnvCfg observation structure
#   [0:3]  base_lin_vel
#   [3:6]  base_ang_vel
#   [6:9]  projected_gravity
#   [9:12] velocity_commands
#   [12:24] joint_pos  (12)
#   [24:36] joint_vel  (12)
#   [36:48] actions    (12)
#   [48:235] height_scan (187, rough only)
# ---------------------------------------------------------------------------


def _transform_policy_obs_layout_a(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for Layout A (with optional height scan)."""
    obs = obs.clone()
    device = obs.device
    # base linear velocity: [vx, vy, vz] -> [vx, -vy, vz]
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # base angular velocity: [wx, wy, wz] -> [-wx, wy, -wz]
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity: [gx, gy, gz] -> [gx, -gy, gz]
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity commands: [vx, vy, wz] -> [vx, -vy, -wz]
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos / joint vel / actions
    obs[:, 12:24] = _switch_ayg_joints_left_right(obs[:, 12:24])
    obs[:, 24:36] = _switch_ayg_joints_left_right(obs[:, 24:36])
    obs[:, 36:48] = _switch_ayg_joints_left_right(obs[:, 36:48])
    # height scan (if present): 187 rays from GridPatternCfg(size=[1.6, 1.0], resolution=0.1)
    # Grid is 17 (x) x 11 (y). Flip along y-axis (dim=1 of the reshaped view).
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)
    return obs


# ---------------------------------------------------------------------------
# Layout B: AygObservationsCfg (spot-inspired, no base_lin_vel)
#   [0:3]  base_ang_vel
#   [3:6]  projected_gravity
#   [6:9]  velocity_commands
#   [9:21]  joint_pos  (12)
#   [21:33] joint_vel  (12)
#   [33:45] actions    (12)
# ---------------------------------------------------------------------------


def _transform_policy_obs_layout_b(obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for Layout B (no base_lin_vel, no height scan)."""
    obs = obs.clone()
    device = obs.device
    # base angular velocity: [wx, wy, wz] -> [-wx, wy, -wz]
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity: [gx, gy, gz] -> [gx, -gy, gz]
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity commands: [vx, vy, wz] -> [vx, -vy, -wz]
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
    # joint pos / joint vel / actions
    obs[:, 9:21] = _switch_ayg_joints_left_right(obs[:, 9:21])
    obs[:, 21:33] = _switch_ayg_joints_left_right(obs[:, 21:33])
    obs[:, 33:45] = _switch_ayg_joints_left_right(obs[:, 33:45])
    return obs


# ---------------------------------------------------------------------------
# Critic observation transform for Layout B (AygObservationsCfg.CriticCfg)
#   [0:3]   base_lin_vel        (3)
#   [3:6]   base_ang_vel        (3)
#   [6:9]   projected_gravity   (3)
#   [9:12]  velocity_commands   (3)
#   [12:24] joint_pos           (12)
#   [24:36] joint_vel           (12)
#   [36:48] actions             (12)
#   [48:52] foot_heights        (4)  — [LF, RF, LH, RH]
#   [52:56] foot_air_time       (4)
#   [56:60] foot_contact        (4)
#   [60:64] foot_contact_forces (4)
# ---------------------------------------------------------------------------


def _transform_critic_obs_layout_b(obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for Layout B critic (privileged observations)."""
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
    # joint pos / joint vel / actions
    obs[:, 12:24] = _switch_ayg_joints_left_right(obs[:, 12:24])
    obs[:, 24:36] = _switch_ayg_joints_left_right(obs[:, 24:36])
    obs[:, 36:48] = _switch_ayg_joints_left_right(obs[:, 36:48])
    # foot-based privileged terms: swap LF<->RF, LH<->RH
    obs[:, 48:52] = _switch_ayg_feet_left_right(obs[:, 48:52])
    obs[:, 52:56] = _switch_ayg_feet_left_right(obs[:, 52:56])
    obs[:, 56:60] = _switch_ayg_feet_left_right(obs[:, 56:60])
    obs[:, 60:64] = _switch_ayg_feet_left_right(obs[:, 60:64])
    return obs


# ---------------------------------------------------------------------------
# Action mirroring (shared by both layouts — actions are always 12-dim joints)
# ---------------------------------------------------------------------------


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for the 12-dim action tensor."""
    return _switch_ayg_joints_left_right(actions.clone())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augment observations and actions with left-right mirror (2x batch).

    For use with **Layout A** tasks:
        - Isaac-Velocity-Flat-Ayg-v0
        - Isaac-Velocity-Rough-Ayg-v0

    Args:
        env: The manager-based RL environment.
        obs: Observation TensorDict with ``"policy"`` (and optionally ``"critic"``) keys.
        actions: Action tensor of shape ``(N, 12)``.

    Returns:
        ``(obs_aug, actions_aug)`` with batch size doubled.  Either may be ``None``
        if the corresponding input was ``None``.
    """
    _verify_joint_ordering(env.unwrapped)

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        # original stays in [:batch_size], mirrored goes into [batch_size:]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_layout_a(env.unwrapped, obs["policy"])
        # Layout A envs (Isaac-Velocity-Flat/Rough-Ayg-v0) do not define a
        # separate critic observation group, so there is no "critic" key to
        # mirror.  If a critic group is added in the future, a corresponding
        # transform function must be added here.
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


@torch.no_grad()
def compute_symmetric_states_spot_inspired(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augment observations and actions with left-right mirror (2x batch).

    For use with **Layout B** tasks:
        - Isaac-Velocity-Spot-Like-Flat-Ayg-v0

    Args:
        env: The manager-based RL environment.
        obs: Observation TensorDict with ``"policy"`` (and optionally ``"critic"``) keys.
        actions: Action tensor of shape ``(N, 12)``.

    Returns:
        ``(obs_aug, actions_aug)`` with batch size doubled.  Either may be ``None``
        if the corresponding input was ``None``.
    """
    _verify_joint_ordering(env.unwrapped)

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][batch_size:] = _transform_policy_obs_layout_b(obs["policy"])
        # Mirror critic observations so the value function learns G-invariance
        # V(g·s) = V(s).  The critic sees privileged state including foot data.
        if "critic" in obs.keys():
            obs_aug["critic"][batch_size:] = _transform_critic_obs_layout_b(obs["critic"])
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
