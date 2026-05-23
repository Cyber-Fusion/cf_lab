# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the multi-gait velocity command generator."""

from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .gait_velocity_command import MultiGaitVelocityCommand


@configclass
class MultiGaitVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command with per-gait sampling ranges and a linear curriculum on the max ``vx``.

    Reads the canonical gait assignment (0=trot, 1=pace, 2=bound, 3=pronk) from a sibling
    :class:`GaitCommandQuad` term and samples velocities from the per-gait range for that env.
    The inherited ``ranges`` field is still used by the parent's heading-control clamp and must
    be provided; set it to cover the widest per-gait ``ang_vel_z`` bounds.
    """

    class_type: type = MultiGaitVelocityCommand

    ranges_per_gait: list = MISSING
    """List of four :class:`UniformVelocityCommandCfg.Ranges`, one per canonical gait, in the
    order ``[trot, pace, bound, pronk]`` matching :attr:`GaitCommandQuad.CANONICAL_GAITS`."""

    initial_max_lin_vel_x: float = 1.0
    """Curriculum-start bound on |lin_vel_x|; the per-env sampling range linearly interpolates
    from ``(-initial, +initial)`` at progress 0 to the per-gait final range at progress 1.

    Only ``lin_vel_x`` is curriculum-ramped. ``lin_vel_y`` and ``ang_vel_z`` are sampled
    directly from their per-gait ranges from step 0, since those ranges are kept small
    (typically ``|vy|, |omega| <= 1``)."""

    gait_command_name: str = "gait_command"
    """Name of the sibling :class:`GaitCommandQuad` term from which to read gait_ids."""

    couple_to_vx: bool = True
    """If True, scale the filtered ``vy`` and ``ang_vel_z`` commands every step by
    ``1 / max(1, |vx| / 1.0)``. Keeps secondary tracking objectives small while the policy is
    asked to run fast in the forward direction. The same ``vx_scale`` rule is applied (in the
    sibling :class:`GaitCommandQuad`) to the pitch, roll, and base-height-deviation commands."""
