# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command_quad import GaitCommandQuad


@configclass
class UniformGaitCommandCfgQuad(CommandTermCfg):
    """Configuration for the WTW gait command generator (9D behavior vector).

    Command layout: [freq, dur, off2, off3, off4, feet_h, base_h, pitch, roll]

    Phase offsets are direct per-foot offsets relative to the reference foot (LF):
        off2 = RF offset, off3 = LH offset, off4 = RH offset

    Canonical quadruped gaits (off2, off3, off4):
        - Trot:  (0.5, 0.5, 0.0)
        - Pace:  (0.5, 0.0, 0.5)
        - Bound: (0.0, 0.5, 0.5)
        - Pronk: (0.0, 0.0, 0.0)
    """

    class_type: type = GaitCommandQuad

    @configclass
    class Ranges:
        """Uniform distribution ranges for the 9D behavior parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait stepping frequency [Hz]."""
        durations: tuple[float, float] = MISSING
        """Range for stance duty cycle [0-1]."""
        offsets2: tuple[float, float] = MISSING
        """Range for RF foot phase offset [0-1]."""
        offsets3: tuple[float, float] = MISSING
        """Range for LH foot phase offset [0-1]."""
        offsets4: tuple[float, float] = MISSING
        """Range for RH foot phase offset [0-1]."""
        feet_height: tuple[float, float] = MISSING
        """Range for foot swing height command [m]."""
        base_height: tuple[float, float] = MISSING
        """Range for body height command [m]."""
        body_pitch: tuple[float, float] = MISSING
        """Range for body pitch command [rad]."""
        body_roll: tuple[float, float] = MISSING
        """Range for body roll command [rad]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""

    multi_gait: bool = True
    """Whether to sample phase offsets from canonical gaits."""

    binary_phases: bool = True
    """When True, use exact canonical offsets (no jitter)."""

    gait_phase_jitter: float = 0.1
    """Jitter added to canonical gait offsets when binary_phases is False."""
