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
    """Configuration for the WTW gait command generator (8D behavior vector).

    Command layout: [theta1, theta2, theta3, frequency, base_height, body_pitch, stance_width, footswing_height]

    theta1/theta2/theta3 are phase offset parameters that encode quadruped gaits:
        - Trot:    (0.5, 0.0, 0.0)
        - Pronk:   (0.0, 0.0, 0.0)
        - Bound:   (0.0, 0.5, 0.0)
        - Pace:    (0.0, 0.0, 0.5)
    """

    class_type: type = GaitCommandQuad

    @configclass
    class Ranges:
        """Uniform distribution ranges for the 8D behavior parameters."""

        theta1: tuple[float, float] = MISSING
        """Range for phase offset parameter 1 [0-1]."""
        theta2: tuple[float, float] = MISSING
        """Range for phase offset parameter 2 [0-1]."""
        theta3: tuple[float, float] = MISSING
        """Range for phase offset parameter 3 [0-1]."""
        frequency: tuple[float, float] = MISSING
        """Range for gait stepping frequency [Hz]."""
        base_height: tuple[float, float] = MISSING
        """Range for body height command [m]."""
        body_pitch: tuple[float, float] = MISSING
        """Range for body pitch command [rad]."""
        stance_width: tuple[float, float] = MISSING
        """Range for foot stance width command [m]."""
        footswing_height: tuple[float, float] = MISSING
        """Range for footswing height command [m]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""

    canonical_gait_probability: float = 0.5
    """Probability of sampling theta from a canonical gait center (vs. uniform)."""

    canonical_gait_std: float = 0.1
    """Standard deviation of Gaussian when sampling theta around a canonical gait."""
