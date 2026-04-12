import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command_quad import GaitCommandQuad  # Import the GaitCommandQuad class


@configclass
class UniformGaitCommandCfgQuad(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommandQuad  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        offsets2: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        offsets3: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        offsets4: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        feet_height: tuple[float, float] = MISSING
        """Range for feet height [m]."""
        base_height: tuple[float, float] = MISSING
        """Range for base height [m]."""
        body_pitch: tuple[float, float] = MISSING
        """Range for commanded body pitch [rad]."""
        body_roll: tuple[float, float] = MISSING
        """Range for commanded body roll [rad]."""


    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    multi_gait: bool = False
    """If True, sample from 4 canonical gaits (trot, pace, bound, pronk) instead of uniform offset ranges."""

    binary_phases: bool = True
    """If True, use exact canonical offsets. If False, add uniform jitter around them."""

    gait_phase_jitter: float = 0.1
    """Half-width of uniform jitter added to canonical offsets when binary_phases is False."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""
