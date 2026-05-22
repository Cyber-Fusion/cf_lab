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

    couple_to_vx: bool = True
    """If True, scale the commanded body pitch, roll, and base-height deviation by
    ``1 / max(1, |vx| / 1.0)`` every step, where ``vx`` is the filtered forward velocity
    command read from the sibling velocity term named by :attr:`velocity_command_name`.
    Sampled values are preserved internally so the command returns to its sampled magnitude
    when the commanded ``vx`` drops back to within ``[-1, 1]``."""

    default_base_height: float = 0.30
    """Nominal stance height used as the zero-deviation reference for the base-height scaling.
    The scaling is applied to ``(sampled_base_height - default_base_height)`` so that the
    commanded base height collapses to this default at high ``|vx|``. Set to match the robot's
    nominal standing height."""

    velocity_command_name: str = "base_velocity"
    """Name of the sibling velocity command term used to read ``vx`` when
    :attr:`couple_to_vx` is True."""

    couple_frequency_to_vx: bool = False
    """If True, the gait frequency range is interpolated between :attr:`ranges.frequencies`
    (at low ``|vx|``) and :attr:`frequencies_high_vx` (at high ``|vx|``) every step. A per-env
    uniform percentile is sampled at resample time so each env keeps a stable position within
    the (shifting) range. The interpolation uses the same ``duration_vx_low`` /
    ``duration_vx_high`` knees as :attr:`couple_duration_to_vx`."""

    frequencies_high_vx: tuple[float, float] = (2.5, 3.5)
    """High-speed gait frequency range [Hz], used when :attr:`couple_frequency_to_vx` is True
    and ``|vx| >= duration_vx_high``. Higher frequencies shorten the swing window so leg
    cycling can keep up with body translation at top speed."""

    couple_duration_to_vx: bool = False
    """If True, the contact duration range is interpolated between
    :attr:`ranges.durations` (used at low ``|vx|``) and :attr:`durations_high_vx`
    (used at high ``|vx|``) every step. A per-env uniform percentile is sampled
    at resample time so each env keeps a stable position within the (shifting)
    range. The interpolation is linear in ``|vx|`` between
    :attr:`duration_vx_low` and :attr:`duration_vx_high`, clamped outside."""

    durations_high_vx: tuple[float, float] = (0.4, 0.5)
    """High-speed contact duration range, used when :attr:`couple_duration_to_vx`
    is True and ``|vx| >= duration_vx_high``. Lower stance fractions enable an
    aerial phase in trot for running gaits."""

    duration_vx_low: float = 1.0
    """``|vx|`` [m/s] at or below which the duration range equals
    :attr:`ranges.durations`."""

    duration_vx_high: float = 2.5
    """``|vx|`` [m/s] at or above which the duration range equals
    :attr:`durations_high_vx`."""
