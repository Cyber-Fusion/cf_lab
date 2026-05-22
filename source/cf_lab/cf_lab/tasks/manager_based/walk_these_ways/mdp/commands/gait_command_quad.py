"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

from .scaling import vx_scale

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg_quad import UniformGaitCommandCfgQuad


class GaitCommandQuad(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: UniformGaitCommandCfgQuad
    """The configuration of the command generator."""

    # Canonical gait offsets: (offsets2/RF, offsets3/LH, offsets4/RH)
    CANONICAL_GAITS = torch.tensor([
        [0.5, 0.5, 0.0],  # trot
        [0.5, 0.0, 0.5],  # pace
        [0.0, 0.5, 0.5],  # bound
        [0.0, 0.0, 0.0],  # pronk
    ])

    def __init__(self, cfg: UniformGaitCommandCfgQuad, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        # command format: [freq, duration, offset2, offset3, offset4, feet_h, base_h, pitch, roll]
        self.gait_command = torch.zeros(self.num_envs, 9, device=self.device)
        # Unscaled (sampled) pitch/roll/base_height targets, kept so the vx-coupling scaling can
        # be re-applied each step without losing the original magnitude when |vx| drops back
        # below the knee.
        self._sampled_pitch = torch.zeros(self.num_envs, device=self.device)
        self._sampled_roll = torch.zeros(self.num_envs, device=self.device)
        # Initialise base_height cache to the default so the deviation `(_sampled - default)`
        # is zero before the first resample (no spurious height change at t=0).
        self._sampled_base_height = torch.full(
            (self.num_envs,), float(cfg.default_base_height), device=self.device
        )
        # Per-env uniform percentile in [0, 1] used to pick a stable position within the
        # vx-dependent duration range (see `couple_duration_to_vx`). Sampled at resample time.
        self._duration_u = torch.zeros(self.num_envs, device=self.device)
        # Same idea for the frequency range (see `couple_frequency_to_vx`).
        self._frequency_u = torch.zeros(self.num_envs, device=self.device)
        # incremental gait phase state
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        # canonical gait index per env (0=trot, 1=pace, 2=bound, 3=pronk); stays 0 when multi_gait=False
        self.gait_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dt = env.step_dt
        # move canonical gaits to device
        self._canonical_gaits = self.CANONICAL_GAITS.to(self.device)
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def reset(self, env_ids=None):
        if env_ids is None:
            self.gait_indices[:] = 0.0
        else:
            self.gait_indices[env_ids] = 0.0
        return super().reset(env_ids)

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 9)."""
        return self.gait_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _sample_multi_gait_offsets(self, env_ids):
        """Sample offsets from canonical gaits for the given environments."""
        n = len(env_ids)
        # assign each env to one of 4 gait categories uniformly
        gait_idx = torch.randint(0, 4, (n,), device=self.device)
        # remember the canonical gait assignment so sibling terms (e.g. MultiGaitVelocityCommand)
        # can read it without re-running the random draw
        self.gait_ids[env_ids] = gait_idx
        offsets = self._canonical_gaits[gait_idx]  # (n, 3)
        # optionally add jitter around canonical values
        if not self.cfg.binary_phases:
            jitter = self.cfg.gait_phase_jitter
            offsets = offsets + torch.empty_like(offsets).uniform_(-jitter, jitter)
            offsets = torch.remainder(offsets, 1.0)
        self.gait_command[env_ids, 2:5] = offsets

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        if self.cfg.couple_frequency_to_vx:
            # store percentile; actual frequency is computed every step in `_update_command`
            self._frequency_u[env_ids] = r.uniform_(0.0, 1.0)
            # Seed gait_command[:, 0] with the low-|vx| range value so it is valid before
            # the first `_update_command` runs (reward terms read it at step 1, which
            # happens before command_manager.compute on the same tick).
            lo_f, hi_f = self.cfg.ranges.frequencies
            self.gait_command[env_ids, 0] = lo_f + self._frequency_u[env_ids] * (hi_f - lo_f)
        else:
            self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        # -- contact duration
        if self.cfg.couple_duration_to_vx:
            # store percentile; actual duration is computed every step in `_update_command`
            self._duration_u[env_ids] = r.uniform_(0.0, 1.0)
            # Same seeding as for frequency above.
            lo_d, hi_d = self.cfg.ranges.durations
            self.gait_command[env_ids, 1] = lo_d + self._duration_u[env_ids] * (hi_d - lo_d)
        else:
            self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.durations)
        # -- phase offsets
        if self.cfg.multi_gait:
            self._sample_multi_gait_offsets(env_ids)
        else:
            self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.offsets2)
            self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.offsets3)
            self.gait_command[env_ids, 4] = r.uniform_(*self.cfg.ranges.offsets4)
        # -- feet height
        self.gait_command[env_ids, 5] = r.uniform_(*self.cfg.ranges.feet_height)
        # -- base height
        self.gait_command[env_ids, 6] = r.uniform_(*self.cfg.ranges.base_height)
        # -- body pitch
        self.gait_command[env_ids, 7] = r.uniform_(*self.cfg.ranges.body_pitch)
        # -- body roll
        self.gait_command[env_ids, 8] = r.uniform_(*self.cfg.ranges.body_roll)
        # Cache sampled values for the vx-coupling scaling in `_update_command`.
        self._sampled_base_height[env_ids] = self.gait_command[env_ids, 6]
        self._sampled_pitch[env_ids] = self.gait_command[env_ids, 7]
        self._sampled_roll[env_ids] = self.gait_command[env_ids, 8]


    def _update_command(self):
        """Incrementally advance gait phase and compute per-foot indices."""
        # Apply vx-coupling first so the phase advance below uses up-to-date frequency.
        # The vx reads the *previous* step's filtered value from the sibling velocity term
        # (it ticks after this one); the 1-step lag is negligible at the IIR time constant.
        if self.cfg.couple_to_vx or self.cfg.couple_duration_to_vx or self.cfg.couple_frequency_to_vx:
            vel_term = self._env.command_manager.get_term(self.cfg.velocity_command_name)
            vx = vel_term.command[:, 0]
            # Uniform 1/max(1,|vx|) shrink on pitch / roll / base-height-deviation.
            if self.cfg.couple_to_vx:
                scale = vx_scale(vx)
                self.gait_command[:, 7] = self._sampled_pitch * scale
                self.gait_command[:, 8] = self._sampled_roll * scale
                default = self.cfg.default_base_height
                self.gait_command[:, 6] = default + (self._sampled_base_height - default) * scale
            # Linearly interpolate duration and/or frequency ranges between low- and high-vx
            # settings, then place each env at its per-env percentile.
            if self.cfg.couple_duration_to_vx or self.cfg.couple_frequency_to_vx:
                vx_abs = vx.abs()
                denom = max(self.cfg.duration_vx_high - self.cfg.duration_vx_low, 1e-6)
                alpha = ((vx_abs - self.cfg.duration_vx_low) / denom).clamp(0.0, 1.0)
                if self.cfg.couple_duration_to_vx:
                    lo_lo, lo_hi = self.cfg.ranges.durations
                    hi_lo, hi_hi = self.cfg.durations_high_vx
                    range_lo = lo_lo + alpha * (hi_lo - lo_lo)
                    range_hi = lo_hi + alpha * (hi_hi - lo_hi)
                    self.gait_command[:, 1] = range_lo + self._duration_u * (range_hi - range_lo)
                if self.cfg.couple_frequency_to_vx:
                    lo_lo, lo_hi = self.cfg.ranges.frequencies
                    hi_lo, hi_hi = self.cfg.frequencies_high_vx
                    range_lo = lo_lo + alpha * (hi_lo - lo_lo)
                    range_hi = lo_hi + alpha * (hi_hi - lo_hi)
                    self.gait_command[:, 0] = range_lo + self._frequency_u * (range_hi - range_lo)

        frequencies = self.gait_command[:, 0]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        offsets2 = self.gait_command[:, 2]
        offsets3 = self.gait_command[:, 3]
        offsets4 = self.gait_command[:, 4]
        self.foot_indices = torch.remainder(
            torch.stack([
                self.gait_indices,
                self.gait_indices + offsets2 + 1,
                self.gait_indices + offsets3 + 1,
                self.gait_indices + offsets4 + 1,
            ], dim=1),
            1.0,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
