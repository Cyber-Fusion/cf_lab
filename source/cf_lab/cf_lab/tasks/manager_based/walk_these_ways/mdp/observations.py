from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get per-foot timing references as observation.

    Computes t_t = [sin(2*pi*t^LF), sin(2*pi*t^RF), sin(2*pi*t^LH), sin(2*pi*t^RH)]
    where per-foot timings are derived from the direct offset parameterization.

    Returns:
        torch.Tensor: The per-foot phase observation. Shape: (num_envs, 4).
    """
    from cf_lab.tasks.manager_based.walk_these_ways.mdp.rewards import (
        IDX_FREQUENCY,
        IDX_OFFSET2,
        IDX_OFFSET3,
        IDX_OFFSET4,
        compute_per_foot_timings,
    )

    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 4, device=env.device)

    command_term = env.command_manager.get_term("gait_command")
    gait_params = command_term.command

    off2 = gait_params[:, IDX_OFFSET2]
    off3 = gait_params[:, IDX_OFFSET3]
    off4 = gait_params[:, IDX_OFFSET4]
    frequency = gait_params[:, IDX_FREQUENCY]

    t = torch.remainder(env.episode_length_buf * env.step_dt * frequency, 1.0)
    t_LF, t_RF, t_LH, t_RH = compute_per_foot_timings(off2, off3, off4, t)

    phases = torch.stack([
        torch.sin(2 * torch.pi * t_LF),
        torch.sin(2 * torch.pi * t_RF),
        torch.sin(2 * torch.pi * t_LH),
        torch.sin(2 * torch.pi * t_RH),
    ], dim=-1)  # (N, 4)

    # Zero out gait phase when velocity command is zero (standing).
    # Without this, the oscillating phase signal tells the policy "follow this gait"
    # while rewards penalize movement — causing the robot to tap in place.
    cmd = env.command_manager.get_command("base_velocity")
    cmd_not_null = (cmd.norm(dim=1, p=1) > 0.05).unsqueeze(1)  # (N, 1)
    phases = phases * cmd_not_null

    return phases


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The 9D gait command [freq, dur, off2, off3, off4,
                      feet_h, base_h, pitch, roll]. Shape: (num_envs, 9).
    """
    return env.command_manager.get_command(command_name)
