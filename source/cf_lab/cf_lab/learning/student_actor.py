# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Networks for Issue #16 student distillation.

`StudentActor` is the deployable network: ego state (+ optional depth stack) -> 12-d joint
action mean. `use_depth=False` produces the blind baseline (no camera, no encoder) used to
measure the proprio-only ceiling. `TeacherMLP` is a frozen wrapper around the locked Phase 1
teacher's actor weights, used only at training time to produce DAgger supervision targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Phase 1 teacher architecture (matches AygRoughPPORunnerCfg + the checkpoint state-dict shapes).
TEACHER_OBS_DIM = 235
TEACHER_HIDDEN_DIMS = (512, 256, 128)
ACTION_DIM = 12


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TeacherMLP(nn.Module):
    """Plain-Sequential mirror of the locked RSL-RL teacher's actor."""

    def __init__(self) -> None:
        super().__init__()
        # ELU + Sequential ordering matches the locked checkpoint's actor.0/2/4/6 layout.
        self.actor = _build_mlp(TEACHER_OBS_DIM, TEACHER_HIDDEN_DIMS, ACTION_DIM, nn.ELU)

    @classmethod
    def load_frozen(cls, ckpt_path: str, device: torch.device | str = "cpu") -> TeacherMLP:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        full_sd = ckpt.get("model_state_dict", ckpt)
        # Take only the actor.* keys; the rest (critic, std, optimizer) is irrelevant for distillation.
        actor_sd = {k[len("actor.") :]: v for k, v in full_sd.items() if k.startswith("actor.")}
        model = cls().to(device)
        model.actor.load_state_dict(actor_sd, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.no_grad()
    def forward(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(teacher_obs)


class DepthEncoder(nn.Module):
    """Small CNN over a (B, T, H, W) frame stack. T is treated as input channels."""

    def __init__(self, num_frames: int = 4, height: int = 45, width: int = 80, latent_dim: int = 64) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, num_frames, height, width)
            flat = self.conv(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, latent_dim), nn.ELU())
        self.latent_dim = latent_dim

    def forward(self, depth_stack: torch.Tensor) -> torch.Tensor:
        # depth_stack: (B, T, H, W). Normalize to [0, 1]-ish using clip-range scaling.
        # We expect raw meters in [0, 9]; divide so the encoder sees consistent magnitudes.
        x = depth_stack / 9.0
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)


class StudentActor(nn.Module):
    """Depth + ego -> action mean. ONNX-exportable single forward pass.

    Set ``use_depth=False`` for the blind variant; the depth encoder is then absent and the
    forward pass ignores ``depth_stack``.
    """

    def __init__(
        self,
        ego_dim: int = 45,
        num_frames: int = 4,
        depth_h: int = 45,
        depth_w: int = 80,
        depth_latent: int = 64,
        ego_hidden: int = 128,
        head_hidden: tuple[int, ...] = (256, 128),
        use_depth: bool = True,
    ) -> None:
        super().__init__()
        self.use_depth = use_depth
        if use_depth:
            self.depth_encoder = DepthEncoder(num_frames, depth_h, depth_w, depth_latent)
            fused_dim = depth_latent + ego_hidden
        else:
            self.depth_encoder = None
            fused_dim = ego_hidden
        self.ego_encoder = nn.Sequential(nn.Linear(ego_dim, ego_hidden), nn.ELU())
        self.policy_head = _build_mlp(fused_dim, head_hidden, ACTION_DIM, nn.ELU)

    def forward(self, ego: torch.Tensor, depth_stack: torch.Tensor | None = None) -> torch.Tensor:
        ego_latent = self.ego_encoder(ego)
        if not self.use_depth:
            return self.policy_head(ego_latent)
        depth_latent = self.depth_encoder(depth_stack)
        return self.policy_head(torch.cat([depth_latent, ego_latent], dim=-1))


def assemble_ego(policy_dict: dict[str, torch.Tensor], term_order: list[str]) -> torch.Tensor:
    """Concatenate the per-term ego tensors in a deterministic order.

    The student env's `policy` group is a Dict (because depth can't concat with vector terms),
    so we have to assemble the ego vector ourselves. Caller passes the term_order list captured
    once at startup so the layout is stable across rollouts.
    """
    return torch.cat([policy_dict[name] for name in term_order], dim=-1)


def stack_depth(buffer: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Roll a per-env circular buffer of depth frames forward by one.

    buffer: (B, T, H, W) — last T frames, oldest at index 0
    frame:  (B, H, W) — new observation
    returns: (B, T, H, W) — buffer shifted, new frame at index -1
    """
    # avoid in-place to keep autograd-clean for any downstream gradient checks
    return torch.cat([buffer[:, 1:], frame.unsqueeze(1)], dim=1)
