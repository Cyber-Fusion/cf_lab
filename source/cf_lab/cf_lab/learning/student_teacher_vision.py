# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""StudentTeacher subclass with a depth-CNN encoder for the vision student.

The locked Phase 1 teacher (privileged ego + 187-dim height_scan) is unchanged
and inherited from the stock `StudentTeacher`. The student is replaced with
`VisionStudentNet`, which expects a flat input of shape

    (N, ego_dim + depth_t * depth_h * depth_w)

— ego terms first, depth-frame stack flattened second — and internally
reshapes the depth portion back to (N, T, H, W) for a small 2-D CNN. This
matches the env's `policy` group when `concatenate_terms=True` and the
depth ObsTerm uses `history_length=T, flatten_history_dim=True`.

The cfg passes ego/depth shape constants explicitly so the network never
has to guess the layout. See
`tasks/manager_based/velocity/agents/rsl_rl_distillation_cfg.py::AygRoughStudentVisionDistillationCfg`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import StudentTeacher


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class VisionStudentNet(nn.Module):
    """Depth-CNN + ego-MLP fusion network. ONNX-exportable single forward pass."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        ego_dim: int,
        depth_t: int,
        depth_h: int,
        depth_w: int,
        depth_latent_dim: int = 64,
        ego_latent_dim: int = 128,
        head_hidden_dims: tuple[int, ...] = (256, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        depth_flat_dim = depth_t * depth_h * depth_w
        if num_obs != ego_dim + depth_flat_dim:
            raise ValueError(
                f"VisionStudentNet expected num_obs={ego_dim + depth_flat_dim} "
                f"(ego_dim={ego_dim} + depth_t*depth_h*depth_w={depth_flat_dim}), got {num_obs}."
            )

        self.ego_dim = ego_dim
        self.depth_t = depth_t
        self.depth_h = depth_h
        self.depth_w = depth_w

        # Treat the T-frame stack as input channels of a small 2-D CNN.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_t, 32, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
        )
        with torch.no_grad():
            conv_out = self.depth_encoder(torch.zeros(1, depth_t, depth_h, depth_w))
        self.depth_head = nn.Sequential(
            nn.Linear(conv_out.flatten(1).shape[1], depth_latent_dim),
            _activation(activation),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, ego_latent_dim),
            _activation(activation),
        )

        fused_dim = depth_latent_dim + ego_latent_dim
        layers: list[nn.Module] = []
        prev = fused_dim
        for h in head_hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.policy_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, ego_dim + T*H*W) — env concatenates ego terms first, depth flat last.
        ego = x[:, : self.ego_dim]
        depth_flat = x[:, self.ego_dim :]
        n = depth_flat.shape[0]
        depth_stack = depth_flat.view(n, self.depth_t, self.depth_h, self.depth_w)
        # Normalize raw depth (meters in roughly [0, 9]) into a stable range.
        depth_stack = depth_stack / 9.0
        depth_latent = self.depth_head(self.depth_encoder(depth_stack).flatten(1))
        ego_latent = self.ego_encoder(ego)
        return self.policy_head(torch.cat([depth_latent, ego_latent], dim=-1))


class StudentTeacherVision(StudentTeacher):
    """`StudentTeacher` with the student replaced by a depth-CNN fusion network.

    Constructor accepts the stock `StudentTeacher` arguments plus the depth/ego
    shape constants needed by `VisionStudentNet`. The stock `student` MLP that
    `super().__init__` builds is discarded and replaced; the teacher network,
    noise distribution, normalizers, and the auto-detecting `load_state_dict`
    (which extracts `actor.*` keys from an ActorCritic teacher checkpoint) are
    all inherited unchanged.
    """

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        ego_dim: int,
        depth_t: int,
        depth_h: int,
        depth_w: int,
        depth_latent_dim: int = 64,
        ego_latent_dim: int = 128,
        head_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        student_hidden_dims=(256, 256, 256),
        teacher_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            **kwargs,
        )

        # Recompute the student-side concatenated obs dim so we can shape-check.
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            num_student_obs += obs[obs_group].shape[-1]

        self.student = VisionStudentNet(
            num_obs=num_student_obs,
            num_actions=num_actions,
            ego_dim=ego_dim,
            depth_t=depth_t,
            depth_h=depth_h,
            depth_w=depth_w,
            depth_latent_dim=depth_latent_dim,
            ego_latent_dim=ego_latent_dim,
            head_hidden_dims=tuple(head_hidden_dims),
            activation=activation,
        )
        # Same noise std parameter shape — handled by parent. Re-print so
        # the runner's startup log shows the actual student we're using.
        n_params = sum(p.numel() for p in self.student.parameters())
        print(
            f"StudentTeacherVision: replaced self.student with VisionStudentNet"
            f" (ego_dim={ego_dim}, depth_stack={depth_t}x{depth_h}x{depth_w},"
            f" params={n_params:,})"
        )
