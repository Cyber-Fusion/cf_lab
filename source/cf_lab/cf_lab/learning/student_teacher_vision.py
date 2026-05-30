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
import torch.utils.checkpoint as torch_checkpoint
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
    """Temporal depth encoder + ego-MLP fusion network. ONNX-exportable, stateless.

    Pipeline (fixes the channel-stack CNN's inability to register a moving camera):

    1. A **shared per-frame 2-D CNN** encodes each of the ``depth_t`` depth frames
       independently into a per-frame latent (so the same physical patch is encoded
       the same way regardless of which frame it appears in).
    2. The per-frame depth latents are concatenated with the aligned **per-frame
       proprio vector** (ego-motion) and fed to a **1-D temporal convolution** over the
       time axis, which can model how the scene/flow evolves across frames.
    3. The pooled temporal latent is fused with the **current ego** latent and mapped
       to actions.

    Input layout (concatenated policy obs, see ``rough_student_env_cfg.py``):
        ``[ ego(ego_dim) | depth(depth_t*depth_h*depth_w) | proprio(proprio_t*proprio_dim) ]``
    Frames are ordered newest-first; the temporal conv is order-agnostic.

    The whole forward pass is a single stateless graph (Conv2d/Conv1d/Linear/ELU),
    so it exports cleanly to ONNX and runs on the Jetson without recurrent state.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        ego_dim: int,
        depth_t: int,
        depth_h: int,
        depth_w: int,
        proprio_dim: int,
        proprio_t: int,
        frame_latent_dim: int = 64,
        temporal_dim: int = 64,
        ego_latent_dim: int = 128,
        head_hidden_dims: tuple[int, ...] = (256, 128),
        activation: str = "elu",
        far_clip: float = 9.0,
    ) -> None:
        super().__init__()
        depth_flat_dim = depth_t * depth_h * depth_w
        proprio_flat_dim = proprio_t * proprio_dim
        if proprio_t != depth_t:
            raise ValueError(f"proprio_t ({proprio_t}) must equal depth_t ({depth_t}) for per-frame fusion.")
        if num_obs != ego_dim + depth_flat_dim + proprio_flat_dim:
            raise ValueError(
                f"VisionStudentNet expected num_obs={ego_dim + depth_flat_dim + proprio_flat_dim} "
                f"(ego_dim={ego_dim} + depth={depth_flat_dim} + proprio={proprio_flat_dim}), got {num_obs}."
            )

        self.ego_dim = ego_dim
        self.depth_t = depth_t
        self.depth_h = depth_h
        self.depth_w = depth_w
        self.proprio_dim = proprio_dim
        self.proprio_t = proprio_t
        self.depth_flat_dim = depth_flat_dim
        self.far_clip = far_clip

        # (1) Shared per-frame depth CNN (single input channel per frame).
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
        )
        with torch.no_grad():
            conv_out = self.frame_cnn(torch.zeros(1, 1, depth_h, depth_w))
        self.frame_head = nn.Sequential(
            nn.Linear(conv_out.flatten(1).shape[1], frame_latent_dim),
            _activation(activation),
        )

        # (2) 1-D temporal conv over the time axis. Channels = per-frame depth latent
        #     + per-frame proprio (ego-motion). Adaptive pool collapses the (short) time
        #     axis so the head input is fixed regardless of depth_t.
        temporal_in = frame_latent_dim + proprio_dim
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(temporal_in, temporal_dim, kernel_size=3, stride=1, padding=1),
            _activation(activation),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, stride=1, padding=1),
            _activation(activation),
            nn.AdaptiveAvgPool1d(1),
        )

        # (3) Current-ego encoder + fused policy head.
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, ego_latent_dim),
            _activation(activation),
        )
        fused_dim = temporal_dim + ego_latent_dim
        layers: list[nn.Module] = []
        prev = fused_dim
        for h in head_hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.policy_head = nn.Sequential(*layers)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Shared per-frame CNN encode: (N*T, 1, H, W) -> (N*T, frame_latent_dim)."""
        return self.frame_head(self.frame_cnn(frames).flatten(1))

    def encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled temporal-depth latent. Exposed for the Phase-2 aux head."""
        n = x.shape[0]
        depth_flat = x[:, self.ego_dim : self.ego_dim + self.depth_flat_dim]
        proprio_flat = x[:, self.ego_dim + self.depth_flat_dim :]
        depth = depth_flat.view(n, self.depth_t, self.depth_h, self.depth_w) / self.far_clip
        proprio = proprio_flat.view(n, self.proprio_t, self.proprio_dim)
        # Encode every frame with the shared CNN (fold time into the batch dim, so the
        # conv runs on N*T images). That batch is the dominant activation cost, and the
        # distillation update holds it across `gradient_length` accumulated steps -> OOM.
        # Gradient-checkpoint the per-frame encode during training: store only the input
        # and recompute the conv in backward (~10x less activation memory, ~30% more
        # compute). Skipped when grad is off (rollout / ONNX export) so those paths are
        # the plain forward.
        frames = depth.reshape(n * self.depth_t, 1, self.depth_h, self.depth_w)
        if torch.is_grad_enabled():
            frame_latents = torch_checkpoint.checkpoint(self._encode_frames, frames, use_reentrant=False)
        else:
            frame_latents = self._encode_frames(frames)
        frame_latents = frame_latents.view(n, self.depth_t, -1)
        # Fuse per-frame proprio, then temporal-conv over time (N, C, T).
        seq = torch.cat([frame_latents, proprio], dim=-1).transpose(1, 2)
        return self.temporal_conv(seq).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ego = x[:, : self.ego_dim]
        temporal_latent = self.encode_temporal(x)
        ego_latent = self.ego_encoder(ego)
        return self.policy_head(torch.cat([temporal_latent, ego_latent], dim=-1))


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
        proprio_dim: int,
        proprio_t: int,
        frame_latent_dim: int = 64,
        temporal_dim: int = 64,
        ego_latent_dim: int = 128,
        head_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        far_clip: float = 9.0,
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
            proprio_dim=proprio_dim,
            proprio_t=proprio_t,
            frame_latent_dim=frame_latent_dim,
            temporal_dim=temporal_dim,
            ego_latent_dim=ego_latent_dim,
            head_hidden_dims=tuple(head_hidden_dims),
            far_clip=far_clip,
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
