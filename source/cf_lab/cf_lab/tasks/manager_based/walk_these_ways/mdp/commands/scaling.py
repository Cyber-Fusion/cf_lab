# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared speed-coupling helpers for the WTW command terms."""

from __future__ import annotations

import torch


def vx_scale(vx_cmd: torch.Tensor, knee: float = 1.0) -> torch.Tensor:
    """Uniform speed-coupling factor ``1 / max(1, |vx| / knee)``.

    Applied to the *secondary* commands (``vy``, ``omega``, ``pitch``, ``roll``, and the
    base-height deviation) so the policy is never asked to track aggressive secondary
    objectives while the commanded forward speed is high.
    """
    return 1.0 / (vx_cmd.abs() / knee).clamp(min=1.0)
