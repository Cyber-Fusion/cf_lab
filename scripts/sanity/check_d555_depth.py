# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the D555 depth sensor in the Ayg student env.

Runs one Play env, steps a few frames, asserts the depth tensor matches the
D555 contract (shape, dtype, clip range), and dumps a PNG for visual check.

Usage:
    python scripts/sanity/check_d555_depth.py --enable_cameras --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test D555 depth in Ayg student env.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Ayg-Student-Play-v0",
    help="Task ID (default: student play).",
)
parser.add_argument(
    "--out",
    type=str,
    default="/tmp/d555_depth_check.png",
    help="Where to write the depth-frame PNG.",
)
parser.add_argument("--steps", type=int, default=5, help="How many env.step() calls before sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cf_lab.tasks  # noqa: E402, F401  # registers Ayg gym envs


def _depth_tensor(obs):
    """Pick the depth tensor out of whatever obs shape the env emits."""
    if isinstance(obs, dict):
        policy = obs.get("policy", obs)
        if isinstance(policy, dict):
            return policy["depth"]
    raise TypeError(f"unexpected obs structure: {type(obs)}")


def _check_obs_groups(obs) -> None:
    """Assert the two-group DAgger contract: policy(Dict with depth+ego) + teacher(1-D vec)."""
    assert isinstance(obs, dict), f"top-level obs must be Dict (got {type(obs).__name__})"
    assert "policy" in obs, f"missing 'policy' group; keys={list(obs.keys())}"
    assert "teacher" in obs, f"missing 'teacher' group; keys={list(obs.keys())}"

    policy = obs["policy"]
    assert isinstance(policy, dict), (
        f"'policy' must be a Dict because depth can't concatenate with vector terms "
        f"(got {type(policy).__name__})"
    )
    assert "depth" in policy, f"missing 'depth' in policy group; keys={list(policy.keys())}"

    # Aggregate the non-depth (ego) terms — what the student MLP head will see.
    ego_terms = {k: v for k, v in policy.items() if k != "depth"}
    ego_total = sum(int(v.shape[-1]) for v in ego_terms.values())
    _log(f"[CHECK] policy ego terms: {sorted(ego_terms.keys())} total_dim={ego_total}")

    teacher = obs["teacher"]
    assert torch.is_tensor(teacher), f"'teacher' must be a 1-D vector tensor (got {type(teacher).__name__})"
    assert teacher.ndim == 2, f"'teacher' expected (N, K), got shape={tuple(teacher.shape)}"
    teacher_dim = int(teacher.shape[-1])
    _log(f"[CHECK] teacher dim={teacher_dim} (expected 235 = 48 ego + 187 height_scan rays)")
    assert teacher_dim > ego_total, (
        f"teacher dim ({teacher_dim}) should exceed student ego dim ({ego_total}) — "
        "teacher has the extra base_lin_vel + height_scan"
    )


def _log(msg: str) -> None:
    # kit logger writes to fd 1 directly; Python stdout becomes block-buffered
    # under redirect and our prints get lost. Bypass with explicit flush.
    print(msg, flush=True)
    sys.stdout.flush()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    _log(f"[INFO] observation_space: {env.observation_space}")
    _log(f"[INFO] action_space: {env.action_space}")

    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, *_ = env.step(action)

    _check_obs_groups(obs)
    depth = _depth_tensor(obs)
    _log(f"[CHECK] depth.shape={tuple(depth.shape)} dtype={depth.dtype}")
    assert depth.shape[1:] == (45, 80, 1), f"unexpected (H, W, C): {depth.shape}"
    assert depth.dtype == torch.float32, f"unexpected dtype: {depth.dtype}"

    nonzero_count = int((depth > 0).sum().item())
    total = int(depth.numel())
    _log(f"[CHECK] non-zero pixels: {nonzero_count}/{total}")
    finite = depth[depth > 0]
    if finite.numel() == 0:
        raise RuntimeError("depth tensor is entirely zero — camera not rendering")
    dmin, dmax = float(finite.min().item()), float(finite.max().item())
    _log(f"[CHECK] depth(non-zero) min={dmin:.3f} m max={dmax:.3f} m")

    # Row-wise stats: in a correctly-oriented forward-facing depth image,
    # top rows (row 0) should be sky (lots of zeros, low mean of non-zero),
    # bottom rows should be close ground (small depth values).
    img = depth[0, ..., 0]  # (H, W)
    H = img.shape[0]
    for label, r in [("row 0 (top)", 0), (f"row {H // 2} (mid)", H // 2), (f"row {H - 1} (bot)", H - 1)]:
        row = img[r]
        nz = row[row > 0]
        nz_pct = 100.0 * nz.numel() / row.numel()
        nz_mean = float(nz.mean().item()) if nz.numel() > 0 else float("nan")
        _log(f"[CHECK] {label}: non-zero={nz_pct:.0f}% mean(nz)={nz_mean:.2f} m")
    assert dmin >= 0.26 - 1e-3, dmin
    assert dmax <= 9.0 + 1e-3, dmax

    arr = depth[0, ..., 0].cpu().numpy()
    png = (np.clip(arr, 0.0, 9.0) / 9.0 * 255.0).astype(np.uint8)

    try:
        import imageio.v3 as iio

        iio.imwrite(args_cli.out, png)
    except ImportError:
        from PIL import Image

        Image.fromarray(png).save(args_cli.out)
    _log(f"[OK] wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
