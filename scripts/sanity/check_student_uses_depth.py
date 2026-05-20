# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sensitivity test: does the trained vision student actually use depth?

Loads a Distillation checkpoint, runs a Play episode in the student env, and at
each step compares the student's action with the same input where the depth
slice has been zeroed out. If the actions match (MSE near zero) the CNN learned
to ignore depth and we're effectively training a blind policy. If they differ
substantively the CNN is using depth — the open question is then whether what
it's extracting is useful, not whether it has access.

Also dumps a PNG of the most recent depth frame every ``--png_every`` steps so
the depth stream can be eyeballed for renderer issues.

Usage (on the server, inside the venv):
    python scripts/sanity/check_student_uses_depth.py \
        --task=Isaac-Velocity-Rough-Ayg-Student-Play-v0 \
        --checkpoint=logs/rsl_rl/ayg_rough/<run>/model_1000.pt \
        --steps=200 \
        --png_every=20 \
        --enable_cameras --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Depth-sensitivity test for vision student.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Ayg-Student-Play-v0",
    help="Task ID (must match the student env the checkpoint was trained on).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to a Distillation checkpoint .pt (e.g. logs/rsl_rl/.../model_1000.pt).",
)
parser.add_argument("--steps", type=int, default=200, help="Total env steps to roll out.")
parser.add_argument("--png_every", type=int, default=20, help="Dump a depth PNG every N steps.")
parser.add_argument("--png_dir", type=str, default="/tmp/student_depth", help="Where to write PNGs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Clear out the remaining sys.argv so Hydra (loaded transitively) doesn't choke.
sys.argv = sys.argv[:1]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cf_lab.tasks  # noqa: E402, F401  registers the gym envs
from cf_lab.learning.student_teacher_vision import StudentTeacherVision  # noqa: E402

# Must match the slicing pinned in rough_student_env_cfg.py + rsl_rl_distillation_cfg.py.
EGO_DIM = 45
DEPTH_T, DEPTH_H, DEPTH_W = 10, 45, 80
DEPTH_FLAT = DEPTH_T * DEPTH_H * DEPTH_W
EXPECTED_POLICY_DIM = EGO_DIM + DEPTH_FLAT


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def _build_policy(env, checkpoint_path: str, device: torch.device) -> StudentTeacherVision:
    """Build a `StudentTeacherVision` matching the train-time architecture and
    load the actor weights from the checkpoint.

    We bypass RSL-RL's DistillationRunner entirely — it would expect the teacher
    obs group to be wired up and would try to load the teacher checkpoint too.
    For this diagnostic we only need the student forward pass.
    """
    # Probe one rollout to get the obs shapes the policy was trained with.
    obs, _ = env.reset()
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}

    # Wrap as TensorDict-compatible obs dict (StudentTeacher only inspects shapes here).
    num_actions = int(env.action_space.shape[-1])

    policy = StudentTeacherVision(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=num_actions,
        ego_dim=EGO_DIM,
        depth_t=DEPTH_T,
        depth_h=DEPTH_H,
        depth_w=DEPTH_W,
        depth_latent_dim=64,
        ego_latent_dim=128,
        head_hidden_dims=(256, 128),
        student_hidden_dims=(256, 256, 256),
        teacher_hidden_dims=(512, 256, 128),
        activation="elu",
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # `StudentTeacherVision.load_state_dict` inherits the auto-detect that handles
    # both Distillation checkpoints and ActorCritic teacher checkpoints.
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


def _save_depth_png(depth_frame: torch.Tensor, path: str) -> None:
    """`depth_frame` is (H, W) raw meters; clip to [0, 9] and write 8-bit grayscale."""
    arr = depth_frame.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 9.0) / 9.0 * 255.0
    arr = arr.astype(np.uint8)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, arr)
    except ImportError:
        from PIL import Image

        Image.fromarray(arr).save(path)


def main() -> None:
    os.makedirs(args_cli.png_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    _log(f"[INFO] env: {args_cli.task}  device={device}")
    _log(f"[INFO] observation_space: {env.observation_space}")

    policy = _build_policy(env, args_cli.checkpoint, device)
    _log(f"[INFO] loaded checkpoint: {args_cli.checkpoint}")
    n_params = sum(p.numel() for p in policy.student.parameters())
    _log(f"[INFO] student params: {n_params:,}")

    obs, _ = env.reset()
    mse_list: list[float] = []
    max_diff_list: list[float] = []
    depth_nonzero_pct: list[float] = []
    depth_mean_nz: list[float] = []

    with torch.inference_mode():
        for step in range(args_cli.steps):
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            assert policy_obs.shape[-1] == EXPECTED_POLICY_DIM, (
                f"policy obs has {policy_obs.shape[-1]} dims, expected {EXPECTED_POLICY_DIM}"
            )

            # Real depth: forward through trained student.
            action_real = policy.student(policy_obs)

            # Zeroed depth: keep ego, blank depth slice.
            policy_obs_zeroed = policy_obs.clone()
            policy_obs_zeroed[:, EGO_DIM:] = 0.0
            action_zeroed = policy.student(policy_obs_zeroed)

            diff = (action_real - action_zeroed).float()
            mse = float((diff**2).mean().item())
            max_diff = float(diff.abs().max().item())
            mse_list.append(mse)
            max_diff_list.append(max_diff)

            # Sanity-check the depth stream itself.
            depth_flat = policy_obs[:, EGO_DIM:]
            stack = depth_flat.view(args_cli.num_envs, DEPTH_T, DEPTH_H, DEPTH_W)
            recent = stack[0, -1]  # most recent frame
            nonzero = recent[recent > 0]
            nz_pct = 100.0 * nonzero.numel() / recent.numel()
            nz_mean = float(nonzero.mean().item()) if nonzero.numel() > 0 else float("nan")
            depth_nonzero_pct.append(nz_pct)
            depth_mean_nz.append(nz_mean)

            if step % args_cli.png_every == 0:
                _log(
                    f"[STEP {step:03d}] mse(real,zero)={mse:.5f}  max|diff|={max_diff:.4f}"
                    f"  depth nz={nz_pct:.0f}%  mean(nz)={nz_mean:.2f}m"
                )
                _save_depth_png(recent, os.path.join(args_cli.png_dir, f"depth_{step:04d}.png"))

            # Step env with the REAL action (so the rollout follows the trained policy).
            obs, *_ = env.step(action_real)

    mean_mse = float(np.mean(mse_list))
    mean_max = float(np.mean(max_diff_list))
    p95_mse = float(np.percentile(mse_list, 95))
    mean_nz = float(np.mean(depth_nonzero_pct))
    mean_mean_depth = float(np.nanmean(depth_mean_nz))

    _log("")
    _log("===== summary =====")
    _log(f"steps:                  {args_cli.steps}")
    _log(f"depth nz coverage:      mean={mean_nz:.1f}%  expected >20% if rendering is fine")
    _log(f"depth mean(non-zero):   {mean_mean_depth:.2f} m  expected 1–5 m typical")
    _log(f"action MSE(real, zero): mean={mean_mse:.5f}  p95={p95_mse:.5f}")
    _log(f"action max|diff|:       mean={mean_max:.4f}")
    _log("")
    _log("interpretation:")
    if mean_mse < 1e-4:
        _log("  → CNN is IGNORING depth (actions identical with/without depth).")
        _log("    Camera signal not flowing into the policy. Root cause is training/")
        _log("    curriculum, not the camera. Check terrain_levels stuck at 0.")
    elif mean_mse < 1e-2:
        _log("  → CNN uses depth WEAKLY. Some sensitivity, but small. Likely the")
        _log("    CNN extracted shallow features that don't change actions much.")
    else:
        _log("  → CNN USES depth meaningfully. Camera info is reaching the policy.")
        _log("    If survival is still bad, the camera info itself is insufficient")
        _log("    (geometry / temporal memory), not a wiring bug.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
