# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a distilled student checkpoint produced by train_student.py.

The student .pt is not an RSL-RL ActorCritic, so the stock play.py cannot load it.
This script rebuilds StudentActor from the saved metadata (ego_dim, depth_shape,
num_frames, use_depth, term_order) and runs the env loop deterministically with
the mean action.

Usage (blind):
    python scripts/rsl_rl/play_student.py \
        --task=Isaac-Velocity-Rough-Ayg-Student-Blind-Play-v0 \
        --checkpoint=logs/student/ayg_rough_blind/2026-05-18_13-08-00_blind_v1/student_001000.pt \
        --num_envs=64

Usage (vision):
    python scripts/rsl_rl/play_student.py \
        --task=Isaac-Velocity-Rough-Ayg-Student-Play-v0 \
        --checkpoint=logs/student/<...>/student_XXXXXX.pt \
        --num_envs=64 --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a distilled student checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student_*.pt produced by train_student.py.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cf_lab.tasks  # noqa: E402, F401
from cf_lab.learning.student_actor import StudentActor, assemble_ego, stack_depth  # noqa: E402


def _extract_depth(policy_obs):
    return policy_obs["depth"].squeeze(-1)


def main() -> None:
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    use_depth = ckpt.get("use_depth", True)
    term_order = ckpt["term_order"]
    ego_dim = int(ckpt["ego_dim"])
    num_frames = int(ckpt.get("num_frames", 0)) or 1
    depth_shape = ckpt.get("depth_shape")
    depth_h, depth_w = (depth_shape if depth_shape else (1, 1))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    student = StudentActor(
        ego_dim=ego_dim,
        num_frames=num_frames,
        depth_h=depth_h,
        depth_w=depth_w,
        use_depth=use_depth,
    ).to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()
    print(
        f"[INFO] loaded student from {args_cli.checkpoint} "
        f"(use_depth={use_depth}, ego_dim={ego_dim}, num_frames={num_frames}, depth_shape={depth_shape})"
    )

    obs, _ = env.reset()
    if use_depth:
        depth_frame = _extract_depth(obs["policy"])
        buf = depth_frame.unsqueeze(1).expand(-1, num_frames, -1, -1).contiguous()
    else:
        buf = None

    with torch.inference_mode():
        while simulation_app.is_running():
            policy_obs = obs["policy"]
            ego = assemble_ego(policy_obs, term_order)
            if use_depth:
                depth_frame = _extract_depth(policy_obs)
                buf = stack_depth(buf, depth_frame)
                action = student(ego, buf)
            else:
                action = student(ego)
            obs, _, terminated, truncated, _ = env.step(action)
            dones = (terminated | truncated).view(-1)
            if use_depth and dones.any():
                fresh = _extract_depth(obs["policy"])
                fresh_stack = fresh.unsqueeze(1).expand(-1, num_frames, -1, -1)
                mask = dones.view(-1, 1, 1, 1).to(buf.dtype)
                buf = mask * fresh_stack + (1.0 - mask) * buf

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
