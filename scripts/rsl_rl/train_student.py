# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger distillation: student (ego + 4-frame depth) imitates the locked Phase 1 teacher.

Issue #16 Phase 2. Standalone trainer — bypasses RSL-RL's runner because the env emits a
Dict policy obs that RSL-RL's ActorCritic doesn't natively consume.

Usage (local smoke test, 1 env, 0 iterations — just verifies boot path):
    python scripts/rsl_rl/train_student.py \
        --task=Isaac-Velocity-Rough-Ayg-Student-v0 \
        --teacher_ckpt=logs/rsl_rl/ayg_rough/long_2_good/model_9999.pt \
        --num_envs=1 --max_iterations=0 --enable_cameras --headless

Usage (server training):
    python scripts/rsl_rl/train_student.py \
        --task=Isaac-Velocity-Rough-Ayg-Student-v0 \
        --teacher_ckpt=logs/rsl_rl/ayg_rough/long_2_good/model_9999.pt \
        --num_envs=1024 --max_iterations=1000 --enable_cameras --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger student distillation for AYG rough-terrain.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Ayg-Student-v0")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to the locked Phase 1 teacher .pt file.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--steps_per_env", type=int, default=24, help="Rollout horizon per iteration (per env).")
parser.add_argument("--num_frames", type=int, default=4, help="Depth frame stack depth.")
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--num_epochs", type=int, default=5, help="SGD epochs over each rollout batch.")
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--exploration_std", type=float, default=0.05, help="Gaussian noise on student action during rollout.")
parser.add_argument(
    "--blind",
    action="store_true",
    help="Train the proprio-only baseline: no depth encoder, no depth obs term required.",
)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--logdir", type=str, default=None, help="Override log dir (default: logs/student/<task>/<timestamp>)")
parser.add_argument("--run_name", type=str, default="", help="Optional suffix on the log dir.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cf_lab.tasks  # noqa: E402, F401  # registers Ayg gym envs
from cf_lab.learning.student_actor import (  # noqa: E402
    ACTION_DIM,
    StudentActor,
    TeacherMLP,
    assemble_ego,
    stack_depth,
)


def _resolve_log_dir(task: str, override: str | None, run_name: str) -> str:
    if override is not None:
        return os.path.abspath(override)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        stamp = f"{stamp}_{run_name}"
    return os.path.abspath(os.path.join("logs", "student", task.replace("/", "_"), stamp))


def _ego_term_order(policy_obs: dict[str, torch.Tensor]) -> list[str]:
    # Stable sorted order excluding 'depth'. Captured once so the student MLP head sees
    # a deterministic layout across rollouts.
    return sorted(k for k in policy_obs.keys() if k != "depth")


def _extract_depth(policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    # env emits depth in (N, H, W, 1); squeeze the trailing channel for the (N, H, W) frame.
    return policy_obs["depth"].squeeze(-1)


def main() -> None:
    log_dir = _resolve_log_dir(args_cli.task, args_cli.logdir, args_cli.run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] log_dir={log_dir}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    print(f"[INFO] num_envs={num_envs} device={device}")

    # Reset once to learn the obs layout.
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    teacher_obs = obs["teacher"]
    term_order = _ego_term_order(policy_obs)
    ego_dim = sum(int(policy_obs[k].shape[-1]) for k in term_order)
    teacher_dim = int(teacher_obs.shape[-1])
    if args_cli.blind:
        depth_h = depth_w = 0
        depth_frame = None
        print(f"[INFO] BLIND mode. ego_dim={ego_dim} teacher_dim={teacher_dim} term_order={term_order}")
    else:
        depth_frame = _extract_depth(policy_obs)  # (N, H, W)
        depth_h, depth_w = int(depth_frame.shape[-2]), int(depth_frame.shape[-1])
        print(f"[INFO] ego_dim={ego_dim} depth=({depth_h},{depth_w}) teacher_dim={teacher_dim} term_order={term_order}")

    student = StudentActor(
        ego_dim=ego_dim,
        num_frames=args_cli.num_frames,
        depth_h=max(depth_h, 1),
        depth_w=max(depth_w, 1),
        use_depth=not args_cli.blind,
    ).to(device)
    teacher = TeacherMLP.load_frozen(args_cli.teacher_ckpt, device=device)
    print(
        f"[INFO] student params={sum(p.numel() for p in student.parameters()):,} "
        f"teacher loaded from {args_cli.teacher_ckpt}"
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)
    writer = SummaryWriter(log_dir=log_dir)

    # Per-env circular depth buffer, seeded by the reset frame so the first iteration
    # doesn't see zero frames in the past. In blind mode the buffer is unused.
    if args_cli.blind:
        buf = None
    else:
        buf = depth_frame.unsqueeze(1).expand(-1, args_cli.num_frames, -1, -1).contiguous()  # (N, T, H, W)

    if args_cli.max_iterations == 0:
        # Boot-path smoke test: confirm shapes and one forward pass.
        ego = assemble_ego(policy_obs, term_order)
        action = student(ego, buf) if not args_cli.blind else student(ego)
        target = teacher(teacher_obs)
        loss = torch.mean((action - target) ** 2)
        depth_shape = "n/a (blind)" if args_cli.blind else tuple(buf.shape)
        print(
            f"[SMOKE] ego.shape={tuple(ego.shape)} action.shape={tuple(action.shape)} "
            f"target.shape={tuple(target.shape)} depth_buf={depth_shape} loss={loss.item():.6f}"
        )
        env.close()
        writer.close()
        return

    global_step = 0
    start_time = time.time()
    for it in range(args_cli.max_iterations):
        # ----- Rollout (DAgger): student drives the env, teacher labels each step. -----
        ego_buf = torch.zeros((args_cli.steps_per_env, num_envs, ego_dim), device=device)
        if not args_cli.blind:
            depth_buf = torch.zeros(
                (args_cli.steps_per_env, num_envs, args_cli.num_frames, depth_h, depth_w),
                device=device,
            )
        else:
            depth_buf = None
        target_buf = torch.zeros((args_cli.steps_per_env, num_envs, ACTION_DIM), device=device)

        for t in range(args_cli.steps_per_env):
            policy_obs = obs["policy"]
            teacher_obs = obs["teacher"]
            ego = assemble_ego(policy_obs, term_order)
            if not args_cli.blind:
                depth_frame = _extract_depth(policy_obs)
                buf = stack_depth(buf, depth_frame)

            with torch.no_grad():
                action_mean = student(ego, buf) if not args_cli.blind else student(ego)
                if args_cli.exploration_std > 0.0:
                    action = action_mean + args_cli.exploration_std * torch.randn_like(action_mean)
                else:
                    action = action_mean
                target = teacher(teacher_obs)

            ego_buf[t] = ego
            if not args_cli.blind:
                depth_buf[t] = buf
            target_buf[t] = target

            obs, _, terminated, truncated, _ = env.step(action)
            dones = (terminated | truncated).view(-1)
            if dones.any() and not args_cli.blind:
                # post-reset, the env has already returned the fresh obs in `obs`.
                # zero the history for the envs that just reset so old-episode depth doesn't bleed in.
                fresh = _extract_depth(obs["policy"])
                fresh_stack = fresh.unsqueeze(1).expand(-1, args_cli.num_frames, -1, -1)
                mask = dones.view(-1, 1, 1, 1).to(buf.dtype)
                buf = mask * fresh_stack + (1.0 - mask) * buf

        global_step += args_cli.steps_per_env * num_envs

        # ----- Supervised update on the rollout batch. -----
        ego_flat = ego_buf.view(-1, ego_dim)
        depth_flat = (
            None
            if args_cli.blind
            else depth_buf.view(-1, args_cli.num_frames, depth_h, depth_w)
        )
        target_flat = target_buf.view(-1, ACTION_DIM)
        batch_size = ego_flat.shape[0]
        mb_size = max(1, batch_size // args_cli.num_mini_batches)

        total_loss = 0.0
        steps = 0
        for _ in range(args_cli.num_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                idx = perm[start : start + mb_size]
                if args_cli.blind:
                    pred = student(ego_flat[idx])
                else:
                    pred = student(ego_flat[idx], depth_flat[idx])
                loss = torch.mean((pred - target_flat[idx]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                steps += 1

        avg_loss = total_loss / max(1, steps)

        # ----- Logging. -----
        with torch.no_grad():
            if args_cli.blind:
                per_dim = torch.mean((student(ego_flat) - target_flat) ** 2, dim=0)
            else:
                per_dim = torch.mean((student(ego_flat, depth_flat) - target_flat) ** 2, dim=0)
        writer.add_scalar("loss/action_mse", avg_loss, it)
        writer.add_scalar("loss/grad_norm", float(grad_norm), it)
        writer.add_scalar("rollout/steps", global_step, it)
        writer.add_scalar("rollout/elapsed_s", time.time() - start_time, it)
        for d in range(ACTION_DIM):
            writer.add_scalar(f"loss/per_dim/{d:02d}", float(per_dim[d].item()), it)
        if (it + 1) % 10 == 0 or it == 0:
            print(f"[iter {it:5d}] loss={avg_loss:.6f} grad_norm={float(grad_norm):.3f}")

        # ----- Checkpoint. -----
        if (it + 1) % args_cli.save_interval == 0 or (it + 1) == args_cli.max_iterations:
            ckpt_path = os.path.join(log_dir, f"student_{it+1:06d}.pt")
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "iter": it + 1,
                    "args": vars(args_cli),
                    "term_order": term_order,
                    "ego_dim": ego_dim,
                    "depth_shape": None if args_cli.blind else (depth_h, depth_w),
                    "num_frames": 0 if args_cli.blind else args_cli.num_frames,
                    "use_depth": not args_cli.blind,
                },
                ckpt_path,
            )
            print(f"[INFO] saved {ckpt_path}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
