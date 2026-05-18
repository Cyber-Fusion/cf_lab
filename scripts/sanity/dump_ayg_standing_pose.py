# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the AYG student Play env, let the robot settle under PD control, and
write the settled (joint angles, base z) to /tmp/ayg_standing_pose.json so the
side-view diagram can use ground-truth instead of guessed joint values.

Usage:
    python scripts/sanity/dump_ayg_standing_pose.py --enable_cameras --headless
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump AYG settled standing pose.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Ayg-Student-Play-v0")
parser.add_argument("--settle_steps", type=int, default=120, help="Number of env.step() calls to settle.")
parser.add_argument("--out", type=str, default="/tmp/ayg_standing_pose.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cf_lab.tasks  # noqa: E402, F401


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    body_names = robot.data.body_names

    with torch.inference_mode():
        for _ in range(args_cli.settle_steps):
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, *_ = env.step(action)

    # Sample env 0.
    joint_pos = robot.data.joint_pos[0].cpu().numpy().tolist()
    base_pos_w = robot.data.root_pos_w[0].cpu().numpy().tolist()
    # Body world poses (terrain origin offset removed by subtracting env origin).
    env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
    body_pos_local = (body_pos_w - env_origin[None, :]).tolist()

    foot_idx = {n: body_names.index(n) for n in ("LF_Foot", "RF_Foot", "LH_Foot", "RH_Foot")}
    pose = {
        "task": args_cli.task,
        "settle_steps": args_cli.settle_steps,
        "joint_names": list(joint_names),
        "joint_pos": joint_pos,
        "base_pos_world": base_pos_w,
        "base_z_above_terrain": float(base_pos_w[2] - env_origin[2]),
        "front_foot_x_above_terrain": float(
            0.5 * ((body_pos_w[foot_idx["LF_Foot"], 0] + body_pos_w[foot_idx["RF_Foot"], 0]) - 2 * env_origin[0])
        ),
        "front_foot_z_above_terrain": float(
            0.5 * ((body_pos_w[foot_idx["LF_Foot"], 2] + body_pos_w[foot_idx["RF_Foot"], 2]) - 2 * env_origin[2])
        ),
        "body_names": list(body_names),
        "body_pos_local": body_pos_local,
    }

    with open(args_cli.out, "w") as f:
        json.dump(pose, f, indent=2)
    _log(f"[OK] base z above terrain = {pose['base_z_above_terrain']:.4f} m")
    _log(f"[OK] front foot x = {pose['front_foot_x_above_terrain']:.4f}, z = {pose['front_foot_z_above_terrain']:.4f}")
    for jn, jp in zip(joint_names, joint_pos):
        _log(f"   {jn:10s} = {jp:+.4f}")
    _log(f"[OK] wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
