# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record a cinematic terrain tour of a trained RSL-RL policy.

Tours every robot in the scene one-by-one with a close-following camera and a gentle
orbit, so each terrain tile gets airtime. Smooth eased transitions between robots.

- Default 60 s video (3000 steps @ 50 Hz).
- No env resets during recording (terminations disabled, long episode).
- Optional overlays: velocity command (--show_commands) and tour index (--show_tour_info).

Note: Do NOT use --headless. The RGB annotator is broken in headless mode with this
Isaac Sim version. Run with a display (the window will open briefly).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import ast
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Record a cinematic terrain tour of a trained RSL-RL policy.")
parser.add_argument(
    "--video_length",
    type=int,
    default=3000,
    help="Length of the recorded video (in steps). Default 3000 = ~60 s at 50 Hz.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=8,
    help="Number of environments (robots) to simulate. The tour visits each one.",
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# camera arguments
parser.add_argument(
    "--follow_offset",
    type=str,
    default="(2.0, 2.0, 1.2)",
    help="Camera offset relative to the currently-followed robot as '(x, y, z)'. Default: '(2.0, 2.0, 1.2)'.",
)
parser.add_argument(
    "--orbit_degrees_per_robot",
    type=float,
    default=40.0,
    help="Camera orbit around each robot during its hold segment (degrees). Default: 40.",
)
parser.add_argument(
    "--transition_fraction",
    type=float,
    default=0.15,
    help="Fraction of each per-robot segment spent transitioning to the next robot. Default: 0.15.",
)
parser.add_argument(
    "--show_commands",
    action="store_true",
    default=False,
    help="Overlay velocity command text on the video (for the currently-followed robot).",
)
parser.add_argument(
    "--show_tour_info",
    action="store_true",
    default=False,
    help="Overlay 'Robot i/N' tour index on the video.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras for video recording
args_cli.enable_cameras = True

if args_cli.headless:
    print("[WARN] --headless is not supported for video recording (Isaac Sim annotator bug). Ignoring --headless.")
    args_cli.headless = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import cf_lab.tasks  # noqa: F401


def parse_tuple(s: str) -> tuple[float, float, float]:
    """Parse a string like '(1.0, 2.0, 3.0)' into a tuple of floats."""
    result = ast.literal_eval(s)
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(f"Expected a tuple of 3 floats, got: {s}")
    return tuple(float(x) for x in result)


def ease_in_out(t: float) -> float:
    """Smooth cosine ease-in-out. t in [0, 1] -> [0, 1]."""
    return 0.5 - 0.5 * math.cos(t * math.pi)


def rotate_around_z(offset: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an XYZ offset vector around the Z-axis by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = offset[0], offset[1], offset[2]
    return np.array([c * x - s * y, s * x + c * y, z])


def overlay_text(frame, lines):
    """Stack short text lines in the top-left corner of the frame."""
    try:
        import cv2
    except ImportError:
        return frame
    if not lines:
        return frame
    frame = np.copy(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    pad = 5
    y = 35
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        x = 15
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += th + baseline + 12
    return frame


def velocity_cmd_text(env_unwrapped, robot_idx: int):
    """Return 'vx=... vy=... wz=...' for the given robot's base_velocity command, or None."""
    try:
        cmd = env_unwrapped.command_manager.get_command("base_velocity")[robot_idx]
        vx, vy, wz = cmd[0].item(), cmd[1].item(), cmd[2].item()
        return f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}"
    except (AttributeError, KeyError, IndexError):
        return None


def disable_terminations(env_cfg):
    """Disable all non-timeout terminations and set a very long episode length."""
    env_cfg.episode_length_s = 10000.0

    if not hasattr(env_cfg, "terminations"):
        return

    terminations = env_cfg.terminations
    for attr_name in list(vars(terminations)):
        if attr_name.startswith("_"):
            continue
        if attr_name == "time_out":
            continue
        setattr(terminations, attr_name, None)

    print("[INFO] Disabled non-timeout terminations. Episode length set to 10000s.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Record a cinematic terrain tour of a trained RSL-RL policy."""
    follow_offset = np.array(parse_tuple(args_cli.follow_offset))

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # disable terminations so the env never resets during recording
    disable_terminations(env_cfg)

    # disable interval pushes for cleaner video
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment with rgb_array render mode for video capture
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # video output setup
    video_dir = os.path.join(log_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # tour bookkeeping
    num_envs = args_cli.num_envs
    segment_steps = max(1, args_cli.video_length // num_envs)
    transition_steps = max(1, int(segment_steps * args_cli.transition_fraction))
    hold_steps = max(1, segment_steps - transition_steps)
    orbit_per_robot_rad = math.radians(args_cli.orbit_degrees_per_robot)

    # reset environment
    obs = env.get_observations()

    def get_robot_pos(idx: int) -> np.ndarray | None:
        try:
            return env.unwrapped.scene["robot"].data.root_pos_w[idx].cpu().numpy()
        except (AttributeError, KeyError, IndexError):
            return None

    # initial camera placement on robot 0
    init_pos = get_robot_pos(0)
    if init_pos is not None:
        env.unwrapped.sim.set_camera_view(eye=init_pos + follow_offset, target=init_pos)

    frames = []
    timestep = 0
    print(
        f"[INFO] Tour: {num_envs} robots × {segment_steps} steps "
        f"({hold_steps} hold + {transition_steps} transition). Total {args_cli.video_length} steps."
    )
    # simulate environment and animate camera through the tour
    while simulation_app.is_running():
        segment_idx = min(timestep // segment_steps, num_envs - 1)
        step_in_segment = timestep - segment_idx * segment_steps
        current_robot = segment_idx
        next_robot = min(segment_idx + 1, num_envs - 1)

        cur_pos = get_robot_pos(current_robot)
        next_pos = get_robot_pos(next_robot)

        if step_in_segment < hold_steps:
            # --- hold phase: close-up on current robot with slight orbit ---
            t_hold = step_in_segment / max(hold_steps - 1, 1)
            angle = t_hold * orbit_per_robot_rad
            target = cur_pos if cur_pos is not None else np.zeros(3)
            eye = target + rotate_around_z(follow_offset, angle)
        else:
            # --- transition phase: morph from current close-up to next close-up ---
            t = (step_in_segment - hold_steps) / max(transition_steps, 1)
            t_eased = ease_in_out(min(t, 1.0))
            cur_t = cur_pos if cur_pos is not None else np.zeros(3)
            nxt_t = next_pos if next_pos is not None else cur_t
            cur_e = cur_t + rotate_around_z(follow_offset, orbit_per_robot_rad)
            nxt_e = nxt_t + follow_offset
            target = cur_t + t_eased * (nxt_t - cur_t)
            eye = cur_e + t_eased * (nxt_e - cur_e)

        env.unwrapped.sim.set_camera_view(eye=eye, target=target)

        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # capture frame
        frame = env.env.render()
        if frame is not None:
            overlay_lines = []
            if args_cli.show_tour_info:
                overlay_lines.append(f"Robot {current_robot + 1}/{num_envs}")
            if args_cli.show_commands:
                cmd_line = velocity_cmd_text(env.unwrapped, current_robot)
                if cmd_line is not None:
                    overlay_lines.append(cmd_line)
            if overlay_lines:
                frame = overlay_text(frame, overlay_lines)
            frames.append(frame)

        timestep += 1
        if timestep == args_cli.video_length:
            break

    # write video
    if frames:
        import imageio

        fps = int(1.0 / env.unwrapped.step_dt)
        video_path = os.path.join(video_dir, f"record_{train_task_name}.mp4")
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"[INFO] Video saved to: {video_path} ({len(frames)} frames, {fps} FPS)")
    else:
        print("[WARN] No frames captured.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
