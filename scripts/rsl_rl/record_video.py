# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record a cinematic video of a trained RSL-RL policy.

Features:
- Still phase: close-up tracking of a single robot (--follow_robot, --follow_offset)
- Zoom-out phase: smooth transition to wide cinematic shot with orbit rotation and ease-out
- Hold phase: wide shot at final rotation angle
- Optional robot centroid tracking for wide shot (--follow)
- Command text overlay (--show_commands)
- No env resets during recording (terminations disabled, long episode)
- Parametrized video length, rotation angle, and phase fractions

Note: Do NOT use --headless. The RGB annotator is broken in headless mode
with this Isaac Sim version. Run with a display (the window will open briefly).
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
parser = argparse.ArgumentParser(description="Record a cinematic video of a trained RSL-RL policy.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps). ~20s at 50Hz.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments (robots) to simulate.")
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
    "--eye_start",
    type=str,
    default="(3.0, 3.0, 2.0)",
    help="Camera start position as '(x, y, z)'. Default: '(3.0, 3.0, 2.0)'.",
)
parser.add_argument(
    "--eye_end",
    type=str,
    default="(12.0, 12.0, 6.0)",
    help="Camera end position as '(x, y, z)'. Default: '(12.0, 12.0, 6.0)'.",
)
parser.add_argument(
    "--lookat",
    type=str,
    default="(0.0, 0.0, 0.0)",
    help="Camera look-at target as '(x, y, z)'. Default: '(0.0, 0.0, 0.0)'.",
)
parser.add_argument(
    "--follow",
    action="store_true",
    default=False,
    help="Track centroid of all robots instead of fixed --lookat target.",
)
parser.add_argument(
    "--follow_robot",
    type=int,
    default=0,
    help="Index of robot to follow during the still phase (close-up tracking). Default: 0.",
)
parser.add_argument(
    "--follow_offset",
    type=str,
    default="(1.5, 1.5, 0.8)",
    help="Camera offset relative to followed robot during still phase as '(x, y, z)'. Default: '(1.5, 1.5, 0.8)'.",
)
parser.add_argument(
    "--still_fraction",
    type=float,
    default=0.15,
    help="Fraction of video where camera stays still (close). Default: 0.15.",
)
parser.add_argument(
    "--zoom_fraction",
    type=float,
    default=0.35,
    help="Fraction of video for the zoom-out transition. The remainder is held at the far position. Default: 0.35.",
)
parser.add_argument(
    "--show_commands",
    action="store_true",
    default=False,
    help="Overlay velocity command text on the video.",
)
parser.add_argument(
    "--rotation_degrees",
    type=float,
    default=120.0,
    help="Total camera orbit rotation around the target (degrees). Default: 120.",
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


def ease_out(t: float) -> float:
    """Ease-out interpolation using cosine curve. t in [0, 1] -> [0, 1]."""
    return 1.0 - math.cos(t * math.pi / 2.0)


def rotate_around_z(offset: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an XYZ offset vector around the Z-axis by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = offset[0], offset[1], offset[2]
    return np.array([c * x - s * y, s * x + c * y, z])


def overlay_commands(frame, env_unwrapped):
    """Overlay velocity command text on a frame. Returns frame (possibly modified)."""
    try:
        import cv2
    except ImportError:
        return frame

    frame = np.copy(frame)
    try:
        cmd = env_unwrapped.command_manager.get_command("base_velocity")[0]
        vx, vy, wz = cmd[0].item(), cmd[1].item(), cmd[2].item()
        text = f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}"
    except (AttributeError, KeyError, IndexError):
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    pos = (15, 35)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


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
    """Record a cinematic video of a trained RSL-RL policy."""
    # parse camera positions
    eye_start = np.array(parse_tuple(args_cli.eye_start))
    eye_end = np.array(parse_tuple(args_cli.eye_end))
    lookat = np.array(parse_tuple(args_cli.lookat))
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

    # set the log directory for the environment
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

    # compute camera parameters (3 phases: still close -> zoom out -> hold far)
    still_steps = int(args_cli.video_length * args_cli.still_fraction)
    zoom_steps = int(args_cli.video_length * args_cli.zoom_fraction)
    total_rotation_rad = math.radians(args_cli.rotation_degrees)

    # reset environment
    obs = env.get_observations()

    # helper to get a single robot's position
    def get_robot_pos(idx: int) -> np.ndarray | None:
        try:
            return env.unwrapped.scene["robot"].data.root_pos_w[idx].cpu().numpy()
        except (AttributeError, KeyError, IndexError):
            return None

    # wide-shot target: centroid (--follow) or fixed lookat
    if args_cli.follow:
        try:
            root_pos = env.unwrapped.scene["robot"].data.root_pos_w  # (num_envs, 3)
            wide_target = root_pos.mean(dim=0).cpu().numpy()
            print(f"[INFO] Follow mode: wide-shot centroid at {wide_target}")
        except (AttributeError, KeyError):
            print("[WARN] Could not get robot positions. Falling back to fixed lookat.")
            wide_target = lookat
    else:
        wide_target = lookat

    # set initial camera on the followed robot
    robot_pos = get_robot_pos(args_cli.follow_robot)
    if robot_pos is not None:
        init_target = robot_pos
        init_eye = robot_pos + follow_offset
    else:
        print("[WARN] Could not get followed robot position. Using eye_start/lookat.")
        init_target = wide_target
        init_eye = eye_start
    env.unwrapped.sim.set_camera_view(eye=init_eye, target=init_target)

    frames = []
    timestep = 0
    print(f"[INFO] Recording {args_cli.video_length} steps...")
    # simulate environment and animate camera
    while simulation_app.is_running():
        if timestep < still_steps:
            # --- still phase: close-up tracking a single robot ---
            robot_pos = get_robot_pos(args_cli.follow_robot)
            if robot_pos is not None:
                target = robot_pos
                eye = robot_pos + follow_offset
            else:
                target = wide_target
                eye = eye_start.copy()
        elif timestep < still_steps + zoom_steps:
            # --- zoom-out phase: transition from robot follow to wide cinematic shot ---
            t_linear = (timestep - still_steps) / max(zoom_steps - 1, 1)
            t_eased = ease_out(min(t_linear, 1.0))
            # snapshot robot position at transition start
            if timestep == still_steps:
                robot_pos = get_robot_pos(args_cli.follow_robot)
                transition_start_target = robot_pos if robot_pos is not None else wide_target
                transition_start_eye = (robot_pos + follow_offset) if robot_pos is not None else eye_start.copy()
            # interpolate target: robot -> wide target
            target = transition_start_target + t_eased * (wide_target - transition_start_target)
            # interpolate eye: robot close-up -> eye_end (with rotation)
            eye = transition_start_eye + t_eased * (eye_end - transition_start_eye)
            # apply rotation
            angle = t_eased * total_rotation_rad
            eye = target + rotate_around_z(eye - target, angle)
        else:
            # --- hold phase: wide cinematic shot at final rotation ---
            target = wide_target
            eye = target + rotate_around_z(eye_end - wide_target, total_rotation_rad)

        env.unwrapped.sim.set_camera_view(eye=eye, target=target)

        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # capture frame
        frame = env.env.render()
        if frame is not None:
            if args_cli.show_commands:
                frame = overlay_commands(frame, env.unwrapped)
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
