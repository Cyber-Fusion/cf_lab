# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a multi-shot cinematic video of a distilled student policy.

Loads a StudentActor checkpoint (the way play_student.py does) and produces a
multi-shot video instead of a single still + zoom-out:

  * N close-orbit shots, each on a different robot picked from a different
    terrain tile (spread across env_origins). This is what makes the video
    actually showcase "the policy walking on different terrains" instead of
    framing one robot at origin.
  * Final wide pull-back shot that auto-frames the whole arena and orbits
    slowly around the centroid, so all robots are visible at once.

Usage (blind):
    python scripts/rsl_rl/record_video_student.py \
        --task=Isaac-Velocity-Rough-Ayg-Student-Blind-Play-v0 \
        --checkpoint=logs/student/ayg_rough_blind/<run>/student_001000.pt \
        --num_envs=64

Note: Do NOT use --headless. The RGB annotator is broken in headless mode with
this Isaac Sim version. Run with a display (the window opens briefly).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import ast
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a multi-shot cinematic video of a distilled student policy.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student_*.pt produced by train_student.py.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments (robots) to simulate.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps). 3000 ~= 60s at 50Hz.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Directory to save the video. Defaults to '<checkpoint_dir>/videos'.",
)
# multi-shot cinematic args
parser.add_argument("--num_shots", type=int, default=4, help="Number of close-orbit shots (each on a different robot).")
parser.add_argument(
    "--shot_length_s",
    type=float,
    default=7.0,
    help="Seconds per close-orbit shot. Total close-up time = num_shots * shot_length_s; remainder is the finale.",
)
parser.add_argument(
    "--shot_robots",
    type=str,
    default=None,
    help="Comma-separated env IDs to feature in the shots (e.g. '0,15,30,45'). If unset, indices are spread evenly across num_envs.",
)
parser.add_argument(
    "--shot_offset",
    type=str,
    default="(1.8, 1.8, 0.9)",
    help="Camera offset relative to followed robot during close-orbit shots as '(x, y, z)'.",
)
parser.add_argument(
    "--orbit_per_shot_deg",
    type=float,
    default=180.0,
    help="Degrees of orbit around the followed robot per close-orbit shot.",
)
parser.add_argument(
    "--finale_orbit_deg",
    type=float,
    default=90.0,
    help="Degrees of orbit during the wide finale.",
)
parser.add_argument(
    "--finale_radius",
    type=float,
    default=None,
    help="Override finale horizontal distance from centroid (meters). If unset, auto-computed from env spread.",
)
parser.add_argument(
    "--finale_height",
    type=float,
    default=None,
    help="Override finale camera height (meters). If unset, auto-computed from env spread.",
)
parser.add_argument(
    "--show_commands",
    action="store_true",
    default=False,
    help="Overlay velocity command text on the video.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras for video recording
args_cli.enable_cameras = True
if args_cli.headless:
    print("[WARN] --headless is not supported for video recording (Isaac Sim annotator bug). Ignoring --headless.")
    args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cf_lab.tasks  # noqa: E402, F401
from cf_lab.learning.student_actor import StudentActor, assemble_ego, stack_depth  # noqa: E402


def parse_tuple(s: str) -> tuple[float, float, float]:
    result = ast.literal_eval(s)
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(f"Expected a tuple of 3 floats, got: {s}")
    return tuple(float(x) for x in result)


def ease_in_out(t: float) -> float:
    # smoothstep
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def rotate_around_z(offset: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = offset[0], offset[1], offset[2]
    return np.array([c * x - s * y, s * x + c * y, z])


def overlay_commands(frame, env_unwrapped):
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


def _extract_depth(policy_obs):
    return policy_obs["depth"].squeeze(-1)


def pick_spread_indices(env_origins_xy: np.ndarray, num: int) -> list[int]:
    """Pick `num` env indices spread across the terrain field via farthest-point sampling."""
    n = env_origins_xy.shape[0]
    if num >= n:
        return list(range(n))
    # seed with the env closest to the bbox corner so we start at one end
    bbox_min = env_origins_xy.min(axis=0)
    seed = int(np.argmin(np.linalg.norm(env_origins_xy - bbox_min, axis=1)))
    chosen = [seed]
    dist = np.linalg.norm(env_origins_xy - env_origins_xy[seed], axis=1)
    while len(chosen) < num:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        new_d = np.linalg.norm(env_origins_xy - env_origins_xy[nxt], axis=1)
        dist = np.minimum(dist, new_d)
    return chosen


def main():
    shot_offset = np.array(parse_tuple(args_cli.shot_offset))

    # load student metadata from checkpoint
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    use_depth = ckpt.get("use_depth", True)
    term_order = ckpt["term_order"]
    ego_dim = int(ckpt["ego_dim"])
    num_frames = int(ckpt.get("num_frames", 0)) or 1
    depth_shape = ckpt.get("depth_shape")
    depth_h, depth_w = (depth_shape if depth_shape else (1, 1))

    # build env cfg, then strip terminations & pushes for a clean take
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    disable_terminations(env_cfg)
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    device = env.unwrapped.device

    # build student and load weights
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

    # resolve video output dir
    if args_cli.video_dir is not None:
        video_dir = os.path.abspath(args_cli.video_dir)
    else:
        video_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "videos")
    os.makedirs(video_dir, exist_ok=True)

    # reset and initialize depth buffer
    obs, _ = env.reset()
    if use_depth:
        depth_frame = _extract_depth(obs["policy"])
        buf = depth_frame.unsqueeze(1).expand(-1, num_frames, -1, -1).contiguous()
    else:
        buf = None

    # ---- plan the shot list ----
    env_origins_xy = env.unwrapped.scene.env_origins[:, :2].cpu().numpy()
    if args_cli.shot_robots:
        shot_indices = [int(x) for x in args_cli.shot_robots.split(",") if x.strip()]
    else:
        shot_indices = pick_spread_indices(env_origins_xy, args_cli.num_shots)
    num_shots = len(shot_indices)
    print(f"[INFO] Shot robots: {shot_indices}")

    step_dt = env.unwrapped.step_dt
    fps = int(1.0 / step_dt)

    shot_steps = int(args_cli.shot_length_s * fps)
    if shot_steps * num_shots >= args_cli.video_length:
        # If shots alone would exceed length, shrink them so finale gets ~20%.
        shot_steps = int(0.8 * args_cli.video_length / num_shots)
    finale_steps = args_cli.video_length - shot_steps * num_shots
    print(f"[INFO] Plan: {num_shots} close-orbit shots x {shot_steps} steps + {finale_steps}-step finale "
          f"({args_cli.video_length} steps total, {fps} FPS)")

    # ---- finale camera (auto-frame the scene) ----
    bbox_min = env_origins_xy.min(axis=0)
    bbox_max = env_origins_xy.max(axis=0)
    spread = float(np.linalg.norm(bbox_max - bbox_min))
    centroid_xy = (bbox_min + bbox_max) / 2.0
    finale_center = np.array([centroid_xy[0], centroid_xy[1], 0.5])
    finale_radius = args_cli.finale_radius if args_cli.finale_radius is not None else max(6.0, spread * 0.55 + 6.0)
    finale_height = args_cli.finale_height if args_cli.finale_height is not None else max(4.0, spread * 0.35 + 4.0)
    print(f"[INFO] Scene spread {spread:.1f}m; finale orbit radius {finale_radius:.1f}m, height {finale_height:.1f}m")

    orbit_per_shot_rad = math.radians(args_cli.orbit_per_shot_deg)
    finale_orbit_rad = math.radians(args_cli.finale_orbit_deg)

    def get_robot_pos(idx: int):
        try:
            return env.unwrapped.scene["robot"].data.root_pos_w[idx].cpu().numpy()
        except (AttributeError, KeyError, IndexError):
            return None

    # initial camera pose (first shot's robot)
    rp0 = get_robot_pos(shot_indices[0])
    init_target = rp0 if rp0 is not None else finale_center
    init_eye = (rp0 + rotate_around_z(shot_offset, -0.5 * orbit_per_shot_rad)) if rp0 is not None else (
        finale_center + np.array([finale_radius, 0.0, finale_height])
    )
    env.unwrapped.sim.set_camera_view(eye=init_eye, target=init_target)

    frames = []
    timestep = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            shots_total = shot_steps * num_shots
            if timestep < shots_total:
                shot_idx = timestep // shot_steps
                t_in_shot = (timestep - shot_idx * shot_steps) / max(shot_steps - 1, 1)
                t_eased = ease_in_out(t_in_shot)
                robot_idx = shot_indices[shot_idx]
                rp = get_robot_pos(robot_idx)
                if rp is None:
                    rp = finale_center
                target = rp
                # orbit centered: start at -orbit/2, end at +orbit/2
                angle = (t_eased - 0.5) * orbit_per_shot_rad
                eye = target + rotate_around_z(shot_offset, angle)
            else:
                t_finale = (timestep - shots_total) / max(finale_steps - 1, 1)
                t_eased = ease_in_out(t_finale)
                angle = t_eased * finale_orbit_rad
                target = finale_center
                base_eye = np.array([finale_radius, 0.0, finale_height])
                eye = target + rotate_around_z(base_eye, angle)

            env.unwrapped.sim.set_camera_view(eye=eye, target=target)

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

            frame = env.render()
            if frame is not None:
                if args_cli.show_commands:
                    frame = overlay_commands(frame, env.unwrapped)
                frames.append(frame)

            timestep += 1
            if timestep == args_cli.video_length:
                break

    if frames:
        import imageio

        ckpt_stem = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        video_path = os.path.join(video_dir, f"record_student_{ckpt_stem}.mp4")
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"[INFO] Video saved to: {video_path} ({len(frames)} frames, {fps} FPS)")
    else:
        print("[WARN] No frames captured.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
