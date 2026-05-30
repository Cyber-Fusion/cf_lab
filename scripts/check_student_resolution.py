# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot sanity check for the vision-student depth resolution change.

Boots `Isaac-Velocity-Rough-Ayg-Student-Play-v0` headless with 2 envs, takes a
single zero-action step, and prints:
  * env.observation_space shapes for both `policy` and `teacher` groups
  * the raw depth-camera tensor shape (N, H, W) before flattening
  * the concatenated policy vector length

Pass criteria for the strided-history vision student:
  * policy obs length == 45 (ego) + 10 (strided frames) * 60 (H) * 80 (W)
        + 10 (proprio frames) * 21 (proprio dim) = 48255
  * depth_camera distance_to_image_plane tensor reports H=60, W=80
  * teacher obs length unchanged (no impact on privileged path)

Run from cf_lab/ with the venv active. Required env vars set up by the script.
"""

import argparse
import os

# Match the project's standard threading neutralization before any heavy imports.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision-student resolution sanity check.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Ayg-Student-Play-v0",
    help="Defaults to the play variant for fastest boot.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Force headless regardless of CLI default.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cf_lab.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

EXPECTED_EGO = 45
EXPECTED_T = 10  # strided depth frames (stride 4)
EXPECTED_H = 60
EXPECTED_W = 80
EXPECTED_PROPRIO_DIM = 21
EXPECTED_PROPRIO_T = 10
EXPECTED_DEPTH_LEN = EXPECTED_T * EXPECTED_H * EXPECTED_W
EXPECTED_PROPRIO_LEN = EXPECTED_PROPRIO_T * EXPECTED_PROPRIO_DIM
EXPECTED_POLICY_LEN = EXPECTED_EGO + EXPECTED_DEPTH_LEN + EXPECTED_PROPRIO_LEN


def _fmt(shape):
    return tuple(int(s) for s in shape)


VERDICT_PATH = "/tmp/check_student_resolution_verdict.txt"


def _emit(msg: str, fh) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    fh = open(VERDICT_PATH, "w")
    _emit("\n=== Observation spaces ===", fh)
    _emit(f"observation_space: {env.observation_space}", fh)
    _emit(f"action_space: {env.action_space}", fh)

    obs, _info = env.reset()
    # `obs` is a dict keyed by group name on manager-based envs.
    if isinstance(obs, dict):
        policy_obs = obs.get("policy")
        teacher_obs = obs.get("teacher")
    else:
        policy_obs = obs
        teacher_obs = None

    _emit("\n=== Reset observation shapes ===", fh)
    if policy_obs is not None:
        _emit(f"policy obs shape: {_fmt(policy_obs.shape)}", fh)
    if teacher_obs is not None:
        _emit(f"teacher obs shape: {_fmt(teacher_obs.shape)}", fh)

    # Pull the raw camera tensor (unflattened) for an unambiguous H/W readout.
    cam = env.unwrapped.scene.sensors.get("depth_camera")
    raw_depth_shape = None
    if cam is not None:
        depth = cam.data.output.get("distance_to_image_plane")
        if depth is not None:
            raw_depth_shape = tuple(int(s) for s in depth.shape)
            _emit(f"depth_camera distance_to_image_plane shape: {raw_depth_shape}", fh)

    # One zero-action step to exercise the depth pipeline end-to-end.
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step_obs, _r, _term, _trunc, _info = env.step(actions)
    step_policy = step_obs["policy"] if isinstance(step_obs, dict) else step_obs
    _emit(f"\npost-step policy shape: {_fmt(step_policy.shape)}", fh)

    _emit("\n=== Verdict ===", fh)
    policy_len = int(step_policy.shape[-1])
    ok = policy_len == EXPECTED_POLICY_LEN
    _emit(
        f"policy length: got {policy_len}, expected {EXPECTED_POLICY_LEN} "
        f"(= {EXPECTED_EGO} ego + {EXPECTED_T}*{EXPECTED_H}*{EXPECTED_W} depth "
        f"+ {EXPECTED_PROPRIO_T}*{EXPECTED_PROPRIO_DIM} proprio)  "
        f"-> {'PASS' if ok else 'FAIL'}",
        fh,
    )
    if raw_depth_shape is not None:
        # TiledCamera output for distance_to_image_plane is (N, H, W, C).
        # Find H and W as the two spatial dims (indices 1 and 2 when len==4).
        if len(raw_depth_shape) == 4 or len(raw_depth_shape) == 3:
            got_h, got_w = raw_depth_shape[1], raw_depth_shape[2]
        else:
            got_h, got_w = -1, -1
        h_ok = got_h == EXPECTED_H
        w_ok = got_w == EXPECTED_W
        _emit(
            f"depth_camera HxW: got ({got_h}, {got_w}), "
            f"expected ({EXPECTED_H}, {EXPECTED_W})  -> {'PASS' if (h_ok and w_ok) else 'FAIL'}",
            fh,
        )
        ok = ok and h_ok and w_ok

    _emit(f"\nOVERALL: {'PASS' if ok else 'FAIL'}", fh)
    fh.close()
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    # Open verdict file immediately so even early crashes surface.
    with open(VERDICT_PATH, "w") as _early:
        _early.write("script started\n")
    try:
        code = main()
    except BaseException as e:
        import traceback

        with open(VERDICT_PATH, "a") as _err:
            _err.write("\n=== EXCEPTION ===\n")
            _err.write(repr(e) + "\n")
            traceback.print_exc(file=_err)
        code = 2
    finally:
        with open(VERDICT_PATH, "a") as _fin:
            _fin.write(f"\nexit_code={code}\n")
        simulation_app.close()
    raise SystemExit(code)
