#!/usr/bin/env bash
# Issue #16 — Phase 2 vision student with the D555 sim sensor tilted 30 deg
# down and a 10-frame depth history. The tilt lives in OffsetCfg.rot in
# rough_student_env_cfg.py (sensor-space only — URDF and physical robot are
# unchanged); 10 frames is passed through --num_frames so the same
# StudentActor sizes its conv input correctly.
set -euo pipefail

cd "$(dirname "$0")/.."

# Required by Isaac Sim on Linux to avoid OpenBLAS thread conflicts.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

TEACHER_CKPT="${TEACHER_CKPT:-logs/rsl_rl/ayg_rough/Teacher(baseline)/model_9999.pt}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
NUM_FRAMES="${NUM_FRAMES:-10}"
RUN_NAME="${RUN_NAME:-vision_tilt30_f10}"

python scripts/rsl_rl/train_student.py \
    --task=Isaac-Velocity-Rough-Ayg-Student-v0 \
    --teacher_ckpt="${TEACHER_CKPT}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --num_frames="${NUM_FRAMES}" \
    --headless \
    --enable_cameras \
    --run_name="${RUN_NAME}" \
    "$@"
