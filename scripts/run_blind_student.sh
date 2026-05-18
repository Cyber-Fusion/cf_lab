#!/usr/bin/env bash
# Issue #16 — Phase 2 blind student baseline.
# Run from the cf_lab repo root with the venv already activated (or rely on the venv
# in .venv/). Designed for the rented vast.ai 4090 server; safe to run locally too
# at a smaller --num_envs.
set -euo pipefail

cd "$(dirname "$0")/.."

# Required by Isaac Sim on Linux to avoid OpenBLAS thread conflicts.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

TEACHER_CKPT="${TEACHER_CKPT:-logs/rsl_rl/ayg_rough/Teacher(baseline)/model_9999.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
RUN_NAME="${RUN_NAME:-blind_v1}"

python scripts/rsl_rl/train_student.py \
    --task=Isaac-Velocity-Rough-Ayg-Student-Blind-v0 \
    --teacher_ckpt="${TEACHER_CKPT}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --blind \
    --headless \
    --run_name="${RUN_NAME}" \
    "$@"
