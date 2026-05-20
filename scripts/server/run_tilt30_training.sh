#!/usr/bin/env bash
# Run the vision-student distillation training on the rented GPU server.
# Intended to be executed *inside* the server's cf_lab venv (after `deploy.sh`
# has rsynced the tree and the venv is provisioned per docs/SERVER_SETUP.md).
#
# Usage (on the server):
#   cd /workspace/cf_lab
#   source .venv/bin/activate
#   scripts/server/run_tilt30_training.sh [--num_envs N] [--max_iterations K] [--seed S] [--run_name NAME]
#
# Defaults are sized for a single RTX 4090 (24 GB) with the 10-frame depth
# stack. Training streams to logs/rsl_rl/ayg_rough/<timestamp>_<run_name>/.
set -euo pipefail

NUM_ENVS=2048
MAX_ITER=2000
SEED=42
RUN_NAME=tilt30_f10
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs)        NUM_ENVS="$2"; shift 2 ;;
        --max_iterations)  MAX_ITER="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --run_name)        RUN_NAME="$2"; shift 2 ;;
        --log_file)        LOG_FILE="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Same threading shim required locally — without these OpenBLAS/MKL crashes
# with an illegal instruction on certain CPUs (CF_LAB CLAUDE.md).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CMD=(.venv/bin/python -u scripts/rsl_rl/train.py
    --task=Isaac-Velocity-Rough-Ayg-Student-v0
    --agent=rsl_rl_distillation_cfg_entry_point
    --headless
    --enable_cameras
    --num_envs="$NUM_ENVS"
    --max_iterations="$MAX_ITER"
    --seed="$SEED"
    --run_name="$RUN_NAME"
    --load_run='Teacher\(baseline\)'
    --checkpoint=model_9999.pt
)

echo "[run_tilt30_training] ${CMD[*]}"
if [[ -n "$LOG_FILE" ]]; then
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    "${CMD[@]}"
fi
