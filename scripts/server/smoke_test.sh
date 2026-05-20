#!/usr/bin/env bash
# Cheap server smoke test: boots Isaac Sim, loads the student env with the D555
# depth sensor, steps a few frames, asserts the obs contract. ~30s end-to-end.
#
# Run *first* on a fresh server instance to confirm the RTX renderer plugins
# load. If this fails with a segfault in librtx.scenedb.plugin.so, the host
# OS is Ubuntu 24.04 (glibc 2.39) and Isaac Sim 5.1 can't load its RTX libs.
# Fix: relaunch on an Ubuntu 22.04 vast.ai template (see docs/SERVER_SETUP.md).
#
# Usage (on the server):
#   cd /workspace/cf_lab
#   source .venv/bin/activate
#   scripts/server/smoke_test.sh
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.venv/bin/python -u scripts/sanity/check_d555_depth.py \
    --enable_cameras --headless \
    --out /tmp/d555_smoke.png

echo "[smoke_test] PNG: /tmp/d555_smoke.png (copy back with scp if you want to inspect visually)"
