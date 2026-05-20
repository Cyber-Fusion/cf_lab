#!/usr/bin/env bash
# Rsync the local cf_lab tree to /workspace/cf_lab on the rented GPU server.
#
# Two-step sync:
#   1) bulk: everything except .venv, __pycache__, .git internals, and logs/
#   2) teacher: the locked Phase 1 checkpoint logs/rsl_rl/ayg_rough/Teacher(baseline)/
#      (everything else under logs/ stays local; we pull training output back
#       afterwards with sync_logs_back.sh)
#
# Usage:
#   scripts/server/deploy.sh -p <port> [-H host] [-u user] [--dry-run]
set -euo pipefail

USER=root
HOST=213.181.123.15
PORT=""
REMOTE=/workspace/cf_lab
DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port) PORT="$2"; shift 2 ;;
        -H|--host) HOST="$2"; shift 2 ;;
        -u|--user) USER="$2"; shift 2 ;;
        -r|--remote) REMOTE="$2"; shift 2 ;;
        -n|--dry-run) DRY="--dry-run -v"; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "$PORT" ]]; then
    echo "Missing --port. vast.ai SSH ports are ephemeral; check the instance dashboard." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

echo "[deploy:bulk] $REPO_ROOT/  ->  $USER@$HOST:$REMOTE  (port $PORT)"
rsync -avh --delete $DRY \
    -e "$SSH" \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/objects/' \
    --exclude '.git/lfs/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.vscode/' \
    --exclude 'outputs/' \
    --exclude '.DS_Store' \
    --exclude 'logs/' \
    "$REPO_ROOT/" "$USER@$HOST:$REMOTE/"

TEACHER_DIR="logs/rsl_rl/ayg_rough/Teacher(baseline)"
if [[ -d "$REPO_ROOT/$TEACHER_DIR" ]]; then
    echo "[deploy:teacher] $TEACHER_DIR -> remote"
    $SSH "$USER@$HOST" "mkdir -p '$REMOTE/logs/rsl_rl/ayg_rough'"
    rsync -avh $DRY \
        -e "$SSH" \
        "$REPO_ROOT/$TEACHER_DIR/" \
        "$USER@$HOST:$REMOTE/$TEACHER_DIR/"
else
    echo "[deploy:teacher] WARN: $TEACHER_DIR missing locally — training will fail without it."
fi

echo "[deploy] done."
