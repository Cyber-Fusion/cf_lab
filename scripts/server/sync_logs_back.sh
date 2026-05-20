#!/usr/bin/env bash
# Pull the student training logs back from the rented GPU server to local
# cf_lab/logs/rsl_rl/ayg_rough/ so we can inspect tensorboard + checkpoints.
#
# Usage: scripts/server/sync_logs_back.sh -p <port> [-H host] [-u user]
set -euo pipefail

USER=root
HOST=213.181.123.15
PORT=""
REMOTE=/workspace/cf_lab

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port) PORT="$2"; shift 2 ;;
        -H|--host) HOST="$2"; shift 2 ;;
        -u|--user) USER="$2"; shift 2 ;;
        -r|--remote) REMOTE="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "$PORT" ]]; then
    echo "Missing --port." >&2; exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[sync_logs_back] $USER@$HOST:$REMOTE/logs/  ->  $REPO_ROOT/logs/  (port $PORT)"
mkdir -p "$REPO_ROOT/logs/rsl_rl/ayg_rough"
rsync -avh \
    -e "ssh -p $PORT" \
    --exclude 'Teacher(baseline)/' \
    "$USER@$HOST:$REMOTE/logs/rsl_rl/ayg_rough/" \
    "$REPO_ROOT/logs/rsl_rl/ayg_rough/"

echo "[sync_logs_back] done."
