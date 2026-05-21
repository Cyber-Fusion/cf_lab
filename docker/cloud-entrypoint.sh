#!/bin/bash
set -e

# Vast.ai / RunPod inject the user's public key via $PUBLIC_KEY env var.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Start sshd in background and hand control to the CMD.
/usr/sbin/sshd
exec "$@"
