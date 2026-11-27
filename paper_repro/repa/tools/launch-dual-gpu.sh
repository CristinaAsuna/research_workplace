#!/usr/bin/env bash
set -euo pipefail

train_python=$1
shift

# Accelerate 1.2.1 forwards port 0 to the static rendezvous unchanged.
master_port=$("$train_python" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)

printf 'Single-node rendezvous port: %s\n' "$master_port"
exec "$train_python" -m accelerate.commands.launch \
    --multi_gpu --num_processes 2 --num_machines 1 \
    --rdzv_backend static --main_process_port "$master_port" --dynamo_backend no "$@"
