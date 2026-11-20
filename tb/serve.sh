#!/usr/bin/env bash
# 启动 TensorBoard。默认 6006 端口，可以传参换端口：./serve.sh 6007
set -e
cd "$(dirname "$0")"

PORT="${1:-6006}"
echo "TensorBoard: http://localhost:${PORT}   （logdir=./logs，Ctrl-C 退出）"
exec .venv/bin/tensorboard --logdir logs --port "$PORT" --reload_interval 5
