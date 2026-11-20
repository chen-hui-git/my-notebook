#!/usr/bin/env bash
# 启动坏案例专属的 TensorBoard（默认 6008，避免和主 demo 的 6006 撞车）
set -e
cd "$(dirname "$0")"

PORT="${1:-6008}"
echo "Bad cases TensorBoard: http://localhost:${PORT}"
exec ../.venv/bin/tensorboard --logdir logs --port "$PORT" --host 127.0.0.1 --reload_interval 5
