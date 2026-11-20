#!/usr/bin/env bash
# 跑完 5 个坏案例，日志写到 ./logs/
set -e
cd "$(dirname "$0")"

PY=../.venv/bin/python
for f in 00_healthy.py 01_vanishing.py 02_exploding.py 03_oscillating.py 04_frozen.py 05_dead_relu.py; do
  echo "=== $f ==="
  "$PY" "$f"
done

echo
echo "完成。启动 TensorBoard：./serve.sh   然后打开 http://localhost:6008"
