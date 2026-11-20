#!/usr/bin/env bash
# 一键跑完三个 demo，生成 logs/ 下的所有数据
set -e
cd "$(dirname "$0")"

echo "=== 01 基础 API ==="
.venv/bin/python 01_basics.py

echo
echo "=== 02 算子 / 硬件 profile ==="
.venv/bin/python 02_profiler.py

echo
echo "=== 03 训练中采硬件指标 ==="
.venv/bin/python 03_hw_monitor.py

echo
echo "全部完成。下一步："
echo "  ./serve.sh          # 启动 TensorBoard"
echo "  打开 http://localhost:6006"
