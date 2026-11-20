"""第 2 步：两个 run 叠在一张图上 —— 理解"目录不同 = 曲线不同"。

跑：  .venv/bin/python learn/step2.py
看：  SCALARS 面板 → 只勾选 learn/step2_slow 和 learn/step2_fast
      tag 同样是 "y"，但两个目录 → 同一张图里两条线，图例是目录名。
"""
import math
import pathlib

from torch.utils.tensorboard import SummaryWriter

BASE = pathlib.Path(__file__).resolve().parent.parent / "logs" / "learn"

for name, freq in [("slow", 0.10), ("fast", 0.25)]:
    writer = SummaryWriter(BASE / f"step2_{name}")      # 目录不同 → 这是两个 run
    for i in range(60):
        writer.add_scalar("y", math.sin(i * freq), i)   # tag 相同 → 画进同一张图
    writer.close()

print("写好了：logs/learn/step2_slow 和 logs/learn/step2_fast")
