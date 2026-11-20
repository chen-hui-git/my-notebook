"""第 1 步：最小可用的 TensorBoard —— 一条曲线，9 行。

跑：  .venv/bin/python learn/step1.py
看：  http://127.0.0.1:6006  →  左侧 SCALARS / Time Series  →  只勾选 learn/step1
"""
import math
import pathlib

from torch.utils.tensorboard import SummaryWriter

LOG = pathlib.Path(__file__).resolve().parent.parent / "logs" / "learn" / "step1"

writer = SummaryWriter(LOG)          # 目录名 = run 名，TensorBoard 左侧就显示 learn/step1
for i in range(100):
    writer.add_scalar("y", math.sin(i / 10), i)   # tag="y"，value，step=i
writer.close()                       # 必须关闭，否则最后几条数据可能没落盘

print(f"写好了：{LOG}")
