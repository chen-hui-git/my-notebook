"""第 1 步的补充：同一个 sin，横轴疏密不同 → 图长得完全不一样。

三个 run 都写 tag "y"、都写 100 步，区别只在 sin 里面的系数。
所以在 TensorBoard 里会画进同一张卡，一眼看出"横轴"和"函数自变量"是两回事。

跑：  .venv/bin/python learn/step1b.py
看：  只勾 learn/step1b_slow / step1b_mid / step1b_fast 三个 run
      mid  = 你现在看到的（sin(i/10)，1.6 个周期）
      slow = sin(i/100)，0.16 个周期，只爬了一小段
      fast = sin(i)，15.9 个周期，密得像梳子
"""
import math
import pathlib

from torch.utils.tensorboard import SummaryWriter

BASE = pathlib.Path(__file__).resolve().parent.parent / "logs" / "learn"

for name, divisor in [("slow", 100), ("mid", 10), ("fast", 1)]:
    writer = SummaryWriter(BASE / f"step1b_{name}")
    for i in range(100):
        writer.add_scalar("y", math.sin(i / divisor), i)   # 横轴永远是 i（第几步）
    writer.close()

print("写好了：logs/learn/step1b_slow、step1b_mid、step1b_fast")
