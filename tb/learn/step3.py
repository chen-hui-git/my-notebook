"""第 3 步：换成真训练 —— loss 往下走，参数往目标值爬。

用最简单的线性回归 y = 2x + 1：模型只有两个参数 w、b，
训练 200 步就能看到 w 爬向 2、b 爬向 1 —— 这就是"把训练过程变成曲线"。

跑：  .venv/bin/python learn/step3.py
看：  SCALARS 面板 → 只勾选 learn/step3
      loss/train 往下掉，param/w 爬到 2，param/b 爬到 1
"""
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

LOG = pathlib.Path(__file__).resolve().parent.parent / "logs" / "learn" / "step3"

torch.manual_seed(0)
x = torch.randn(256, 1) * 3
y = 2 * x + 1 + torch.randn_like(x) * 0.1      # 真值：w=2, b=1，加点噪声

w = torch.randn(1, requires_grad=True)         # 待学习的两个参数
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.05)
writer = SummaryWriter(LOG)

for step in range(200):
    loss = ((x * w + b - y) ** 2).mean()       # 均方误差
    opt.zero_grad()
    loss.backward()
    opt.step()

    # 三个数字，三条曲线。任何你想盯住的标量都可以这样写进来
    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("param/w", w.item(), step)
    writer.add_scalar("param/b", b.item(), step)

writer.close()
print(f"w={w.item():.3f}（目标 2.0）  b={b.item():.3f}（目标 1.0）")
