"""第 5 步：直方图 —— 一次看一整个分布，而不是一个数字。

前面 4 步的 add_scalar 每次只记录「一个数」。
但一层权重有几百上千个数，你没法给每个数画一条曲线，
所以这里把「整层的所有数」打包成一个分布，每 20 步记录一次。

跑：  .venv/bin/python learn/step5.py
看：  左侧 HISTOGRAMS 或 DISTRIBUTIONS 面板 → 只勾 learn/step5
     两个 tag：weights/fc1（第一层权重）、grads/fc1（它的梯度）
"""
import pathlib

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

LOG = pathlib.Path(__file__).resolve().parent.parent / "logs" / "learn" / "step5"

torch.manual_seed(0)

# 造点简单数据：4 个特征，标签看总和正负
x = torch.randn(1024, 4)
y = (x.sum(dim=1) > 0).long()

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG)

for step in range(201):
    logits = model(x)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()

    # 关键：把「整层权重矩阵」和「它的梯度」整体交给 add_histogram
    # 传进去的是张量，TB 会自己按分桶统计每个区间的数量
    if step % 20 == 0:
        writer.add_histogram("weights/fc1", model[0].weight, step)
        writer.add_histogram("grads/fc1", model[0].weight.grad, step)
        writer.add_scalar("loss/train", loss.item(), step)

        # 直方图看"形状"，下面三条曲线看"趋势"（范数 = 所有元素平方和开根号）
        w1, g1 = model[0].weight, model[0].weight.grad
        writer.add_scalar("norm/w_fc1", w1.norm().item(), step)          # 权重范数
        writer.add_scalar("norm/g_fc1", g1.norm().item(), step)          # 梯度范数
        writer.add_scalar("norm/ratio_fc1", (g1.norm() / w1.norm()).item(), step)  # 梯度/权重比

    opt.step()

writer.close()
print(f"写好了：{LOG}（11 个时间点的直方图）")
