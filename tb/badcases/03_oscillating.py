"""坏案例 3：震荡不收敛 —— 学习率 2.0（正常是 0.05）。

现象：loss 在 0.15 ~ 0.7 之间来回跳，300 步过去了还在原地震荡，永远不降到位。
原因：lr=2 太大 + batch 只有 32，每步的梯度方向噪声大，更新总是越过最优点。
修法：lr ÷ 3~10、增大 batch、加 lr warmup/衰减、换 Adam。

（注意：这里用的是 XOR 型标签，比前几个案例难，所以更容易看出"不收敛"）

跑：.venv/bin/python badcases/03_oscillating.py
看：loss/train 是锯齿状而不是平滑下滑；gradnorm/total 上下乱跳
"""
import pathlib
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import LOG_ROOT, log, make_data

torch.manual_seed(0)
x, y = make_data(xor=True)

model = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2),
)
opt = torch.optim.SGD(model.parameters(), lr=2.0)    # ← 正常应该是 0.05
batch_size = 32                                      # ← 再配个小 batch，梯度噪声更大
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_ROOT / "03_oscillating")

for step in range(301):
    idx = torch.randint(0, len(x), (batch_size,))
    loss = loss_fn(model(x[idx]), y[idx])
    opt.zero_grad()
    loss.backward()
    log(writer, model, loss.item(), step)
    opt.step()

writer.add_text("说明", "lr=2.0 造成震荡：loss 锯齿、gradnorm 忽大忽小，收敛不了。", 0)
writer.close()
print("done: logs/03_oscillating")
