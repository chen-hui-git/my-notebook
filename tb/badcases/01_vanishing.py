"""坏案例 1：梯度消失 —— 20 层 Sigmoid + 故意很小的初始化。

现象：gradnorm 从最后一层往前指数衰减，fc01 接近 0；loss 卡在 0.693（= ln2，等于瞎猜）。
原因：Sigmoid 导数最大只有 0.25，20 层连乘 ≈ 0.25^20 ≈ 1e-12；再乘以小初始化更小。
修法：换 ReLU + He 初始化、加残差连接、加 BatchNorm、减少层数。

跑：.venv/bin/python badcases/01_vanishing.py
看：gradnorm/fc01 vs gradnorm/fc20（差十几个数量级），loss/train 卡住不动
"""
import pathlib
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import LOG_ROOT, log, make_data

torch.manual_seed(0)
x, y = make_data()

layers = []
for i in range(20):
    lin = nn.Linear(4 if i == 0 else 32, 32)
    nn.init.normal_(lin.weight, std=0.05)   # ← 罪魁祸首之一：初始化太小
    nn.init.zeros_(lin.bias)
    layers += [lin, nn.Sigmoid()]           # ← 另一个：Sigmoid 会压扁梯度
layers.append(nn.Linear(32, 2))
model = nn.Sequential(*layers)

opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_ROOT / "01_vanishing")

for step in range(201):
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    log(writer, model, loss.item(), step)
    opt.step()

writer.add_text("说明", "20 层 Sigmoid + std=0.05 初始化 → 梯度消失，前几层完全学不动。", 0)
writer.close()
print("done: logs/01_vanishing")
