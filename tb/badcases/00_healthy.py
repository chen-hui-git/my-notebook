"""00_healthy：健康基线 —— 和那 5 个坏案例用同一套数据、同一个模型、同一套记录方式，
唯一区别就是 **lr = 0.05**（坏案例里分别是 10、2、只传一层参数、死 ReLU、以及 20 层 Sigmoid）。

放在这里的用途是对照：当你在真实训练里怀疑"这曲线是不是有问题"时，
先拿它对照一下"正常应该长什么样"。

跑：.venv/bin/python badcases/00_healthy.py
看：loss/train 平滑下滑；gradnorm/total 缓降不塌陷；gradnorm/fc01 与 fc20 同一数量级；
    DISTRIBUTIONS 里 weights/fc01 是一条缓慢移动的亮带（不是一条死竖线）。
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

model = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2),
)
opt = torch.optim.SGD(model.parameters(), lr=0.05)   # ← 唯一正确的选择：0.05
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_ROOT / "00_healthy")

for step in range(201):
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    log(writer, model, loss.item(), step)
    opt.step()

writer.add_text(
    "说明",
    "健康基线：lr=0.05。loss 平滑下降、gradnorm 缓降、层间同量级、权重分布缓慢移动。",
    0,
)
writer.close()
print("done: logs/00_healthy")
