"""坏案例 2：梯度爆炸 —— 学习率 10（正常是 0.05）。

现象：gradnorm 出现数量级尖峰（0.4 → 385），loss 从 0.73 冲到 1593，之后卡死在 2.1~2.8
      （比初始的 0.73 还差），权重被推到 223（原来 4.5）。注意：这个例子不会 NaN，
      更严重的情况（深层网络 + 混合精度）才会 NaN。
原因：一步更新迈太大，参数被推到饱和区，梯度反而更大，正反馈直到溢出。
修法：降 lr、梯度裁剪 clip_grad_norm_、加归一化、检查数据是否未标准化。

跑：.venv/bin/python badcases/02_exploding.py
看：gradnorm/total 的尖峰 + loss/train 冲高到 1593 后卡在 2.1~2.8 下不来
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
opt = torch.optim.SGD(model.parameters(), lr=10.0)   # ← 罪魁祸首
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_ROOT / "02_exploding")

for step in range(300):
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()

    if not torch.isfinite(loss):
        writer.add_text("崩溃", f"step {step}：loss = {loss.item()}，训练已经废了。", step)
        print(f"step {step}: loss = {loss.item()}  → 崩了")
        break

    log(writer, model, loss.item(), step)
    opt.step()

writer.add_text(
    "说明",
    "lr=10 造成梯度爆炸：gradnorm 尖峰到 385，loss 冲到 1593 后卡死在 2.1~2.8，"
    "权重被推到 223（原来 4.5），训练再也回不来。",
    0,
)
writer.close()
print("done: logs/02_exploding")
