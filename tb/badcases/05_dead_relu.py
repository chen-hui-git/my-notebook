"""坏案例 5：神经元全死（Dying ReLU）—— 偏置初始化到很负。

现象：从第一步开始 gradnorm 就接近 0（1e-8 量级），loss 卡在 0.693（= ln2，瞎猜），
      权重直方图一条竖线，永远不动。
原因：第一层 bias 被设成 -100，所有 ReLU 的输入都是负的 → 输出全 0 → 导数全 0 →
      梯度传不回去，神经元永久死亡。
     真实场景里常见原因：lr 太大把 bias 推到很负、输入没归一化、weight decay 过强。
修法：换 LeakyReLU/GELU、降低 lr、加 BatchNorm、检查输入是否标准化。

跑：.venv/bin/python badcases/05_dead_relu.py
看：gradnorm/total 从 step 0 就贴着 0；loss 一条水平线 0.693
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
with torch.no_grad():
    model[0].bias.fill_(-100.0)   # ← 罪魁祸首：把所有 ReLU 顶到死区

opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_ROOT / "05_dead_relu")

for step in range(201):
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    log(writer, model, loss.item(), step)
    opt.step()

writer.add_text("说明", "第一层 bias=-100，ReLU 全死，梯度恒为 0，模型永远停在瞎猜。", 0)
writer.close()
print("done: logs/05_dead_relu")
