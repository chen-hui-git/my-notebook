"""坏案例 4：权重冻结 —— 参数没交给优化器（经典 bug）。

现象：gradnorm/fc01 明明有值、而且越涨越大（0.06 → 19），但 weights/fc01 的直方图
      11 个时刻完全重叠，一点没动；loss 只靠最后一层在降，卡在 0.37 下不去。
原因：`opt = SGD(model[-1].parameters())` 只把最后一层传进去了。
      常见变体：漏传某个子模块、requires_grad=False、自己 new 了参数却没换回去。
修法：`opt = SGD(model.parameters())`；打印 len(list(opt.param_groups[0]['params'])) 核对数量。

跑：.venv/bin/python badcases/04_frozen.py
看：weights/fc01 直方图不动 + gradnorm/fc01 却不为 0 → "有梯度 ≠ 在学"
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

# ← 罪魁祸首：只把最后一层交给优化器，前两层永远不更新
opt = torch.optim.SGD(model[-1].parameters(), lr=0.05)

print(f"优化器管着 {sum(len(g['params']) for g in opt.param_groups)} 个参数张量，"
      f"模型一共 {len(list(model.parameters()))} 个")

loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_ROOT / "04_frozen")

for step in range(201):
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    log(writer, model, loss.item(), step)
    opt.step()

writer.add_text("说明", "只有最后一层进了 optimizer，前两层权重冻结；梯度不为 0 但参数不动。", 0)
writer.close()
print("done: logs/04_frozen")
