"""第 4 步：把训练过程拍成照片 —— IMAGES 面板。

还是线性回归，但每 20 步画一张"当前模型拟合得怎么样"的图存进去。
在 IMAGES 面板拖动下方的 step 滑块，就能看到红线（模型）一步步贴到散点（数据）上。

跑：  .venv/bin/python learn/step4.py
看：  左侧 IMAGES 面板 → 选 learn/step4 → tag "fit" → 拖滑块
"""
import os
import pathlib

LOG = pathlib.Path(__file__).resolve().parent.parent / "logs" / "learn" / "step4"
# 本机 ~/.matplotlib 不可写，把字体缓存指到项目里（放在 import matplotlib 之前）
os.environ.setdefault("MPLCONFIGDIR", str(LOG.parent.parent.parent / ".mplcache"))

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接渲染成图片
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
x = torch.randn(200, 1) * 3
y = 2 * x + 1 + torch.randn_like(x) * 0.3

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.05)
writer = SummaryWriter(LOG)

for step in range(201):
    loss = ((x * w + b - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 20 == 0:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=90)
        ax.scatter(x.numpy(), y.numpy(), s=6, alpha=0.5, label="data")
        ax.plot(x.numpy(), (x * w + b).detach().numpy(), color="crimson", label="model")
        ax.set_title(f"step {step}  loss={loss.item():.2f}", fontsize=9)
        ax.legend(fontsize=7)
        writer.add_figure("fit", fig, step)   # 传 matplotlib figure
        plt.close(fig)                        # 不关会一直占内存

writer.close()
print(f"写好了：{LOG}（11 张图）")
