"""
01_basics.py —— TensorBoard 最常用的 API，一次看全
===================================================

任务：给一堆 2D 点做二分类（两条交错的螺旋线，经典 two-spirals）。
线性模型分不开，用一个很小的 MLP 训练十几秒，但足够把 TensorBoard 的常用面板全部喂满。

跑完以后启动：
    tensorboard --logdir logs --port 6006
浏览器打开 http://localhost:6006

本文件用到的 API 和它们对应的 TensorBoard 面板：
    add_scalar         -> SCALARS      最重要，一条曲线
    add_scalar + 共享前缀 -> SCALARS   同一张图里画多条线（loss/train、loss/val）
    add_custom_scalars -> SCALARS      自定义排版，把 loss 和 acc 并排
    add_histogram      -> DISTRIBUTIONS / HISTOGRAMS  看权重、梯度的分布
    add_figure         -> IMAGES       传 matplotlib 图（决策边界动画）
    add_image          -> IMAGES       传图像张量（C,H,W）
    add_text           -> TEXT         Markdown 文本，写实验配置很方便
    add_graph          -> GRAPHS       网络结构
    add_pr_curve       -> PR CURVES    precision-recall 曲线
    add_hparams        -> HPARAMS      超参对比表格

顺便演示一个最关键的组织约定：
    **同一个 tag + 不同的 log_dir = 同一个面板里的不同 run**
下面会跑两个只有 "隐藏层宽度" 不同的实验，TensorBoard 里自动叠在一张图上对比。
"""
import math
import os
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
# matplotlib 默认把字体缓存写到 ~/.matplotlib，受限环境里不可写会报warning，指到项目内即可
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".mplcache"))

import matplotlib

matplotlib.use("Agg")  # 无窗口环境（脚本 / 服务器）必须用 Agg，否则会弹窗或直接报错
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# 有 Apple GPU 就用 MPS，没有就退到 CPU。这个 demo 小到 CPU 也是秒级。
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LOG_ROOT = os.path.join(_HERE, "logs")


# --------------------------------------------------------------------------- #
# 1. 造数据：两条交错的螺旋线
# --------------------------------------------------------------------------- #
def make_data(n_per_class=700, noise=0.08, flip_rate=0.03, seed=0):
    """两条互相缠绕的螺旋，半径从 0.25 到 1.0。

    noise 让点不要那么干净；flip_rate 是故意标错的样本比例 ——
    有了它准确率的天花板大约在 97%，曲线才不会假到一条直线。
    """
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(2):
        t = torch.linspace(0.25, 1.0, n_per_class).unsqueeze(1)  # 沿螺旋线的参数
        theta = t * 3.5 * math.pi + c * math.pi  # 第二条螺旋整体转半圈，于是和第一条交错
        pts = torch.cat([t * torch.cos(theta), t * torch.sin(theta)], dim=1)
        pts = pts + torch.randn(pts.shape, generator=g) * noise
        xs.append(pts)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    flip = torch.rand(len(x), generator=g) < flip_rate
    y[flip] = 1 - y[flip]
    return x, y


def make_model(hidden=128):
    return nn.Sequential(
        nn.Linear(2, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),  # 混用两种激活，纯属让 GRAPHS 面板好看一点，没别的意思
        nn.Linear(hidden, 2),
    ).to(DEVICE)


# --------------------------------------------------------------------------- #
# 2. 画决策边界：训练过程可视化最直观的东西，喂给 add_figure
# --------------------------------------------------------------------------- #
def decision_boundary_figure(model, x, y, step, hidden):
    model.eval()
    # 在 [-1.6, 1.6]^2 上铺网格，算出每个格点属于类别 1 的概率
    lin = torch.linspace(-1.6, 1.6, 120)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(grid), dim=1)[:, 1].cpu().reshape(120, 120)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
    ax.contourf(gx, gy, prob, levels=20, cmap="RdBu_r", alpha=0.85)
    ax.contour(gx, gy, prob, levels=[0.5], colors="k", linewidths=1.2)  # 决策边界
    xc, yc = x.cpu(), y.cpu()
    ax.scatter(xc[yc == 0, 0], xc[yc == 0, 1], s=4, c="navy", alpha=0.5)
    ax.scatter(xc[yc == 1, 0], xc[yc == 1, 1], s=4, c="crimson", alpha=0.5)
    ax.set_title(f"step {step}   hidden={hidden}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    model.train()
    return fig  # add_figure 会把它转成图像；调用方记得 plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. 训练一个 run
# --------------------------------------------------------------------------- #
def run_experiment(hidden: int, tag: str, lr: float = 1e-2, epochs: int = 160, batch_size: int = 128):
    torch.manual_seed(0)
    x, y = make_data()
    x, y = x.to(DEVICE), y.to(DEVICE)

    # 打乱后再切分：造数据时按类别顺序拼接的，不打乱会切出"一堆同类"
    perm = torch.randperm(len(x), device=DEVICE)
    x, y = x[perm], y[perm]
    n_train = int(len(x) * 0.8)
    xtr, ytr, xva, yva = x[:n_train], y[:n_train], x[n_train:], y[n_train:]

    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # === 每个 run 一个自己的目录，目录名就是 TensorBoard 曲线图例里显示的名字 ===
    run_dir = os.path.join(LOG_ROOT, "basics", tag)
    writer = SummaryWriter(log_dir=run_dir)

    # ---------- add_text：用 Markdown 写实验笔记 ----------
    writer.add_text(
        "config",
        f"### 实验配置 `{tag}`\n\n"
        f"- hidden = **{hidden}**\n"
        f"- lr = {lr:g}，optimizer = Adam，batch = {batch_size}，epochs = {epochs}\n"
        f"- device = `{DEVICE.type}`\n"
        f"- train / val = {len(xtr)} / {len(xva)}\n\n"
        f"> 这段文字来自 `add_text`，写实验记录非常方便。",
        global_step=0,
    )

    # ---------- add_custom_scalars：自己排 SCALARS 面板的版式 ----------
    # 注意到我们写的是 "loss/train"，这里就写 "loss/train"；Multiline 表示一张图多条线
    writer.add_custom_scalars(
        {
            "训练总览": {
                "Loss": ["Multiline", ["loss/train", "loss/val"]],
                "Accuracy": ["Multiline", ["acc/train", "acc/val"]],
            },
        }
    )

    # ---------- add_graph：网络结构图，写一次就够 ----------
    writer.add_graph(model, xtr[:1].clone())

    # ---------- add_image：也接受 HWC 的 uint8 numpy 数组，记得声明 dataformats ----------
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    xc, yc = xtr.cpu(), ytr.cpu()
    ax.scatter(xc[yc == 0, 0], xc[yc == 0, 1], s=5, c="navy")
    ax.scatter(xc[yc == 1, 0], xc[yc == 1, 1], s=5, c="crimson")
    ax.set_title("train data", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # (H,W,3) uint8
    plt.close(fig)
    writer.add_image("images/train_data", img, global_step=0, dataformats="HWC")

    step = 0
    n_batches = len(xtr) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(len(xtr), device=DEVICE)  # 每个 epoch 打乱一次
        xtr_s, ytr_s = xtr[perm], ytr[perm]

        for i in range(n_batches):
            xb = xtr_s[i * batch_size : (i + 1) * batch_size]
            yb = ytr_s[i * batch_size : (i + 1) * batch_size]

            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # ---------- add_histogram：看梯度分布，排查梯度消失/爆炸的第一步 ----------
            if step % 50 == 0 and model[0].weight.grad is not None:
                writer.add_histogram("grads/fc1", model[0].weight.grad, global_step=step)
                writer.add_histogram("weights/fc1", model[0].weight, global_step=step)

            opt.step()
            acc = (logits.argmax(1) == yb).float().mean().item()

            # ---------- add_scalar：最基础的用法 (tag, value, step) ----------
            writer.add_scalar("loss/train", loss.item(), global_step=step)
            writer.add_scalar("acc/train", acc, global_step=step)
            writer.add_scalar("lr", lr, global_step=step)

            # ---------- add_figure：每 100 步存一张决策边界，凑成 GIF 一样的效果 ----------
            if step % 100 == 0:
                fig = decision_boundary_figure(model, xtr, ytr, step, hidden)
                writer.add_figure("images/decision_boundary", fig, global_step=step)
                plt.close(fig)  # 不关会一直占内存，matplotlib 的 figure 不会被自动回收

            step += 1

        # ---------- epoch 级别评估（每轮结束评估一次验证集） ----------
        with torch.no_grad():
            model.eval()
            val_logits = model(xva)
            val_loss = loss_fn(val_logits, yva).item()
            val_acc = (val_logits.argmax(1) == yva).float().mean().item()
            model.train()

        # 想在同一个面板里画多条线，不需要 add_scalars：只要 tag 共享前缀
        # （loss/train 和 loss/val），TensorBoard 就会自动合并成一张图、分两个图例。
        # 老 API add_scalars 的副作用是给每个子 tag 建子目录，左侧 run 列表里会
        # 多出一堆"假 run"，所以这里直接写两条 add_scalar。
        writer.add_scalar("loss/val", val_loss, global_step=step)
        writer.add_scalar("acc/val", val_acc, global_step=step)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[{tag}] epoch {epoch:3d}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    # ---------- add_pr_curve：precision-recall，类别不均衡时比 accuracy 更有信息量 ----------
    with torch.no_grad():
        model.eval()
        prob = torch.softmax(model(xva), dim=1)[:, 1].cpu()
    writer.add_pr_curve("pr_curve", yva.cpu(), prob, global_step=step)

    # ---------- add_hparams：超参 + 最终指标记进表格，HPARAMS 面板里横向对比 ----------
    writer.add_hparams(
        {"hidden": hidden, "lr": lr, "epochs": epochs, "optimizer": "Adam"},
        {"hparam/val_acc": val_acc, "hparam/val_loss": val_loss},
        run_name="hparam_" + tag,
    )

    writer.flush()  # 确保数据落盘（进程被 kill 时不至于丢）
    writer.close()
    print(f"[{tag}] 完成，日志在 {run_dir}")


if __name__ == "__main__":
    print(f"device = {DEVICE}")
    os.makedirs(os.path.join(LOG_ROOT, "basics"), exist_ok=True)

    # 两个 run 只有隐藏层宽度不同：128 能拟合螺旋，8 只能画出几条折线 → 对比一眼可见
    run_experiment(hidden=128, tag="hidden_128")
    run_experiment(hidden=8, tag="hidden_8")

    print("\n接下来运行：tensorboard --logdir logs --port 6006")
