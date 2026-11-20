"""坏案例公用工具：造数据、算梯度范数、统一记录方式。

所有坏案例都用同一套数据、同一个记录格式，方便横向对比：
    loss/train            损失
    gradnorm/total        整个模型梯度总范数
    gradnorm/fc01 等      第 1/5/10/20 层的梯度范数
    weights/fc01          第一层权重的分布（直方图）
"""
import pathlib

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent
LOG_ROOT = ROOT / "logs"


def make_data(n=1024, dim=4, xor=False):
    """4 维特征。默认标签 = 各维之和是否 > 0（线性可分，健康训练必收敛）；
    xor=True 时标签 = x0*x1 是否 > 0（异或型，非线性，难得多）。"""
    x = torch.randn(n, dim)
    y = ((x[:, 0] * x[:, 1] > 0) if xor else (x.sum(dim=1) > 0)).long()
    return x, y


def total_grad_norm(model):
    norms = [p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return torch.sqrt(sum(norms)).item()


def linear_layers(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def log(writer, model, loss, step):
    writer.add_scalar("loss/train", loss, step)
    writer.add_scalar("gradnorm/total", total_grad_norm(model), step)
    lins = linear_layers(model)
    for idx in (1, 5, 10, 20):
        if idx <= len(lins):
            w = lins[idx - 1].weight
            if w.grad is not None:
                writer.add_scalar(f"gradnorm/fc{idx:02d}", w.grad.norm().item(), step)
    if step % 20 == 0 and lins:
        writer.add_histogram("weights/fc01", lins[0].weight, step)
