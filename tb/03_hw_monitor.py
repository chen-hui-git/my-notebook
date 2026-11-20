"""
03_hw_monitor.py —— 把 "硬件指标" 和 "训练指标" 画在同一块面板上
================================================================

02 的 torch.profiler 回答的是 "哪个算子慢"（kernel 级、抽样、开销大）。
但日常训练里你更常问的是另一类问题：
    - 内存是不是一路在涨（泄漏）？
    - GPU / 统一内存用了多少，离上限还有多远？
    - CPU 是不是被数据加载占满了？
    - 吞吐（samples/s）稳不稳定？

答案很简单：自己采指标，用 add_scalar 写进去。TensorBoard 不关心数字从哪来，
只要有 tag + step 就能画图 —— 这是把硬件监控接进训练看板最通用的办法。

macOS 上 MPS 能查的显存指标（torch.mps.*）：
    current_allocated_memory()  —— 张量当前真正占用（MB），删掉张量就掉回去
    driver_allocated_memory()   —— 驱动层面的池子（MB），回收了也不一定马上还给系统
    recommended_max_memory()    —— 系统建议的上限

NVIDIA 机器上换成 torch.cuda.memory_allocated() / memory_reserved() /
max_memory_allocated()，写法一模一样。
"""
import os
import time

import psutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs", "hw_monitor", "mps_run")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MB = 1024 * 1024


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))


def hw_snapshot(proc):
    """采一次硬件快照，返回 {tag: value}。key 直接就是 TensorBoard 里的 tag。"""
    snap = {
        "hardware/cpu_pct": proc.cpu_percent(),        # 本进程 CPU（% of 1 core，可能 >100）
        "hardware/sys_cpu_pct": psutil.cpu_percent(),  # 整机 CPU
        "hardware/rss_mb": proc.memory_info().rss / MB,  # 进程常驻内存
        "hardware/sys_mem_pct": psutil.virtual_memory().percent,
    }
    if DEVICE.type == "mps":
        snap["hardware/mps_alloc_mb"] = torch.mps.current_allocated_memory() / MB
        snap["hardware/mps_driver_mb"] = torch.mps.driver_allocated_memory() / MB
        snap["hardware/mps_limit_mb"] = torch.mps.recommended_max_memory() / MB
    elif DEVICE.type == "cuda":
        snap["hardware/cuda_alloc_mb"] = torch.cuda.memory_allocated() / MB
        snap["hardware/cuda_reserved_mb"] = torch.cuda.memory_reserved() / MB
        snap["hardware/cuda_peak_mb"] = torch.cuda.max_memory_allocated() / MB
    return snap


def make_batch(bs):
    """造 "横条纹 / 竖条纹" 两类图，标签由条纹方向决定 —— 需要卷积才学得会。

    之所以不直接用随机标签：随机标签的 loss 只能停在 ln(2)，曲线是平的，
    看不出 "loss 往下走的时候硬件在干嘛" 这个联动关系。
    """
    yy = torch.arange(64, device=DEVICE).view(1, 1, 64, 1).float()
    xx = torch.arange(64, device=DEVICE).view(1, 1, 1, 64).float()
    y = torch.randint(0, 2, (bs,), device=DEVICE)
    horizontal = torch.sin(yy * 0.4)  # (1,1,64,1)
    vertical = torch.sin(xx * 0.4)    # (1,1,1,64)
    pattern = torch.where(y.view(bs, 1, 1, 1) == 1, horizontal, vertical)
    x = torch.randn(bs, 3, 64, 64, device=DEVICE) * 1.2 + pattern
    return x, y


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.manual_seed(0)

    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # 第一次调用只是初始化基线，返回 0，丢掉
    psutil.cpu_percent()

    model = Net().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=LOG_DIR)
    writer.add_text("note", f"device = `{DEVICE.type}`，每步采样一次硬件指标", global_step=0)

    # 把 loss 和硬件指标排进同一张卡片，"loss 掉的时候内存在干什么" 一眼可见
    writer.add_custom_scalars(
        {
            "训练 vs 硬件": {
                "Loss": ["Multiline", ["loss/train"]],
                "Accuracy": ["Multiline", ["acc/train"]],
                "统一内存 (MB)": ["Multiline", ["hardware/mps_alloc_mb", "hardware/mps_driver_mb"]],
                "CPU %": ["Multiline", ["hardware/cpu_pct", "hardware/sys_cpu_pct"]],
                "吞吐 (samples/s)": ["Multiline", ["perf/throughput"]],
            }
        }
    )

    bs, steps = 64, 120
    print(f"device={DEVICE.type}，跑 {steps} 步，每步采样一次硬件指标 ...")

    for step in range(steps):
        xb, yb = make_batch(bs)

        t0 = time.perf_counter()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        # 在 backward 之前采样：此时 forward 的激活值全都在，显存处于高位。
        # （backward 之后 PyTorch 会给梯度算完就释放中间张量，那时只能看到 ~4MB 的空载数字）
        snap = hw_snapshot(proc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if DEVICE.type == "mps":
            torch.mps.synchronize()  # 异步设备必须同步，否则测到的是"下发命令"的时间
        dt = time.perf_counter() - t0

        # ---- 训练指标 ----
        writer.add_scalar("loss/train", loss.item(), global_step=step)
        writer.add_scalar("acc/train", (logits.argmax(1) == yb).float().mean().item(), global_step=step)
        writer.add_scalar("perf/step_ms", dt * 1000, global_step=step)
        writer.add_scalar("perf/throughput", bs / dt, global_step=step)

        # ---- 硬件指标：和上面完全一样，只是数字来源不同 ----
        for tag, val in snap.items():
            writer.add_scalar(tag, val, global_step=step)

        if step % 20 == 0:
            print(f"  step {step:3d}  loss={loss.item():.4f}  "
                  f"mps_alloc={snap.get('hardware/mps_alloc_mb', 0):.0f}MB  step={dt*1000:.1f}ms")

    # 这条 alloc 曲线平着走 = 没有内存泄漏；如果你的实现里它一路往上，就是有东西没释放。
    # ---- 收尾：看看 empty_cache 到底释放了什么 ----
    # driver 是"缓存池"，训练中只涨不缩是正常的（池子留着下次复用，避免频繁向系统申请）。
    # 想让内存真正还给系统，就要显式调用 empty_cache()。
    if DEVICE.type == "mps":
        before = torch.mps.driver_allocated_memory() / MB
        torch.mps.empty_cache()
        after = torch.mps.driver_allocated_memory() / MB
        writer.add_scalar("summary/mps_driver_mb", before, global_step=steps)
        writer.add_scalar("summary/mps_driver_mb_after_empty_cache", after, global_step=steps)
        print(f"\nempty_cache: driver {before:.0f}MB -> {after:.0f}MB")
    writer.add_scalar("summary/rss_mb", proc.memory_info().rss / MB, global_step=steps)
    writer.flush()
    writer.close()

    print(f"\n完成，日志在 {LOG_DIR}")
    print("提示：想要比 '每一步' 更密的采样（比如 10ms 一次），开个后台线程定时采，")
    print("      用 add_scalar(tag, value, walltime=time.time()) 写到墙钟时间轴上即可。")


if __name__ == "__main__":
    main()
