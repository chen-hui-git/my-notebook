"""
02_profiler.py —— 用 torch.profiler 做算子 / 硬件层面的 profile
==============================================================

这是 "硬件 profiling" 的主力工具。它不像 loss 曲线那样看"训练得好不好"，
而是回答"时间花在哪了"：每个算子耗时多少、哪里在等 GPU、显存峰值出现在哪一步。

TensorBoard 的 PROFILE 面板提供四种视图（需要额外装 TensorFlow 才会出现，见下面的【本机说明】）：
    Overview   —— 每步耗时拆解（计算 / 数据加载 / 空转），先看这个找瓶颈
    Operator   —— 按算子聚合的耗时表，CPU 时间和设备时间分列
    Kernel     —— 更细一层（CUDA kernel 名），只在有 CUDA 的机器上出现
    Trace      —— 时间线视图，能放大到微秒，看算子排布和 GPU 空泡
    Memory     —— 显存曲线，找峰值和泄漏

关键参数：
    schedule        —— 别从第一步就开始采（还在预热 / 编译），用 schedule 跳过前面几步
    record_shapes   —— 记录张量形状，方便分清是哪个 matmul
    profile_memory  —— 记录内存分配事件（数据量会变大，按需开）
    with_stack      —— 记录 Python 调用栈，能直接定位到源码行（开销稍大）
    on_trace_ready  —— 采集完自动把 trace 写成 TensorBoard 能读的格式

本机说明（macOS + Python 3.14）：
    TensorBoard 的 PROFILE 面板由 tensorboard-plugin-profile 提供，而这个插件在加载时
    会探测 `import tensorflow`；TensorFlow 目前没有 Python 3.14 的 macOS 轮子，
    所以面板不会出现。trace 文件本身是通用的 JSON，直接拖进 https://ui.perfetto.dev
    （或 chrome://tracing）就能看时间线，效果一样，还不用装几百 MB 的 TensorFlow。
    如果你的环境能装 TensorFlow（Python <= 3.13），PROFILE 面板会自动出现。

注意：profile 区域里不要写 .item() / print() / torch.mps.synchronize()。
这些操作会强制同步，把 aten::_local_scalar_dense 顶到耗时榜第一名，
你就看不到真实的计算分布了 —— 这也是新手最容易踩的坑。
"""
import os
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs", "profiler", "cnn_profile")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SmallCNN(nn.Module):
    """卷积 / BN / 池化 / 全连接都有，方便在 trace 里看到不同类型的 kernel。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def pick_activities():
    """能采什么设备就打什么设备。CUDA 机器上是 CPU + CUDA。

    macOS 的 MPS 目前没有对应的 ProfilerActivity，trace 里只有 CPU 侧的算子时间。
    想在 Mac 上看 GPU kernel 详情，得用 torch.mps.profiler（os_signpost）
    配合 Xcode Instruments 来录，这属于另一套工具，README 里提了。
    """
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    return acts


def main():
    torch.manual_seed(0)
    os.makedirs(LOG_DIR, exist_ok=True)

    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    scalars = SummaryWriter(log_dir=LOG_DIR)

    # 固定形状的假数据：profile 不需要真数据，还能省显存
    def make_batch(bs=32):
        x = torch.randn(bs, 3, 64, 64, device=DEVICE)
        y = torch.randint(0, 10, (bs,), device=DEVICE)
        return x, y

    def one_step(xb, yb):
        # record_function 给自己关心的代码段打标签，标签会出现在 trace 里，非常好用
        with torch.profiler.record_function("forward"):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        with torch.profiler.record_function("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with torch.profiler.record_function("optimizer_step"):
            opt.step()
        return loss.detach()  # 只返回张量，不当场 .item()，避免同步

    # ---------------- 1. 预热 ----------------
    # 第一步里有各种初始化（算子选择、内存池预分配），不预热的话它会把整张表带偏
    for _ in range(5):
        one_step(*make_batch())
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    # ---------------- 2. 正式采集 ----------------
    activities = pick_activities()
    print(f"device={DEVICE.type}  activities={[a.name for a in activities]}")
    print("（macOS 上 MPS 没有 kernel 级采样，只能看到 CPU 侧时间，这是正常的）")

    # schedule：wait=2 跳过 2 步，warmup=2 再暖机 2 步（trace 里会标记但不算进统计），
    # active=8 采 8 步，repeat=1 只来一轮 → 一共 2+2+8=12 步
    schedule = torch.profiler.schedule(wait=2, warmup=2, active=8, repeat=1)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(LOG_DIR),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(20):
            one_step(*make_batch())
            prof.step()  # 告诉 profiler 这一步结束了，schedule 靠它推进

    # ---------------- 3. 命令行里也打一份聚合表，不开网页也能看 ----------------
    print("\n=== 按 CPU 总耗时排序 Top 15 ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    ka = prof.key_averages()
    device_col = "self_device_time_total"
    if hasattr(ka, device_col) and ka.total_average().self_device_time_total > 0:
        print(f"\n=== 按设备 ({DEVICE.type}) 耗时排序 Top 15 ===")
        print(ka.table(sort_by=device_col, row_limit=15))

    # ---------------- 4. 测吞吐：这种"要同步"的事放在 profile 外面做 ----------------
    times = []
    for i in range(10):
        xb, yb = make_batch()
        t0 = time.perf_counter()
        one_step(xb, yb)
        if DEVICE.type == "mps":
            torch.mps.synchronize()  # 异步设备必须同步，否则测到的是"下发命令"的时间
        dt_ms = (time.perf_counter() - t0) * 1000
        times.append(dt_ms)
        scalars.add_scalar("perf/step_ms", dt_ms, global_step=i)
        scalars.add_scalar("perf/throughput_samples_per_s", 32 / (dt_ms / 1000), global_step=i)

    scalars.close()
    print(f"\n中位 step 耗时 {statistics.median(times):.2f} ms  "
          f"（batch=32，{32 / (statistics.median(times) / 1000):.0f} samples/s）")
    print(f"\ntrace 已写入：")
    for fn in sorted(os.listdir(LOG_DIR)):
        if fn.endswith(".json"):
            print("  " + os.path.join(LOG_DIR, fn))
    print("看时间线：打开 https://ui.perfetto.dev ，把上面的 json 拖进去；")
    print("或者 tensorboard --logdir logs --port 6006 看 SCALARS 里的 perf/step_ms。")


if __name__ == "__main__":
    main()
