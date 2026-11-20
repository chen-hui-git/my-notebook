"""把 TensorBoard 记录的数据读回来，自己做统计 / 重新画图。

用法：
    .venv/bin/python learn/read_logs.py            # 默认读 tag "loss/train"
    .venv/bin/python learn/read_logs.py y          # 读 tag "y"（多个 run 都有）

它会做的事：
    1. 递归扫描 logs/ 下所有 event 文件
    2. 把每个 run 里指定 tag 的 (step, value) 读成普通 Python 列表
    3. 打印统计信息
    4. 用 matplotlib 重新画一张图 → learn/replot.png

这条路的用途：训练时用 TensorBoard 盯，出报告/论文图时从这里取数据自己画。
"""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_HERE.parent / ".mplcache"))  # 本机 ~/.matplotlib 不可写

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = _HERE.parent
LOG_ROOT = ROOT / "logs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "loss/train"
OUT = _HERE / "replot.png"


def load(tag):
    """返回 {run 名: [(step, value), ...]}"""
    series = {}
    for event_file in sorted(LOG_ROOT.rglob("events.out.tfevents.*")):
        run_dir = event_file.parent
        # size_guidance={"scalars": 0} = 不做抽样，全部读出来
        ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            series[str(run_dir.relative_to(LOG_ROOT))] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return series


def main():
    series = load(TAG)
    if not series:
        print(f"没有找到 tag = {TAG}")
        return

    print(f"tag = {TAG}")
    for run, pts in series.items():
        vals = [v for _, v in pts]
        print(f"  {run:22s} {len(pts):4d} 点   末值={vals[-1]:9.4f}   最小={min(vals):9.4f}")

    # ---- 加工：想怎么算就怎么算（这里演示找最小值落在第几步）----
    print()
    for run, pts in series.items():
        best_step, best_val = min(pts, key=lambda p: p[1])
        print(f"  [{run}] 最小值 {best_val:9.4f} 出现在 step {best_step}")

    # ---- 自己画图 ----
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for run, pts in series.items():
        ax.plot([s for s, _ in pts], [v for _, v in pts], label=run)
    ax.set_xlabel("step")
    ax.set_ylabel(TAG)
    ax.set_title(f"re-plotted from events: {TAG}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"\n图已保存：{OUT}")


if __name__ == "__main__":
    main()
