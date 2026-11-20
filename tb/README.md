# TensorBoard 小 demo：从看懂面板，到硬件 profiling

三个独立小脚本，从"画一条 curve"讲到"查清每一步的时间花在哪"。
每个脚本几秒钟跑完，不依赖外部数据集。

## 目录结构

```
tb/
├── 01_basics.py       基础 API：scalar / histogram / figure / text / graph / PR / hparams
├── 02_profiler.py     算子级 & 硬件级 profile：torch.profiler → trace 时间线
├── 03_hw_monitor.py   训练中采 CPU / 内存 / 显存 → 和 loss 画在同一块面板
│
├── learn/             手把手 5 个小台阶（推荐从这里入门）
│   ├── step1.py       最小可用：一条曲线
│   ├── step1b.py      同一个 sin，横轴疏密不同
│   ├── step2.py       两个 run 叠在一张图上
│   ├── step3.py       真训练：loss + 参数曲线
│   ├── step4.py       把训练过程拍成照片（IMAGES 面板）
│   ├── step5.py       直方图 + 范数趋势（HISTOGRAMS / DISTRIBUTIONS）
│   └── read_logs.py   把 TB 的数据读回来，自己加工重画
│
└── badcases/          训练是怎么坏掉的：1 个健康基线 + 5 个坏案例
    ├── README.md      一览表
    ├── DIAGNOSIS.md   逐案诊断：差在哪、为什么、怎么修
    ├── 00_healthy.py  健康基线
    └── 01~05_*.py     梯度消失 / 爆炸 / 震荡 / 冻结 / 神经元死亡
```

**新手建议路线**：先按 `learn/` 的 5 个台阶走一遍，再回来看 `01_basics.py`；
想知道"坏掉的训练长什么样"就开 `badcases/`。

---

## 1. 快速开始

```bash
cd tb                      # 本目录
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt    # 国内建议加 -i https://pypi.tuna.tsinghua.edu.cn/simple

./run_all.sh               # 依次跑 01 / 02 / 03，生成 logs/
./serve.sh                 # 启动 TensorBoard（也可以 ./serve.sh 6007 换端口）
```

浏览器打开 **http://localhost:6006** 。左上角勾选框控制显示哪些 run。

> `learn/` 和 `badcases/` 各自有独立的日志目录和启动脚本：
> `cd learn && ../.venv/bin/python step1.py`、`cd badcases && ./run_all.sh && ./serve.sh`。

---

## 2. 三个脚本，分别该看哪里

### `01_basics.py` —— 八个最常用 API

跑两个只有隐藏层宽度不同的实验（`hidden_128` vs `hidden_8`），各自一个目录，
TensorBoard 里自动叠在同一张图上对比。

| 在网页上看 | 对应代码 | 你应该看到 |
|---|---|---|
| **SCALARS**（新版叫 Time Series） | `add_scalar` | 两组 `loss/train`、`loss/val` 曲线。hidden_128 收敛到 val_acc≈0.91，hidden_8 卡在 ≈0.70 |
| 同上，同一个卡片里两条线 | 两个 tag 共享前缀 `loss/train`、`loss/val` | 自动合成一张图，图例分 train / val |
| **CUSTOM SCALARS** | `add_custom_scalars` | 我排过版的"训练总览"卡片：Loss、Accuracy 上下并排 |
| **IMAGES** | `add_figure` / `add_image` | `images/decision_boundary` 有 13 帧，把滑块从 step 0 拖到 1200，能看到决策边界一步步把两条螺旋拧开 |
| **DISTRIBUTIONS / HISTOGRAMS** | `add_histogram` | `weights/fc1`、`grads/fc1` 的分布随训练变化 |
| **TEXT** | `add_text` | `config` 里的 Markdown 实验记录 |
| **GRAPHS** | `add_graph` | 网络结构图（双击节点可以展开） |
| **PR CURVES** | `add_pr_curve` | precision-recall 曲线 + 阈值滑条 |
| **HPARAMS** | `add_hparams` | 表格视图：两个 run 的 `hidden` / `lr` / `epochs` 和最终的 `hparam/val_acc`、`hparam/val_loss` |

**这次要记住的一件事**：多个 run 的曲线能叠在一张图上，靠的是
**同一个 tag + 不同的 log 目录**。目录名就是图例名。

### `02_profiler.py` —— 硬件 / 算子级 profile

`torch.profiler` 采的是 kernel 级信息：每个算子花多久、什么时候在等设备、内存峰值在哪。

脚本跑完会在终端打出两张表：

```
=== 按 CPU 总耗时排序 Top 15 ===
aten::native_batch_norm          24.72%   2.325ms
aten::native_batch_norm_backward 10.29%   967.7us
aten::threshold_backward          8.06%   757.7us
aten::_mps_convolution            7.82%   735.7us
...
```

以及实测吞吐（本机：batch=32 的 3×64×64 CNN，中位 step ≈ 5ms，5～7k samples/s，每次跑会有浮动）。

时间线 trace 在：

```
logs/profiler/cnn_profile/*.pt.trace.json
```

**看 trace 的方式**：打开 <https://ui.perfetto.dev>，把 json 拖进去（或 chrome://tracing）。
你会看到一行 `ProfilerStep*` 一个 step，展开能看到 `forward` / `backward` / `optimizer_step`
三个自定义区间（来自 `record_function`）和里面每一个 `aten::` 算子；横向看谁挨着谁、
哪个算子慢、有没有空泡。

> **为什么这里不用 TensorBoard 的 PROFILE 面板**
> 那个面板由 `tensorboard-plugin-profile` 提供，它加载时会执行 `import tensorflow` 探测能力；
> TensorFlow 目前没有 Python 3.14 的 macOS 轮子，所以插件不会激活，面板不会出现。
> trace 文件本身是通用 JSON，Perfetto 看效果一样。
> 如果你的环境能装 TensorFlow（Python ≤ 3.13），`tensorboard --logdir logs` 里会自动多出 PROFILE 面板。

### `03_hw_monitor.py` —— 把硬件指标接进训练看板

`torch.profiler` 适合"查一次"，而日常训练要的是"全程盯着"。做法很土但最通用：
**自己采指标，用 `add_scalar` 写进去**。

| 面板 | tag | 本机实测 |
|---|---|---|
| 自定义卡片"训练 vs 硬件" | `loss/train`、`acc/train` | loss 0.69 → 0.03 |
| | `hardware/mps_alloc_mb` | 稳定在 196MB（forward 后的激活值）；**平的就是没泄漏，一路涨才是有东西没释放** |
| | `hardware/mps_driver_mb` | 1071MB（驱动缓存池，只涨不缩是正常的） |
| | `hardware/cpu_pct` / `sys_cpu_pct` | 本进程 / 整机 CPU |
| | `perf/throughput` | 约 2k samples/s |

结束时脚本会打印 `empty_cache: driver 1071MB -> 47MB`，这是"缓存池"和"真实占用"的区别：
池子留着给下次复用，想让内存还给系统必须显式 `torch.mps.empty_cache()`。

---

## 3. 心智模型（比记 API 重要）

1. **一个 log 目录 = 一个 run**。`SummaryWriter(log_dir="logs/basics/hidden_128")`
   → 左侧列表里出现 `basics/hidden_128`。想对比实验就多开几个目录。
2. **tag 用 `/` 分层**。`loss/train`、`loss/val` 会自动归到 `loss` 组、画进同一张图；
   不需要 `add_scalars` 这个老 API（它的副作用是给每个子 tag 建子目录，左侧会冒出一堆"假 run"）。
3. **step 可以是你自己的计数器，也可以不传**（默认用墙钟时间）。
   跨 run 对比要用同一套 step 语义，否则 x 轴对不齐。
4. **写完要 `flush()` / `close()`**。尤其是长训练被 kill 的时候，没 flush 的最后一段会丢。

---

## 4. 硬件 profiling 的三层手段

| 层级 | 工具 | 开销 | 回答什么问题 |
|---|---|---|---|
| 宏观（全程） | 自己采指标 → `add_scalar`（`03`） | 极低 | 内存涨不涨？CPU 谁占的？吞吐稳不稳？ |
| 微观（抽样） | `torch.profiler` → trace / 算子表（`02`） | 中高，别全程开 | 哪个算子慢？哪里有同步等待？ |
| 系统级 | NVIDIA: `nsys` / `ncu`；macOS: Xcode Instruments + `torch.mps.profiler` | 低 ~ 很高 | 驱动层、显存带宽、kernel 内部 |

常用 API 备忘：

```python
# 显存（macOS MPS）
torch.mps.current_allocated_memory()   # 张量真实占用，删了张量就掉
torch.mps.driver_allocated_memory()    # 驱动缓存池
torch.mps.recommended_max_memory()     # 系统建议上限
torch.mps.empty_cache()                # 把池子还给系统

# 显存（NVIDIA CUDA，写法对称）
torch.cuda.memory_allocated() / memory_reserved() / max_memory_allocated()

# 全设备内存快照（torch >= 2.x，跨设备统一接口）
torch.cuda.memory.memory_summary()     # CUDA
```

---

## 5. 常见坑（都是我这次踩过 / 验证过的）

1. **端口被占**：`./serve.sh 6007`，或者先 `lsof -iTCP:6006 -sTCP:LISTEN` 看看是谁。
2. **面板看不到数据**：先确认 `--logdir` 指对了（指到 `logs`，不是指到某个 run 目录）；
   再点右上角刷新，或者用 `--reload_interval 5`（`serve.sh` 已经带了）。
3. **重跑 demo 曲线会叠加**：同一个目录写两次，曲线会连在一起，看着像"跳变"。
   想干净就 `rm -rf logs` 再跑。
4. **`add_figure` 用完必须 `plt.close(fig)`**，否则跑几百步之后内存爆掉（我在 `01` 里每处都关了）。
5. **别在 profile 区域里写 `.item()` / `print()` / `synchronize()`**。它们会强制同步，
   让 `aten::_local_scalar_dense` 直接冲上耗时榜第一，你就看不到真实的计算分布了。
   （`02` 里专门把测吞吐的循环放在 profile 外面。）
6. **MPS 是异步的**：算时间必须 `torch.mps.synchronize()`，否则测到的是"下发命令"的时间。
7. **图片/图表的 tag 不要用中文 + 空格混排**，某些插件对路径字符敏感；用 `loss/train` 这种最省事。
8. **`add_hparams` 会另建一个子目录**（`hparam_xxx/`），所以左侧会多一个 run —— 这是正常的，
   HPARAMS 面板正是按这个目录里的 experiment 摘要来读超参的。

---

## 6. 想继续练手

- 把 `03` 的采样改成后台线程 + `walltime=` 时间轴，做到 10ms 一次（曲线会更细）；
- 在 `01` 里换 SGD / 加学习率调度，用 CUSTOM SCALARS 对比三组 lr；
- 把 `logs` 换成本地跑的 MNIST 训练，体会 IMAGES 面板里"预测结果越来越对"的过程；
- 用 `torch.profiler` 对比 `batch=32` 和 `batch=256` 的 trace，看 GPU 空泡怎么变。

---

## 7. 清理

```bash
rm -rf logs .mplcache     # 数据 + matplotlib 字体缓存
```
