# 坏案例集：训练是怎么坏掉的

1 个**健康基线** + 5 个**故意练坏**的小实验，每个 200~300 步、几秒钟跑完。
用 TensorBoard 对比着看，能一眼认出"梯度消失 / 爆炸 / 震荡 / 冻结 / 神经元死亡"长什么样 ——
以后在真实训练里看到同样的曲线，你就知道该查哪里。

## 怎么跑

```bash
cd /Users/chenhui/Documents/projects/session/tb/badcases
./run_all.sh          # 跑完 6 个案例，日志写到 ./logs/
./serve.sh            # 启动 TensorBoard → http://localhost:6008
```

> 逐案拆解（每个案例"差在哪、为什么坏、怎么修"的完整版）见 **[DIAGNOSIS.md](DIAGNOSIS.md)**。

## 5 个案例

| 目录 | 一句话 | 在 TB 里看什么 | 修法 |
|---|---|---|---|
| **00_healthy**<br>健康基线 | 同样的数据/模型，**lr=0.05** | `loss` 0.729 → **0.086** 平滑下滑；`gradnorm` 0.40 → 0.086 缓降；`weights/fc01` 是一条缓慢移动的亮带 | 这就是"正常"该有的样子 |
| **01_vanishing**<br>梯度消失 | 20 层 Sigmoid + 小初始化 | `gradnorm/fc01` = 0，`gradnorm/fc20` ~ 1e-2，跨层差 **18 个数量级**；`loss` 一条水平线 0.693（= ln2，等于瞎猜） | 换 ReLU/GELU、He 初始化、加残差/BN、减层数 |
| **02_exploding**<br>梯度爆炸 | lr=10（正常 0.05） | `gradnorm/total` 出现 **385 的尖峰**；`loss` 从 0.73 冲到 **1593**，之后卡死在 2.1~2.8 再也回不来；权重 `|W|` 从 4.5 涨到 **223** | 降 lr、`clip_grad_norm_(params, 1.0)`、输入归一化 |
| **03_oscillating**<br>震荡不收敛 | lr=2 + batch=32（XOR 数据） | `loss` 在 0.009 ~ 0.7 之间反复跳，300 步还不收敛；`gradnorm` 上下乱窜 | lr ÷ 3~10、增大 batch、加 warmup/衰减 |
| **04_frozen**<br>权重冻结 | 只把最后一层传给优化器 | `gradnorm/fc01` 有值且**涨到 19**，但 `weights/fc01` 的直方图 11 个时刻**完全重叠**；`loss` 卡在 0.376 | 传 `model.parameters()`；核对优化器管了几个张量 |
| **05_dead_relu**<br>神经元全死 | 第一层 bias 初始化为 −100 | `gradnorm` 从第 0 步就贴着 0（1e-5 量级）；`loss` 恒定 0.693；权重直方图一条竖线不动 | 换 LeakyReLU/GELU、降 lr、加 BN、输入标准化 |

## 对照"健康"样本

目录里第一个 `00_healthy` 就是基线：同一套数据、同一个 3 层模型，**只有 lr=0.05 不同**。

```
loss       0.729 → 0.086     平滑下降
gradnorm   0.40  → 0.086     缓降，不塌陷、不尖峰
```

也就是说，**健康 = 单调缓降 + 层间同量级 + 无数量级尖峰 + 权重分布缓慢移动**。

> 注意：`00_healthy` 是 3 层网络，所以只有 `gradnorm/fc01`；`fc05/fc10/fc20`
> 只出现在 20 层的 `01_vanishing` 里。对比时看"形状"而不是看重合。

## 使用建议

1. 先切到 `DISTRIBUTIONS` 面板看 `weights/fc01`：健康是"一条缓慢移动的亮带"，冻结是"一条纹丝不动的竖线"。
2. 再看 `gradnorm/*`：一定要把 y 轴切成 **log scale**（梯度跨好几个数量级，线性坐标看不出来）。
3. 最后看 `loss/train`：水平线 = 没学，锯齿 = lr 太大，冲高 = 爆炸。

## 文件

```
DIAGNOSIS.md         逐案诊断：差在哪、为什么、怎么修（建议先读这个）
common.py            公用：造数据、算梯度范数、统一记录（loss / gradnorm / histogram）
01_vanishing.py      梯度消失
02_exploding.py      梯度爆炸
03_oscillating.py    震荡不收敛
04_frozen.py         权重冻结
05_dead_relu.py      神经元全死
00_healthy.py        健康基线（对照用）
run_all.sh           跑全部
serve.sh             启动 TensorBoard（默认 6008 端口）
```
