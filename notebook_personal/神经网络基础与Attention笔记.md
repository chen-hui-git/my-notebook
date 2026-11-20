# FFN 与卷积：一个数值示例

假设有一个只有 3 个位置的一维序列：

\[
x=[1,\,2,\,4]
\]

这个例子用于直观比较 FFN 和卷积如何处理序列中的数值。为方便计算，FFN 部分使用简化变换；真实的 Transformer FFN 通常包含两层线性变换和一个激活函数。

## FFN：逐个位置独立处理

把 FFN 简化为 \(y=2x+1\)，并对每个位置分别应用同一个变换：

| 位置 | 输入 | 计算 | FFN 输出 |
|---|---:|---|---:|
| 1 | 1 | \(2\times1+1\) | 3 |
| 2 | 2 | \(2\times2+1\) | 5 |
| 3 | 4 | \(2\times4+1\) | 9 |

因此输出是：

\[
[3,\,5,\,9]
\]

每个位置的输出只依赖该位置自己的输入，不会直接使用其他位置的数值。

## 卷积：结合邻近位置

假设卷积核为 \([1,1,1]\)，步幅为 1，并在序列两端补 0。每个位置的输出是它与左右邻居的加权和：

| 位置 | 计算 | 卷积输出 |
|---|---|---:|
| 1 | \(0+1+2\) | 3 |
| 2 | \(1+2+4\) | 7 |
| 3 | \(2+4+0\) | 6 |

因此输出是：

\[
[3,\,7,\,6]
\]

中间位置的卷积输出用到了它左右邻居的信息。实际卷积核中的权重通常是通过训练学出来的，这里用全 1 的卷积核只是为了便于演示。

## 小结

- **FFN**：对每个位置分别做特征变换，不直接混合不同位置的信息。
- **卷积**：在局部窗口内结合相邻位置的信息。
- 在 Transformer 中，FFN 主要加工每个 token 的特征；注意力层负责让不同 token 交换信息。

## NAS 中的 FFN 分支：一个数值示例

假设输入 token 的特征是 \(x=[1,2]\)。NAS 超网提供两个候选 FFN 分支：

\[
\text{FFN}_{\text{小}}: 2\rightarrow2\rightarrow2
\qquad
\text{FFN}_{\text{大}}: 2\rightarrow3\rightarrow2
\]

这里用 ReLU 作为激活函数，并省略偏置，方便手算。

### 小 FFN 分支

令：

\[
W_1=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_2=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

逐步计算：

\[
[1,2]W_1=[1,2]
\rightarrow \operatorname{ReLU}([1,2])=[1,2]
\rightarrow [1,2]W_2=[1,2]
\]

小分支输出 \([1,2]\)。

### 大 FFN 分支

令：

\[
W_1=
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix},
\qquad
W_2=
\begin{bmatrix}
1&0\\
0&1\\
0&1
\end{bmatrix}
\]

逐步计算：

\[
[1,2]W_1=[1,2,3]
\rightarrow \operatorname{ReLU}([1,2,3])=[1,2,3]
\rightarrow [1,2,3]W_2=[1,5]
\]

大分支输出 \([1,5]\)。

| 候选分支 | FFN 结构 | FFN 输出 |
|---|---|---|
| 小 FFN | \(2\rightarrow2\rightarrow2\) | \([1,2]\) |
| 大 FFN | \(2\rightarrow3\rightarrow2\) | \([1,5]\) |

NAS 会在任务数据上评估不同候选分支，再结合准确率、延迟或参数量选择架构。更宽的 FFN 可能有更强的表达能力，但也可能需要更多计算。上面的权重仅用于演示，实际权重由训练得到。

若模块使用残差连接，通常会把 FFN 输出加回输入。例如小分支的结果会变成：

\[
x+\operatorname{FFN}_{\text{小}}(x)=[1,2]+[1,2]=[2,4]
\]

## Embedding：一个数值示例

Embedding（嵌入层）把离散的 token ID 映射成连续向量。可以把它理解为一张可训练的查找表：每个 token ID 对应一行向量。

假设词表和 embedding 表如下，向量维度为 3：

| Token | ID | Embedding 向量 |
|---|---:|---|
| 我 | 0 | \([0.2,\,0.1,\,0.7]\) |
| 爱 | 1 | \([0.6,\,0.3,\,0.2]\) |
| 猫 | 2 | \([0.1,\,0.8,\,0.4]\) |

句子“我爱猫”经过分词后得到 token ID：

\[
[0,1,2]
\]

Embedding 层按 ID 查表，得到三个向量：

\[
\begin{bmatrix}
0.2&0.1&0.7\\
0.6&0.3&0.2\\
0.1&0.8&0.4
\end{bmatrix}
\]

矩阵的每一行对应一个 token，三列是它的特征维度。实际模型会通过训练学习这些向量；这里的数值仅用于演示。

Transformer 通常还需要表示 token 的位置信息。假设第一个位置的位置向量是 \([0.01,\,0,\,0]\)，则第一个 token 的输入表示可以是：

\[
\text{Embedding}(\text{“我”})+\text{Position}(1)
=[0.2,0.1,0.7]+[0.01,0,0]
=[0.21,0.1,0.7]
\]

因此，Embedding 负责把 token ID 变成向量；位置编码再为向量补充位置信息，之后这些表示才送入注意力层和 FFN。

## 交叉熵：一个数值示例

交叉熵用来衡量模型给正确答案分配了多大概率。以预测下一个 token 为例，假设候选词是“我”“爱”“猫”，真实答案是“猫”。模型给出的概率为：

| 候选 token | 模型预测概率 | 真实标签 |
|---|---:|---:|
| 我 | 0.1 | 0 |
| 爱 | 0.2 | 0 |
| 猫 | 0.7 | 1 |

真实标签可以写成 one-hot 向量 \([0,0,1]\)。单个样本的交叉熵为：

\[
L=-\sum_i y_i\log(p_i)
\]

代入标签和预测概率：

\[
L=-(0\log 0.1+0\log 0.2+1\log 0.7)
=-\log(0.7)
\approx 0.357
\]

这里使用自然对数，所以损失单位是 nat。正确答案的概率越高，交叉熵越小；如果模型只给“猫”概率 \(0.1\)，损失就会变成 \(-\log(0.1)\approx2.303\)。训练时，模型会调整参数，让训练数据中的正确 token 获得更高概率；多个样本的损失通常再取平均。

## 模型训练中的 forward：最小示例

`forward` 的意思是“前向计算”：把输入送进模型，用当前参数计算预测结果。它本身通常不负责计算梯度或更新参数；训练循环会另外计算损失、反向传播梯度，再让优化器更新参数。

一个最小的二分类模型可以只包含一个线性层：输入有 2 个特征，输出 2 个类别的分数（logits）。

```python
import torch
from torch import nn
from torch.nn import functional as F


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)  # 2 个输入特征 -> 2 个类别分数

    def forward(self, x):
        return self.linear(x)


model = TinyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 两个样本，每行是一个样本的两个特征；标签 0 或 1 表示类别
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
labels = torch.tensor([0, 1])

logits = model(x)                         # forward：算出预测分数，形状为 [2, 2]
loss = F.cross_entropy(logits, labels)    # 预测分数与正确类别比较
optimizer.zero_grad()                     # 清空上一步留下的梯度
loss.backward()                           # 反向传播：计算参数的梯度
optimizer.step()                          # 更新参数
```

这个模型最核心的部分是**输入、带可训练参数的计算层、`forward` 计算规则和输出**。要进行训练，还需要正确标签、损失函数、反向传播和优化器。对于这个例子，`forward` 的计算可写成：

\[
\text{logits}=xW^T+b
\]

其中 \(W\) 和 \(b\) 是线性层的可训练参数。`logits` 是每个类别的原始分数；交叉熵会把这些分数转换为损失所需的概率并与标签比较。训练时，模型反复执行“前向计算 → 计算损失 → 反向传播 → 更新参数”。

如果是语言模型，最小结构还要根据任务接收 token ID，通常先做 Embedding，再经过网络层，最后输出词表中每个 token 的分数；但训练流程里的 `forward` 含义相同：给定输入和当前参数，计算预测结果。

## 自定义反向传播：完整的最小模型示例

下面把自定义运算、自定义层和模型组合起来。模型结构是：

```text
输入 → Linear → 平方层 → Linear → 预测值
```

```python
import torch
from torch import nn


# 自定义自动微分运算：y = x²
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # 保存 backward 计算梯度所需的输入
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * 2 * x  # d(x²)/dx = 2x
        return grad_x


# 将自定义运算包装成 nn.Module 层
class SquareLayer(nn.Module):
    def forward(self, x):
        return SquareFunction.apply(x)


# 组装完整模型
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 2)
        self.square_layer = SquareLayer()
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.square_layer(x)
        x = self.output_layer(x)
        return x


model = TinyModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 两个样本，每个样本有 2 个特征
x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
target = torch.tensor([[1.0], [0.0]])

# 一轮训练
prediction = model(x)             # 前向计算
loss = loss_fn(prediction, target)
optimizer.zero_grad()
loss.backward()                   # 反向传播
optimizer.step()                  # 更新 Linear 层参数
```

`TinyModel` 和 `SquareLayer` 都是 `nn.Module`，`SquareFunction` 则是定义运算及其梯度的自定义函数。它们之间的关系由下面的代码建立：

```python
self.square_layer = SquareLayer()  # 在 TinyModel 中创建并注册子层
x = self.square_layer(x)           # TinyModel.forward 中调用子层
return SquareFunction.apply(x)     # SquareLayer.forward 中调用自定义运算
```

因此前向调用链是 `TinyModel.forward()` → `SquareLayer.forward()` → `SquareFunction.forward()`。反向传播时，梯度经过 `SquareFunction.apply()` 会进入自定义的 `SquareFunction.backward()`。这个示例用于说明自定义梯度的接法；平方运算本身受 PyTorch 自动求导支持，实际模型通常无需为它手写 `backward()`。

## Attention 层：完整的数值示例

Attention（注意力）让序列中的每个 token 根据当前表示，对其他 token 分配不同权重，并汇总它们的信息。下面用三个 token 演示自注意力（self-attention）的计算。

### 输入与简化假设

假设三个 token 已经转换成二维向量，并按行放入矩阵 \(X\)：

\[
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
\]

为方便手算，假设查询、键和值的投影矩阵都是二维单位矩阵：

\[
W_Q=W_K=W_V=I=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

因此：

\[
Q=XW_Q=X,\qquad K=XW_K=X,\qquad V=XW_V=X
\]

这个 \(Q=K=V=X\) 是教学用的简化假设。实际模型通常会学习不同的 \(W_Q,W_K,W_V\)，让 Q、K、V 从不同角度表示输入。

### 1. 计算 Query 与 Key 的匹配分数

每个 Query 与每个 Key 做点积，得到分数矩阵：

\[
S=QK^T=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix}
=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
\]

例如左上角的分数，是第一个 Query \([1,0]\) 与第一个 Key \([1,0]\) 的点积：

\[
S_{11}=[1,0]\cdot[1,0]=1\times1+0\times0=1
\]

矩阵第 \(i\) 行第 \(j\) 列，表示第 \(i\) 个 token 的 Query 与第 \(j\) 个 token 的 Key 的匹配分数。

### 2. 缩放分数

将分数除以 \(\sqrt{d_k}\)，其中 Key 的维度 \(d_k=2\)：

\[
S_{\text{scaled}}=\frac{QK^T}{\sqrt{d_k}}
=\frac{S}{\sqrt2}
\approx
\begin{bmatrix}
0.7071&0&0.7071\\
0&0.7071&0.7071\\
0.7071&0.7071&1.4142
\end{bmatrix}
\]

例如左上角变为：

\[
\frac{S_{11}}{\sqrt2}=\frac{1}{\sqrt2}\approx0.7071
\]

### 3. 对每一行做 Softmax，得到注意力权重

对分数矩阵逐行应用 Softmax：

\[
A_{ij}=\frac{\exp(S_{\text{scaled},ij})}
{\sum_{j'}\exp(S_{\text{scaled},ij'})}
\]

例如左上角权重为：

\[
A_{11}=
\frac{\exp(0.7071)}
{\exp(0.7071)+\exp(0)+\exp(0.7071)}
\approx0.4012
\]

完整注意力权重矩阵约为：

\[
A\approx
\begin{bmatrix}
0.4012&0.1977&0.4012\\
0.1977&0.4012&0.4012\\
0.2483&0.2483&0.5034
\end{bmatrix}
\]

每行对应一个 Query 对所有 Key 的注意力分配，每行权重之和约为 1。比如第一行表示第一个 token 汇总信息时，分别从三个 token 取约 \(40.12\%\)、\(19.77\%\)、\(40.12\%\) 的信息。

### 4. 用权重汇总 Value

将注意力权重乘以 Value 矩阵：

\[
Z=AV\approx
\begin{bmatrix}
0.4012&0.1977&0.4012\\
0.1977&0.4012&0.4012\\
0.2483&0.2483&0.5034
\end{bmatrix}
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
=
\begin{bmatrix}
0.8024&0.5990\\
0.5990&0.8024\\
0.7517&0.7517
\end{bmatrix}
\]

例如输出矩阵左上角，是第一行注意力权重对所有 Value 的第一个分量进行加权求和：

\[
Z_{11}=0.4012\times1+0.1977\times0+0.4012\times1=0.8024
\]

因此第一个 token 的输出向量约为 \([0.8024,0.5990]\)。它是三个 Value 向量的加权和，权重由该 token 的 Query 与各 Key 的匹配分数决定。

### 计算流程与解读

\[
X\rightarrow(Q,K,V)\rightarrow QK^T
\rightarrow\frac{QK^T}{\sqrt{d_k}}
\rightarrow\operatorname{softmax}
\rightarrow AV=Z
\]

简而言之：Query 和 Key 决定“关注谁、关注多少”，Value 提供“取回什么信息”。示例里的输入向量和投影矩阵是为便于计算而设定的，并不代表真实语言模型对 token 的语义判断。真实 Transformer 还可能使用多头注意力、输出投影、残差连接和 LayerNorm；自回归语言模型还会用因果遮罩，避免当前位置读取未来 token。
