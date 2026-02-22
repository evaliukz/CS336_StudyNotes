# 1. 回顾 standard transformer
你实现的那个 Transformer，只是“教材版”, 真正 LLM 早就改了很多东西


<img width="427" height="521" alt="image" src="https://github.com/user-attachments/assets/0f003575-e30d-4bc0-88f5-63ee6820dd9b" />

现代 LLM 核心四件事：

1️⃣ pre-norm
2️⃣ RoPE
3️⃣ SwiGLU
4️⃣ no bias

你以后写 transformer 就用这个。

不要迷信：“Transformer就是固定结构”, 它一直在进化。

1️⃣ Pre-Norm vs Post-Norm（超重要）
2️⃣ LayerNorm vs RMSNorm（系统级）
3️⃣ Activation：ReLU → GELU → SwiGLU（模型表达力）
4️⃣ Positional encoding：sin → RoPE（长上下文关键）
5️⃣ Hyperparameters（真正影响模型）

## 第一大块：Normalization（整个第三章最核心）
🧩 为什么 Transformer 一定要 Norm？

深网络训练最大问题：

梯度爆炸

梯度消失

activation scale 不稳定

Norm 的作用：

把每层输入都“压回稳定范围”

Transformer 像一个几十层的 pipeline：

如果某层输出突然变得非常大：

→ 后面层全部崩

Norm 就像：

每一层前面都加一个“电压稳压器”

🧩 Post-Norm（老版本）

原始 transformer：

y = LayerNorm(x + f(x))

先算 attention/ffn，再 normalize。

问题

深模型训练非常难：

梯度传不回来

unstable

learning rate 要调很小

想象：

你已经算出一个巨大输出，再去 normalize

这时候梯度已经被放大/破坏。

🧩 Pre-Norm（现代 LLM 全部用）

y = x + f(LayerNorm(x))

先 normalize，再算 attention / ffn。

为什么更好？

因为：

每层看到的输入都是稳定的

梯度更容易 backprop

深层模型更稳

🗣️ 超大白话理解

Post-norm：

“我先把水泼出去，再想办法收拾”

Pre-norm：

“我先把水装进杯子，再倒”

当然更稳。

🧠 第二大块：LayerNorm vs RMSNorm

现代模型很多已经不用 LayerNorm 了。

🧩 LayerNorm 做了什么？

(x - mean) / std

同时：

减均值

除方差

🧩 RMSNorm 做了什么？

x / RMS(x)

只做：

scale normalization

不减 mean。

为什么这么改？

因为：

减 mean 不一定必要

计算更慢

GPU kernel 更复杂

RMSNorm：

更简单

更快

几乎不损性能

LayerNorm：

把数据“居中 + 缩放”

RMSNorm：

只把“幅度压回去”

结果：

差不多，但更省算力

现在很多模型：

Llama

Mistral

DeepSeek

→ 用 RMSNorm

第三大块：Activation evolution

这一段 slides 在讲：

FFN 里的激活函数决定模型表达力

🧩 最早：ReLU

max(0, x)

问题：

死神经元

不平滑

表达能力有限

🧩 后来：GELU

x * Φ(x)

🧩 现代：SwiGLU（重点）

FFN(x) = (xW1 ⊗ σ(xW2)) W3

为什么更强？

因为：

学习“哪些特征通过”

比单 activation 表达更强

更像 attention gating

本质：gated feedforward

🗣️ 大白话

GELU：

神经元开关

SwiGLU：

两个神经元互相商量再开

当然更聪明。

🎯 必记

现代 LLM：

FFN = SwiGLU

不是 GELU。

第四大块：FFN width

这一块 slides 非常关键。

🧩 FFN 才是算力黑洞

Transformer 每层：

attention cost

FFN cost

实际：

FFN FLOPs > attention

🧩 hidden dim 通常是：

d_ff ≈ 4 × d_model

原因：

提升表达能力

类似 MLP hidden expansion

🗣️ 大白话

attention 是“沟通”

FFN 是“思考”

而：

思考需要更多脑细胞

所以 FFN 很宽。

🧠 第五大块：RoPE（位置编码革命）

这一块是第三章第二大重点。

🧩 为什么需要 position？

Transformer 是 permutation invariant。

不知道 token 顺序。

🧩 最早：sin/cos encoding

绝对位置。

但：

extrapolation 差

长上下文崩

🧩 RoPE 原理

把位置编码：

写进 Q / K 的旋转矩阵里

attention 时自动包含相对距离。

🗣️ 大白话

sin pos：

每个 token 带一个“编号”

RoPE：

每个 token 带一个“方向”

attention = 看方向差。

为什么强？

relative position

长 context 更稳

extrapolate 更好

🎯 必记

现在所有模型：

GPT-4

Llama

Claude

→ RoPE 或其变体。

🧠 第六大块：Bias removal

slides 提到：

很多现代模型：

linear 无 bias

norm 无 bias

为什么？

因为：

bias 不太影响表达力

占参数

kernel 不好融合

大白话

删 bias：

减参数
提效率
几乎没损失

🧠 第七大块：Architecture 设计哲学（第三章灵魂）

这页是整章最重要。

Transformer architecture ≠ 数学问题

是：

engineering problem

决策依据：

训练稳定性

GPU效率

scaling law

memory bandwidth

大白话

不是：

“这个结构更优雅”

而是：

“这个结构训得出来 + 训得快”

🧠 第八大块：Hyperparameter

最后 slides 会强调：

architecture 只是第一步

真正影响：

depth

width

head 数

FFN ratio

scaling law 结论：

模型效果 ≈ 参数量 + 数据量 + compute

architecture 只是 baseline。

