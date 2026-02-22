# 回顾 standard transformer
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

# 🧠 第二大块：LayerNorm vs RMSNorm

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

# 第三大块：Activation evolution

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

# 第四大块：FFN width

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

# 🧠 第五大块：RoPE（位置编码革命）

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

RoPE 的核心：
不是把“位置”当成一个额外向量加到 embedding 上，而是把位置变成一种“旋转”，直接作用在 Q / K 上，让注意力天然带上“相对距离”的信息。

大白话：

每个 token 的 Q/K 都被“按位置转了一个角度”，两个 token 的相对距离，就等价于它们旋转角度差。

### 1）为什么普通位置编码（sin/cos 绝对位置）会不够好？
经典做法（Absolute Positional Encoding）

最常见的老方法是：

token embedding：表示“这个词是什么”

positional embedding：表示“它在第几个位置”

两个 相加

大白话：

我给每个词贴一个“座位号”，然后告诉模型：“你坐第 137 号位”。

问题：

这是“绝对座位号”，模型学到的很多模式其实更像“相对距离”
比如：主语和动词一般相距不远，括号闭合对应距离等等。

长上下文外推差
训练时最多见到 2k/4k 位置，推理时拉到 32k/128k，会“座位号没见过”。

RoPE 想解决的就是：

让模型更自然地学“相对位置/距离”，并对长上下文更友好。

### 2）RoPE 的直觉：用“方向”表示位置

想象你有一个向量（Q 或 K），它像一根箭头：

在第 1 个位置：箭头朝 10°

在第 2 个位置：箭头朝 20°

在第 100 个位置：箭头朝 1000°（当然会循环）

当你算 attention 的相似度：q · k
如果 q 和 k 的“方向差”很大，它们点积会变小；方向差小，点积更大。

大白话：

RoPE 不告诉你“你坐几号位”，它告诉你“你面朝哪个方向”。
两个人相距多远，就等于“你们面朝方向差多少”。

### 3）RoPE 到底怎么进 attention：只改 Q 和 K，不动 V

Transformer attention 里最关键的是：
<img width="534" height="269" alt="image" src="https://github.com/user-attachments/assets/2dc17b2a-c5bc-40b8-a4f6-fcdf956f61ae" />

R(t) 是一个“旋转矩阵”，取决于位置 t。

大白话

Q / K 先“按自己的位置转一圈”

再去做点积算相似度

V 不用转（很多实现就是这样）

4）关键魔法：它把“绝对位置”变成“相对位置”

这是 RoPE 最核心、也最值得记的一句话：

<img width="709" height="167" alt="image" src="https://github.com/user-attachments/assets/bd7610a6-4366-4129-b11b-8b23797789f4" />

看到了吗？最后变成 j-i（相对距离）。

大白话版本：

你把两个人都按自己的位置转了角度，最后他们的相似度只和“你俩相隔几步”有关，而不是“你俩分别坐第几排”。

这就是为什么 RoPE 特别适合语言模型：
语言规律大多跟“相对距离”强相关。

### 5）RoPE 实际怎么“旋转向量”？（最实用的实现直觉）

你不需要真的构造大矩阵 R。
RoPE 的旋转是对 hidden dim 做 两两一组 的 2D 旋转。

把向量拆成：

(x0, x1) 一组

(x2, x3) 一组

…

每组做旋转：
<img width="357" height="63" alt="image" src="https://github.com/user-attachments/assets/29743540-cf24-43ca-aaef-50277455f6c8" />

其中角度 
θ = 位置 * 某个频率。


把每两个维度当成一个平面坐标 (x,y)，然后按位置转角度。

为什么用不同频率？

有的维度转得慢：适合表示长距离

有的维度转得快：适合表示短距离
就像 sin/cos positional encoding 的多频率本质一样。

### 总结：

RoPE 只应用在 Q/K 上（主流实现）

它本质是“把位置变成旋转”，不是加向量

它让 attention score 依赖相对位置 j-i（核心性质）

实现上是两两维度旋转，不需要大矩阵

对长上下文更友好（尤其相对位置建模）

KV cache 时 RoPE 很自然：每个新 token 只需要用自己的 position 生成对应 cos/sin，然后旋转它的 Q/K（cache 里存旋转后的 K/V 或存原始 K 再旋转，取决于实现）

# 🧠 第六大块：Bias removal

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

# 🧠 第八大块：Hyperparameter 超参数

很多model的参数都是照抄别的model的，但T5很大胆。

最后 slides 会强调：

architecture 只是第一步

真正影响：

depth

width

head 数

FFN ratio

scaling law 结论：

模型效果 ≈ 参数量 + 数据量 + compute

## Aspect ratio：模型不是“参数越多越好”，而是“深度 vs 宽度比例（aspect ratio）非常关键”

#### 什么是 aspect ratio？

在 Transformer 里，两个最重要的结构超参数：

Depth（层数 L）

Width（隐藏维度 d_model）

模型参数大约是：Params≈L×dmodel^2
​
所以当你固定参数量时：

你可以选：

深而窄（很多层，小宽度）

浅而宽（少层，大宽度）

Aspect ratio 就是：D/W

#### 宽度 = 表达能力（每一层多聪明）

宽度大：

每个 token 表示空间更大

FFN 更强

每层“脑容量”更大

在 Transformer 里，每个 token 是：一个向量

"cat" → [0.12, -0.3, … , 0.88]

这个向量长度叫：d_model

为什么叫“宽”？

因为：向量维度越大，就像“每层的空间越宽”。4096 维 vs 768 维：4096 可以表达更复杂特征, 类似“脑容量更大”

width 在 Transformer 里具体影响什么？

非常重要。

1️⃣ embedding size

token 表示能力。

2️⃣ attention 维度

Q/K/V 全是 width 维。

3️⃣ FFN 规模

FFN 通常：d_ff=4×d_model, width 越大，FFN 越爆炸。
	​
**这就是 Transformer 的 width**

#### 深度 = 推理步骤（能做多少次变换）

Transformer 是一层一层叠的 block：
```
embedding
↓
Transformer block 1
↓
Transformer block 2
↓
Transformer block 3
...
↓
output
```
这些 block 的数量：L 就是 depth
每个 block 包含：
```
attention
+
ffn
+
norm
```
每一层：attention 重新组合信息、信息交流, ffn 重新加工信息

层数多：

信息可以逐层抽象

更复杂的组合推理

表达层次更多

大白话：

深模型 = 思考步骤多, 信息经过更多次变换。

depth 在 Transformer 里影响什么？

1️⃣ reasoning chain

层数越多：

抽象能力越强

复杂组合能力越强

2️⃣ receptive field processing

多层 attention 可以逐层传播信息。

### 三、那是不是越深越好？

不是。

这里是关键点。

情况 A：太宽太浅（胖但不高）

问题：

每层很强

但推理步数不够

复杂组合能力弱

大白话：

脑容量大，但只思考两步。

情况 B：太深太窄（高但很瘦）

问题：

每层表达能力不足

信息压缩严重

梯度更难传

大白话：

思考很多步，但每步都很笨。
