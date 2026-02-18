# 什么是 Memory accounting？

一句话：

算清楚“内存都被谁吃掉了”

在深度学习里，内存主要被下面几类东西占用：

模型参数（weights）

中间激活（activations）

梯度（gradients）

优化器状态（optimizer states）

临时 buffer（算子内部用）

很多人只算参数，这是最常见误区。

### 大白话例子

假设你有一个模型：

参数：1B（10 亿参数）

dtype：float16（2 bytes）

❌ 错误理解

模型大小 = 1B × 2B = 2GB
我的 GPU 16GB，稳了！

✅ 真实情况（训练时）：👉 直接爆显存

| 项目               | 内存     |
| ---------------- | ------ |
| 参数               | 2GB    |
| 梯度               | 2GB    |
| Adam 优化器（m + v）  | 4GB    |
| 激活（可能是参数的 2～5 倍） | 4–10GB |

# Tensor 

1️⃣ Tensor 是什么？

一句话：Tensor = 带形状的多维数组

0D：scalar（标量）

1D：vector

2D：matrix

3D+：tensor

x = torch.randn(32, 128)


意思是：

32 个样本

每个 128 维

2️⃣ Tensor 关键属性（非常重要）

| 属性            | 含义          |
| ------------- | ----------- |
| shape         | 形状          |
| dtype         | 数据类型        |
| device        | 在 CPU / GPU |
| requires_grad | 是否要算梯度      |

# tensors_memory

1️⃣ Tensor 占多少内存？

memory = num_elements × bytes_per_element

dtype 对照

| dtype              | bytes |
| ------------------ | ----- |
| float32            | 4     |
| float16 / bfloat16 | 2     |
| int64              | 8     |

例子：x = torch.randn(1024, 1024, dtype=torch.float32)

内存：1024 × 1024 × 4 ≈ 4MB

2️⃣ view vs copy（OOM 元凶）

y = x.view(-1)     # 不拷贝内存
z = x.clone()     # 真正拷贝

👉 看起来一样，内存完全不同

# Compute accounting（计算量核算）

1️⃣ 为什么要算计算量？

一句话：内存决定“能不能跑”，计算量决定“跑多快”

GPU 其实是：算力很强 + 内存访问很慢

2️⃣ 计算量单位：FLOPs

FLOP：一次浮点运算

FLOPs：总运算次数

FLOPS 指的是：Floating Point Operations Per Second（每秒浮点运算次数）

它是衡量算力（compute power）的核心指标，用来表示一个系统每秒能执行多少次浮点数计算。

为什么 AI 里特别看重 FLOPS？

深度学习训练本质上是：

大规模矩阵乘法

向量计算

反向传播梯度更新

这些操作几乎全部是浮点数运算，所以 FLOPS 就直接等于：

👉 你的模型能跑多快
👉 你能训练多大的模型
👉 训练成本有多高

| 单位     | 含义   | 规模    |
| ------ | ---- | ----- |
| GFLOPS | 10⁹  | 十亿次   |
| TFLOPS | 10¹² | 万亿次   |
| PFLOPS | 10¹⁵ | 千万亿次  |
| EFLOPS | 10¹⁸ | 百万亿亿次 |

需要注意的关键点

在 AI infra 里，FLOPS 有几个重要区分：

① 理论 FLOPS vs 实际 FLOPS

理论峰值 ≠ 实际利用率

实际利用率可能只有 30%–60%

② 精度不同 FLOPS 不同

FP32

FP16

BF16

FP8

精度越低，FLOPS 越高（因为硬件支持更密集计算）

FLOPS ≠ 训练速度

很多人误解：

FLOPS 高 = 模型训练快 ❌

实际还取决于：

内存带宽

网络带宽（多机训练）

IO

并行策略

kernel 优化

通信效率

AI infra 的核心就是：

如何把理论 FLOPS 转化成有效 FLOPS

一句话总结

在 AI infra 里：

FLOPS = 衡量算力规模的核心指标

有效 FLOPS = 真正决定训练效率的关键
