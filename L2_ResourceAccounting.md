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

