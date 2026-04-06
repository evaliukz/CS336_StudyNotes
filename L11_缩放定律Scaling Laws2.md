👉 Lecture 9 = scaling laws 是什么
👉 Lecture 11 = 怎么用 scaling laws 真的做模型（industry recipe）

🧠 一、这节课核心一句话

Scaling laws 不是拿来看图的，是拿来指导你“怎么训模型”的。

Lecture 11 的核心是：

不再讲公式

而是讲：真实公司怎么用这些规律做决策

（包括 Cerebras-GPT、MiniCPM、DeepSeek 等案例）

🧠 二、Lecture 11 的三大主线

这一讲其实围绕 3 件事展开：

1️⃣ 如何把 scaling law 用在真实训练里
2️⃣ muP（超重要）
3️⃣ learning rate / batch / schedule 如何 scale

👉 你可以理解为：

Lecture 9 是 physics
Lecture 11 是 engineering

🧠 三、为什么 Lecture 9 不够？

Lecture 9 告诉你：

loss ∝ power law

Chinchilla tradeoff

但没有告诉你：

❌ learning rate 怎么调
❌ batch size 怎么调
❌ 不同模型 size 怎么复用超参

👉 现实问题：

你现在：

小模型调好了

想放大 10 倍

问题：

超参数全崩了怎么办？

🧠 四、muP（这讲最重要的概念）
🌟 muP = Maximal Update Parameterization

一句话：

让不同大小模型用“同一套超参数”训练

🗣️ 大白话

现在问题是：

你有两个模型：

1B

10B

如果用同一个 learning rate：

👉 会发生：

小模型正常

大模型直接炸（loss 爆）

muP 的目标

让你可以：

在小模型上调好 → 直接用在大模型

🧠 为什么会炸？

因为：

参数变大 → 梯度规模变了 → update 不稳定

🗣️ 类比

你开车：

小车：油门踩 10% 很稳

大卡车：同样踩 10% → 直接冲出去

👉 muP 的本质：

调整“初始化 + 学习率”，让不同模型 behave 一样

🧠 muP 做了什么？

Lecture 里讲了三件核心事：

1️⃣ 初始化 scaling

不同层的 weight scale 不一样

2️⃣ activation scale 保持稳定

forward 不爆炸

3️⃣ update size 保持一致

backward 不崩

👉 目标：

模型变大，但“训练动态不变”

🧠 一句话总结 muP

让大模型像小模型一样训练

🧠 五、为什么 muP 很重要（工业视角）

因为：

👉 你不可能直接调 100B 模型

太贵了。

所以你只能：

1️⃣ 用小模型调参
2️⃣ extrapolate

👉 如果没有 muP：

scaling 上去全要重调

成本爆炸

👉 有 muP：

小模型调一次 → 大模型直接用

🧠 六、MiniCPM / Cerebras GPT 在干嘛

Lecture 11 用这些 case 讲：

scaling law + muP 在真实世界怎么用

🧠 核心 insight

这些模型在做：

用小模型找到“最佳训练配方”

然后：

→ scale up

🗣️ 大白话

就像：

小锅试菜

成功了再开大锅

🧠 七、第二个重点：WSD Learning Rate
🌟 WSD = Warmup + Stable + Decay
为什么要这个 schedule？

因为训练分三阶段：

1️⃣ Warmup（刚开始）

问题：

初始化不稳定

梯度乱跳

👉 所以：

学习率从小慢慢升

2️⃣ Stable（中期）

👉 模型进入正常学习

3️⃣ Decay（后期）

👉 精细优化

🗣️ 大白话

训练就像：

刚跑步：慢慢加速

中间：稳定跑

最后：冲刺但更精细

🧠 Lecture 的重点

WSD 的关键不是形式

而是：

这个 schedule 可以随着模型 scale 一起用

🧠 八、Batch Size scaling（很关键）
问题

模型变大：

👉 batch size 要不要变？

结论（工业经验）

batch size 通常要增大

但不能乱增

🧠 原因

batch 太小：

→ noise 太大

batch 太大：

→ generalization 变差

🗣️ 大白话

学习：

每次看 1 题 → 太 noisy

每次看 10000 题 → 学不到细节

👉 要找一个平衡点。

🧠 九、Scaling Recipe（这讲最重要总结）

Lecture 11 最核心就是：

一个完整的 scaling recipe

包含 5 个东西：
1️⃣ 模型大小（N）
2️⃣ 数据量（D）

（Chinchilla）

3️⃣ learning rate

（muP + scaling）

4️⃣ batch size
5️⃣ training steps

👉 这些必须一起调！

🧠 十、DeepSeek / LLaMA3 等案例在说明什么

Lecture 提到：

DeepSeek

LLaMA3

Hunyuan

这些 case 的共同点：

👉 不是简单 scaling

而是：

scaling + correct recipe

🗣️ 大白话

不是：

模型越大越好

而是：

模型大 + 参数对 + 训练方式对

🧠 十一、为什么 Lecture 11 很关键

因为：

👉 这是你从“理解模型”到“设计模型”的分水岭

Lecture 9 mindset

scaling exists

Lecture 11 mindset

I can design a scaling strategy

🧠 十二、你必须记住的 5 个结论
1️⃣ Scaling law ≠ 只看参数

必须一起看：

model

data

compute

2️⃣ muP 是关键工具

👉 用小模型调大模型

3️⃣ learning rate 不是固定的

👉 必须跟 scale 一起调

4️⃣ batch size 很重要

👉 影响稳定性 + 泛化

5️⃣ 成功模型 = recipe + scale
🧠 十三、最重要的大白话总结

我帮你压缩成一句你可以记一辈子的：

👉 Lecture 9：

告诉你“规律存在”

👉 Lecture 11：

教你“怎么用规律赚钱（省钱）”

🔥 最后给你一个“AI infra级理解”

你现在做 AI infra 转型，这一讲特别关键：

你真正要学的是：

不是：

❌ 怎么调一个模型

而是：

✅ 怎么设计一条“从小模型 → 大模型”的路径

真正高手做的是：

1️⃣ 小模型验证
2️⃣ scaling law 拟合
3️⃣ muP 固定超参
4️⃣ 一次性训练大模型

👉 这就是：

现代 LLM 工程方法论
