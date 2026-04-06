好，这一讲（CS336 Lecture 15）基本是整门课的“最后一块拼图”之一：
👉 从 Data → Training → Inference → Evaluation 走到：

❗Post-training / Alignment（后训练 & 对齐）

我会按你熟悉的方式：
结构 +（基于公开课transcription的讲法重建）+ 大白话 + 工程视角

🧠 一、Lecture 15 核心一句话

预训练让模型“会说话”，后训练让模型“会做人”

🗣️ 大白话

Pretraining：

学语言模式

学知识

但：

❌ 不听话
❌ 不安全
❌ 不符合用户期望

👉 所以需要：

Post-training（后训练）

🧠 二、为什么 Pretraining 不够？

Lecture里一个核心观点：

Pretraining 目标 ≠ 用户目标

Pretraining 目标

👉 预测下一个 token

用户真实需求

👉

回答问题

写代码

帮助人

安全可靠

🗣️ 大白话

模型学的是：

“什么词最可能出现”

但用户想要的是：

“给我一个有用答案”

👉 中间有 gap

🧠 三、Post-training 的三大方法

Lecture 15 主要讲三类：

1️⃣ SFT（Supervised Fine-tuning）
2️⃣ Preference Learning（偏好学习）
3️⃣ RLHF / RL（强化学习）

我们一个个讲透。

🧠 四、SFT（最基础）
方法

用人工标注数据：

👉 input → ideal output

训练模型模仿。

🗣️ 大白话

就像：

老师给你标准答案，让你照着写

例子

Q: 写一个Python排序
A:（高质量代码）

🧠 优点

简单

稳定

成本低

❌ 缺点

不会优化“偏好”

无法表达“更好 vs 一般好”

🗣️ 大白话

SFT只能教：

什么是“对的”

但不能教：

什么是“更好”

🧠 五、Preference Learning（核心升级）
方法

不是给答案

而是：

👉 给两个答案，让人选：

A better than B

🧠 训练 reward model

学：

什么答案更好

🗣️ 大白话

不是背答案

而是：

学“审美”

举例

同一个问题：

A：正确但啰嗦

B：清晰简洁

人选 B

👉 模型学：

B > A

🧠 六、RLHF（最关键）
RLHF = Reinforcement Learning from Human Feedback
流程（Lecture核心）

1️⃣ SFT 初始化模型
2️⃣ 训练 reward model
3️⃣ 用 RL 优化模型

🧠 目标

最大化：

reward（人类喜欢程度）

🗣️ 大白话

像：

写作文

老师打分

你不断改进

🧠 七、为什么 RLHF 有用？

因为它解决一个关键问题：

❗人类偏好无法用“标准答案”表达

🗣️ 大白话

比如：

“更礼貌”

“更有帮助”

“更安全”

这些：

❌ 没有唯一答案

👉 RLHF 能学这种东西

🧠 八、RLHF 的问题（Lecture重点）
1️⃣ Reward hacking

模型学会：

👉 “骗 reward model”

🗣️ 大白话

像学生：

专门写老师喜欢的话，而不是好答案

2️⃣ Over-optimization

👉 模型变得：

很啰嗦

很“AI味”

3️⃣ 不稳定

RL训练：

难调

容易崩

🧠 九、DPO（更简单的替代）
Direct Preference Optimization
思路

👉 不用 RL

直接：

用 preference 数据训练

🗣️ 大白话

跳过：

reward model

RL

直接学：

A 比 B 好

🧠 优点

更稳定

更简单

更便宜

👉 现在 industry 用很多

🧠 十、Alignment（对齐）到底是什么

Lecture里一个关键问题：

什么叫 aligned？

三个层次：
1️⃣ Helpful（有用）
2️⃣ Honest（真实）
3️⃣ Harmless（安全）

👉 经典：

HHH

🗣️ 大白话

一个好模型：

能帮你

不胡说

不害人

🧠 十一、为什么 Alignment 很难
原因 1️⃣ 人类没有统一标准
原因 2️⃣ 场景不同
原因 3️⃣ tradeoff
🗣️ 大白话

比如：

太安全 → 不有用

太自由 → 不安全

👉 必须平衡

🧠 十二、System 级 alignment

Lecture 后半会强调：

alignment 不是只靠模型

还包括：

prompt

system message

tool

guardrails

🗣️ 大白话

不是：

模型自己变好

而是：

系统帮它变好

🧠 十三、为什么 post-training 是竞争核心
Pretraining：

👉 越来越同质化

Post-training：

👉 差异巨大

🗣️ 大白话

大家都有：

GPU

数据

但：

谁更“会教模型做人” → 谁更强

🧠 十四、Lecture 15 总结
你必须记住的 6 件事
1️⃣ Pretraining ≠ 产品模型
2️⃣ SFT 是基础
3️⃣ Preference learning 是关键
4️⃣ RLHF 是强但复杂
5️⃣ DPO 是更简单路线
6️⃣ Alignment 是系统问题
🔥 最后给你一个“AI infra级理解”

你现在做 AI infra，这一讲真正要学的是：

不只是：

❌ 模型怎么训

而是：

✅ 如何设计“人类反馈系统”

包括：

数据收集

标注

ranking

feedback loop

👉 这才是：

LLM 产品的核心引擎
