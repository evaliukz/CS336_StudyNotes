🧠 一、Lecture 13 & 14 核心一句话

❗数据不是“收集来的”，而是“设计出来的”

🗣️ 大白话

你不是：

❌ 抓一堆数据喂进去

而是：

✅ 你在构造一个世界，让模型在里面成长

🧠 二、为什么 Data 是 LLM 的“真正瓶颈”

Lecture 里反复强调一个点：

scaling laws 假设数据是“无限的”
但现实不是

🧠 现实问题

高质量数据有限

已经被大模型用过

再往上提升越来越难

🗣️ 大白话

GPU可以买
模型可以做大

但：

❗好数据是不可复制的

🧠 三、Data pipeline（核心结构）

这两讲的主线就是这条 pipeline：

1️⃣ 数据来源（Collection）

常见来源：

Common Crawl（网页）

GitHub（代码）

Books（书籍）

Wikipedia

Forums

🗣️ Lecture里的关键点

👉 不同来源 = 不同能力

🗣️ 大白话

你喂：

code → 会写代码

论文 → 更严谨

Reddit → 更像人

👉 数据来源决定模型性格

🧠 四、最大问题：Common Crawl 是垃圾场

Lecture 明确讲：

Common Crawl ≠ 高质量数据

🧠 问题

spam

广告

重复

非语言

🗣️ 大白话

就像：

把整个互联网都倒进来
但里面有很多垃圾

👉 所以必须：

filtering（过滤）

🧠 五、Filtering（Lecture重点）
方法 1️⃣ 规则过滤

长度

HTML结构

URL规则

方法 2️⃣ 语言检测

是否是目标语言

方法 3️⃣ 模型过滤（重点）

用一个模型判断：

quality

toxicity

usefulness

🗣️ transcript里的核心思想

用一个模型去筛选数据，再训练更强模型

🗣️ 大白话

先找一个“老师模型”：

👉 判断哪些数据值得学

🧠 六、Perplexity filtering（非常重要）
方法

用语言模型计算：

这段文本的 perplexity

结论

低 perplexity → 更自然

高 perplexity → 垃圾

🗣️ 大白话

就是：

这段话“像不像人写的”

⚠️ 但 Lecture 强调一个坑

perplexity 低 ≠ 一定高质量

🗣️ 举例

简单句子 → 低 perplexity

高质量复杂内容 → 可能更高 perplexity

👉 所以不能只靠它

🧠 七、Deduplication（去重）
为什么重要

Lecture强调：

重复数据会破坏 scaling

🧠 原因

重复数据：

降低有效信息量

让模型 overfit

🗣️ 大白话

就像：

一直刷同一套题

你会：

这题很强

但不会新题

🧠 实际做法

exact match

fuzzy match（更难）

🧠 八、Data Mixing（最关键决策之一）
问题

不同数据怎么配？

🧠 Lecture insight

mixing 决定能力分布

🗣️ 大白话

你给模型吃：

50% code → 更强 coding

50% math → 更强推理

👉 这是能力控制杆

🧠 工业做法

不同阶段不同 mixing

curriculum learning

🧠 九、Curriculum Learning（进阶点）
思路

不是一次喂所有数据

而是：

👉 按顺序喂

例子

1️⃣ 简单数据
2️⃣ 中等
3️⃣ 难

🗣️ 大白话

像：

小学 → 初中 → 高中

🧠 十、Tokenization（容易忽略但很关键）
为什么重要

token 数 = 训练成本

🧠 不同 tokenizer：

token 数不同

表达能力不同

🗣️ 大白话

一句话：

tokenizer 不同 → “单词切法不同”

👉 直接影响：

compute cost

数据效率

🧠 十一、Data contamination（污染）
问题

训练数据包含：

test set

benchmark

后果

👉 模型作弊

🗣️ 大白话

考试前看过答案

Lecture 强调：

这是当前 LLM research 最大问题之一

🧠 十二、Synthetic Data（Lecture 14重点）
思路

用模型生成数据训练模型

优点

无限扩展

可控

风险

错误放大

模型退化

🗣️ transcript核心思想

模型训练模型，是未来趋势，但很危险

🗣️ 大白话

AI教AI：

可能越来越偏

🧠 十三、Self-training / Bootstrapping
方法

1️⃣ 用模型生成答案
2️⃣ 过滤
3️⃣ 再训练

👉 这是：

GPT

DeepSeek

都在用的路线

🧠 十四、为什么数据工程比模型更难

Lecture隐含一个很深的点：

模型：

可以复现

有论文

数据：

不公开

不可复制

是核心壁垒

🗣️ 大白话

模型：

👉 everyone can copy

数据：

👉 moat（护城河）

🧠 十五、Lecture 13 & 14 总结
你必须记住的 6 件事
1️⃣ 数据是设计出来的
2️⃣ 数据来源决定模型能力
3️⃣ filtering 是必须的
4️⃣ dedup 是 scaling 的前提
5️⃣ mixing 决定能力结构
6️⃣ synthetic data 是未来，但有风险
🔥 最后给你一个“AI infra级理解”

你现在在做 AI infra，这两讲其实在教你：

真正的工作不是：

❌ 调模型

而是：

✅ 设计 data pipeline

包括：

data ingestion

filtering

ranking

mixing

iteration

👉 这就是：

LLM 数据系统
