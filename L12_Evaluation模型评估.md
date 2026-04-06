🧠 一、Lecture 12 核心一句话

你怎么评估模型，就会得到什么样的模型

🗣️ 大白话

不是：

❌ 模型决定评估

而是：

✅ 评估决定模型进化方向

📌 课程里明确讲了：

整个行业都在“追 benchmark”

评估指标会直接影响模型设计

🧠 二、Evaluation 其实在做什么？

Lecture 里给了一个结构（非常重要）：

一个完整评估 = 4个问题

1️⃣ 输入是什么（inputs）
2️⃣ 怎么调用模型（prompt / agent）
3️⃣ 怎么评估输出（metrics）
4️⃣ 怎么解释结果

🗣️ 大白话

评估不是：

“跑个 benchmark 出个分数”

而是：

一整套系统设计

🧠 三、Evaluation 的最大问题：看起来简单，其实很坑

Lecture 强调：

评估“看起来很机械”，但其实非常复杂

🗣️ 大白话

你以为：

输入问题

模型回答

算分

就完了？

其实每一步都能“作弊”👇

🧠 四、问题1：你测的到底是什么？
例子

你想评估：

👉 模型能力

但实际上测到的是：

prompt engineering

chain-of-thought

tool usage

📌 Lecture 明确问：

你是在评估模型，还是评估整个系统？

🗣️ 大白话

就像：

你说测试学生水平

但：

有人用计算器

有人请家教

👉 结果不公平

🧠 五、Perplexity（基础但很重要）
定义（Lecture里给的）
𝑝
𝑒
𝑟
𝑝
𝑙
𝑒
𝑥
𝑖
𝑡
𝑦
=
(
1
/
𝑝
(
𝐷
)
)
1
/
∣
𝐷
∣
perplexity=(1/p(D))
1/∣D∣

表示：

模型对数据的“惊讶程度”

🗣️ 大白话

低 perplexity = 模型觉得“很合理”

高 perplexity = 模型觉得“看不懂”

🧠 为什么重要

Lecture 讲了两个关键点：

✅ 优点

平滑（适合 scaling law）

通用（所有任务都能用）

❌ 缺点

和真实能力不完全一致

用户不关心 perplexity

🗣️ 大白话

perplexity 像：

数学成绩

但用户关心的是：

会不会写代码 / 聊天好不好

🧠 六、Benchmark（主流评估体系）

Lecture 12 会系统讲各种 benchmark。

🧠 七、知识类 benchmark（Knowledge）
常见：

MMLU

GPQA

MATH

🧠 本质

测：

模型“知道多少”

🗣️ 大白话

像：

高考选择题

⚠️ 问题

Lecture 提到：

数据来源不透明

可能被训练过（泄漏）

👉 train-test contamination

🧠 八、Instruction / Chat 评估
常见：

Chatbot Arena

AlpacaEval

IFEval

🧠 特点

人类偏好

pairwise comparison

🗣️ 大白话

不是：

“对不对”

而是：

“哪个好”

⚠️ 问题

subjective

不稳定

容易被 prompt hack

🧠 九、Agent / Tool 评估
例子：

SWE-bench（写代码）

tool usage

🧠 本质

测：

模型能不能“做事”

🗣️ 大白话

不是考试

而是：

实际工作能力

🧠 十、Safety evaluation
测：

harmful output

jailbreak

🧠 难点

Lecture 提出一个关键问题：

什么是“安全”？

🗣️ 大白话

不同人标准不同：

医疗 → 不能错

聊天 → 可以宽松

🧠 十一、Open-ended generation（最难）
问题

没有标准答案

📌 Lecture 问：

怎么评估没有 ground truth 的任务？

🗣️ 大白话

比如：

写文章

聊天

👉 没有“唯一正确答案”

解决方案

human eval

LLM-as-judge

🧠 十二、Evaluation Crisis（超重要）

Lecture 强调：

我们正在经历“评估危机”

为什么？
1️⃣ Benchmark saturation

模型已经接近满分

2️⃣ 数据污染

模型见过题

3️⃣ 指标被优化

模型专门“刷分”

🗣️ 大白话

就像：

高考题泄露 + 刷题训练

👉 分数不再可信

🧠 十三、Cost-aware evaluation（工业核心）

Lecture 提到一个非常重要点：

评估必须考虑 cost

为什么？

因为：

更强模型 ≠ 更好产品

🗣️ 大白话

你有两个模型：

模型	分数	成本
A	90	$1
B	92	$100

👉 选哪个？

工业答案：

看 Pareto frontier（性能 vs 成本）

🧠 十四、真实世界评估（最重要）

Lecture 提到：

OpenRouter usage

用户选择

🧠 insight

用户用谁，谁就是好模型

🗣️ 大白话

不是：

leaderboard 第一

而是：

用户愿意用

🧠 十五、Evaluation 的四大陷阱（总结）
1️⃣ 测错东西

（model vs system）

2️⃣ 数据泄漏
3️⃣ 指标失真
4️⃣ 忽略成本
🧠 十六、Lecture 12 最重要的 5 个结论
1️⃣ Evaluation = 系统设计

不是跑分

2️⃣ Benchmark 会决定研究方向
3️⃣ Perplexity 重要但不够
4️⃣ 没有完美评估
5️⃣ 用户行为也是评估
🔥 最后给你一个“AI infra级理解”

你现在做 AI infra，这一讲非常关键：

真正工业问题是：

不是：

❌ 模型多强

而是：

✅ 在成本、延迟、能力之间找平衡

Evaluation 真正作用：

👉 帮你做决策：

用哪个模型

用什么配置

成本是否值得

🧠 最后一层理解（非常关键）

我帮你总结成一句你可以长期记的：

Evaluation 不是 measuring performance，而是 defining success

🧠 我给你一个非常关键的问题（面试级）

如果：

一个模型：

MMLU 95分

用户满意度很低

另一个：

MMLU 80分

用户很喜欢

👉 你选哪个？

我可以帮你讲：
OpenAI / Anthropic 实际是怎么选的（非常反直觉）
