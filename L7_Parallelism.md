# 一、这一章到底在讲什么

这一章想解决两个特别现实的问题：

第一，模型太大了，一张 GPU 放不下。
第二，就算放得下，只用一张 GPU 训练也太慢。

所以第 7 章本质上是在讲：

怎么把“一个模型、一个 batch、一次训练步骤”拆给很多 GPU 和很多机器一起做。
而且拆法不是只有一种，得根据你最缺什么来选：
有时候你最缺的是显存，有时候你最缺的是带宽，有时候你最缺的是大 batch。这一讲反复强调，这三样都像资源，要省着花。

大白话版一句话：

第 7 章就是“大模型训练怎么组团干活”。

# 二、这一章最核心的总思路

你先记这一句，后面所有内容都会围着它转：

并行训练不是免费午餐。
你想省显存，就可能多通信。
你想多加 GPU，就可能受 batch size 限制。
你想让模型分到很多机器，就要决定：
到底传的是梯度、参数，还是激活值。

所以这一讲不是只教你名词，而是在教你一种判断框架：

先问：我卡的是显存，还是算力，还是通信？

再问：我该复制模型，还是拆模型，还是拆激活？

最后问：我的 GPU 之间连接快不快？是在一台机器里，还是跨机器？

# 三、先打底：为什么多机训练会突然变复杂

单 GPU 的世界里，你主要操心：

前向

反向

优化器更新

但到了多 GPU / 多机，突然多了一个新世界：

GPU 和 GPU 要互相说话

同一台机器里的 GPU 说话比较快

跨机器说话比较慢

有的并行方案特别吃带宽

有的并行方案对慢网络更友好

讲课里拿 GPU 集群举例：
一台机器里多张 GPU 往往用 NVLink / NVSwitch 这类很快的互连；
但跨机器通常要走更慢的网络，比如 InfiniBand。
这件事非常重要，因为它直接决定：

哪些并行方式适合“机内”，哪些适合“跨机”。

大白话：

同屋的人喊一声就能听见。隔壁楼的人要打电话。
所以你不能把所有通信都当成一样便宜。

GPU vs. TPU

<img width="2071" height="1239" alt="image" src="https://github.com/user-attachments/assets/869f81c8-8483-44be-b787-081bf86ead5e" />


# 四、先学会“几种集体通信”，不然后面会晕

这一讲先花了一块内容讲 collective communication，也就是“集体通信操作”。最重要的是：

all-reduce

all-gather

reduce-scatter

broadcast

reduce

### 1）all-reduce 是什么

每张 GPU 都有一份东西，比如梯度。
现在要把它们加起来，然后让每张 GPU 都拿到总和。

大白话：

4 个同学各自算了一部分分数，最后要把总分算出来，而且每个人手里都要有一份总分。
这就是 all-reduce。

### 2）all-gather 是什么

每张 GPU 只拿着自己那一小块数据。
现在要把这些小块拼起来，让所有 GPU 都看到完整数据。

大白话：

每个人手里只有拼图的一小块，现在要互相交换，最后每个人都看到整幅图。

### 3）reduce-scatter 是什么

先把大家的数据做 reduce，比如求和；
但不是每个人都拿完整结果，而是每个人只拿结果中的一部分。

大白话：

大家一起做总账，但做完后不是每人拿整本账本，
而是甲拿第一页，乙拿第二页，丙拿第三页。

### 4）这一讲最关键的等价关系

讲课里特别强调了一件事：

all-reduce ≈ reduce-scatter + all-gather
至少在带宽受限的角度看，它们是等价的。

<img width="2000" height="1204" alt="image" src="https://github.com/user-attachments/assets/52027bde-b2fb-4218-9917-159ae52c258a" />


这个等价关系非常重要，因为后面 ZeRO / FSDP 就是靠这个思路玩的：

以前你是先 all-reduce 梯度，让每个人都拿完整梯度；
现在你可以先 reduce-scatter，让每人只拿自己负责那一块的总梯度；
更新完后再 all-gather，把更新后的参数拼回来。

大白话：

以前是“先把整本账本复印给所有人，再各自更新”。
现在变成“每人只负责自己那几页，更新完再拼起来”。
这就省显存了。

# 五、第一大类并行：Data Parallel（数据并行）

这是最自然、最常见、也是大家最早接触的并行方式。讲课里把它定义得很直接：

模型参数先复制到每个 GPU 上。
再把 batch 切成几份，每个 GPU 吃不同数据。
每张 GPU 各自前向、反向，算出自己的梯度；
然后大家把梯度同步，再一起更新参数。

大白话：

像 4 个老师拿着同一本教材，
但每人批改不同学生的作业。
批改完之后，4 个人开会，把意见汇总，
然后一起修改教材。

### 1）它最大的优点：算力扩展很舒服

如果 batch 足够大，每个 GPU 都能拿到一块不小的数据，
每张卡都能比较忙，算力利用率不错。

大白话：

活够多的话，多招几个人就真能更快。

### 2）它最大的缺点：显存特别浪费

因为每张 GPU 都要放：

一整份模型参数

一整份梯度

一整份优化器状态，比如 Adam 的一阶矩、二阶矩

还有可能有 FP32 master weights

课程里明确说，朴素数据并行的参数相关内存非常夸张，可能需要“每个参数多份副本”，总的参数相关内存很容易到十几字节每参数，真正的负担往往主要来自优化器状态。

大白话：

你本来只想带一本书出门，
结果每个人还背着：

书的正文

批注本

历史批注本

统计表

备份本

最后包最重的不是书本身，而是“管理这本书的那些记录”。

# 六、为什么 Adam 这么吃显存

这一段你一定要建立直觉，因为后面 ZeRO 就是在治这个病。

Adam 不是只存参数本身。它还要存：

参数本体

梯度

一阶矩（momentum）

二阶矩（variance）

有时还有 FP32 master copy

课程里专门点出来：优化器状态是朴素数据并行里非常大的内存来源。

大白话：

Adam 像一个“记性很好”的老师。
它不只看你这次梯度，还要记：

你过去大概往哪边走

你过去波动大不大

所以它训练效果常常不错，
但代价就是：很占内存。

# 七、ZeRO：数据并行的“聪明升级版”

这一讲最重要的一块内容之一，就是 ZeRO。
ZeRO = Zero Redundancy Optimizer。
核心思想非常朴素：

既然复制太浪费，那就别什么都复制到每张 GPU 上。

# 八、ZeRO Stage 1：先把优化器状态拆开
它做了什么

每张 GPU 还是有完整参数、完整梯度。
但是优化器状态不再每张卡都复制一份，
而是分片存到不同 GPU 上。

训练怎么做

每张 GPU 还是对自己那份数据算完整梯度。
然后做 reduce-scatter，让每张 GPU 只拿到“自己负责那部分参数”的总梯度。
接着各自用自己手里的优化器状态更新自己那部分参数。
最后再 all-gather，把更新后的参数同步给大家。

为什么很妙

讲课里明确说，Stage 1 的关键魔法就在于：

reduce-scatter + all-gather 的带宽代价，和原来 all-reduce 差不多。
所以在带宽受限视角下，Stage 1 基本像是“白捡”的显存收益。

大白话

以前是每个人都带着全套会计账本。
现在变成：

甲专门管工资账

乙专门管报销账

丙专门管税务账

大家先把数据汇总给对应负责人，
负责人改完，再把结果发回来。

所以：

每个人脑子里不用都记全部账了。

# 九、ZeRO Stage 2：再把梯度也拆开
它做了什么

在 Stage 1 基础上，进一步把梯度也分片。

新的难点是什么

课程里说得很清楚：
不能先完整算出整个梯度向量，再慢慢分发，
因为那一瞬间你又会爆显存。

所以正确做法是：

反向传播到某一层，就立刻把这一层梯度 reduce / scatter 给对应 GPU，然后立刻释放。

大白话

不是等全班试卷都堆满办公室才开始分发，
而是老师批完一叠，立刻交给负责这一科的人，
自己桌上马上清空。

所以 Stage 2 的本质是：

边反向、边传、边扔。

# 十、ZeRO Stage 3 / FSDP：连参数也拆开

这一阶段最猛。
课程里直接说：

FSDP 本质上就是 ZeRO Stage 3。

它做了什么

现在不只是优化器状态和梯度分片，
连参数本身也分片。
也就是说，默认情况下，没有哪张 GPU 永久保存完整模型。

那前向怎么算

要用到某一层时，先 all-gather 把这一层参数临时凑齐；
算完这一层前向，就把这层参数释放掉；
下一层再临时 gather。

反向怎么算

反向时也是类似思路：

需要哪层参数，就 gather

算出这层梯度

再 reduce-scatter

再释放不需要的参数和梯度

为什么听起来很慢，实际上还行

因为可以做 通信和计算重叠。
课程里专门讲了 prefetch 的思路：
你在算当前层时，后台就开始拉下一层的参数。
所以虽然通信变多，但很多时间能被掩盖掉。

大白话

这像你家太小，放不下整套家具。
于是你改成：

这会儿要用餐桌，就先搬餐桌进来

吃完饭立刻搬出去

下一步要用书桌，再搬书桌进来

听上去很折腾，
但如果你能做到“上一件还在用，下一件已经在路上”，
那就没你想得那么慢。

# 十一、为什么 FSDP 很强，但也不是白来的

FSDP 的优点很大：

显存节省非常明显

能让大很多的模型塞进同样硬件

对模型结构要求相对没那么苛刻，包装通用网络也比较方便

但代价也存在：

通信比朴素 DDP 更多

需要更精细地 overlap communication and computation

它不自动解决所有 activation memory 问题

大白话：

FSDP 是显存省钱高手，但通信账单会更高。

# 十二、这一讲里一个特别重要的思想：Batch size 也是“资源”

这句话是第 7 章很值钱的地方。

课程里明确说：

数据并行不是无限扩展的，因为你最多只能把 batch 切到每张 GPU 至少有一个样本。
再往下就没法切了。
而且 batch size 太大也会有收益递减，和 critical batch size 有关。

大白话：

你有一锅饭。
4 个人吃可以分 4 碗。
如果来 100 个人，就没法“每人分 0.04 碗”还指望大家都吃得有效率。

所以 batch size 不是想多大就多大。
它像预算一样，得分配给不同并行方式使用。

# 十三、第二大类并行：Model Parallel（模型并行）

当模型大到复制不起时，就得考虑：

不是每张 GPU 都放整个模型，
而是把模型本身拆开。

课程里把模型并行主要分成两类来讲：

pipeline parallel

tensor parallel

# 十四、Pipeline Parallel（流水线并行）
1）最直观的想法

神经网络不是一层一层吗？
那就把不同层分给不同 GPU：

GPU0 负责前几层

GPU1 负责中间几层

GPU2 负责后几层

前向时激活从前往后传，
反向时梯度从后往前传。

大白话：

像工厂流水线：

第一站切菜

第二站炒菜

第三站装盘

### 2）为什么朴素流水线很烂

如果你一次只送一个样本进流水线，
就会出现很大的 bubble：
大部分 GPU 在很多时间里都在等。

课程里直接说，这种最简单的流水线利用率很差，
甚至像“加了 4 张 GPU，却只有 1 张 GPU 的吞吐”。

大白话：

第一站在干活时，第二三四站都闲着；
等第一站干完，第二站才开始；
这不叫高效分工，这叫排队发呆。

### 3）怎么改进：microbatch

更聪明的做法是把一个大 batch 再拆成很多 microbatch。
第一个 microbatch 从 GPU0 传给 GPU1 后，
GPU0 就可以立刻处理第二个 microbatch。
这样几张 GPU 就能像流水线一样同时忙起来。

Bubble 大小和什么有关

课程里给了直觉公式：

overhead / useful compute ≈ (stages - 1) / microbatches

所以 microbatch 越多，bubble 相对越小。

大白话

流水线要高效，不是因为站点多，
而是因为前一个盘子刚离开第一站，第二个盘子立刻补上。

### 4）为什么大家还是会用 pipeline parallel

虽然它麻烦，还会有 bubble，
但它有两个现实优点：

第一，能把模型和激活都沿深度方向分散到多张卡上，省显存。

第二，它传的是激活，很多时候是点对点通信，对慢网络比较友好。

课程里明确说，pipeline parallel 常常用在更慢的跨节点链路上。

大白话：

pipeline parallel 虽然笨一点，但特别适合“远距离接力”。

### 5）为什么它让人头疼

课程里很直白地说：
pipeline parallel 在工程实现上会非常复杂。
甚至讲了个八卦：有前沿实验室里只有极少数人真正懂自家 pipeline 系统，走一个人就变成“单点懂王”。

大白话：

你看原理图很简单，
真做起来像地铁调度系统。
只要时序、队列、反向调度哪里错一点，
整条线都乱。

### 6）更高级：zero-bubble / dual-pipe

课程里还提到一种很聪明的做法：
把 backward 里“必须立刻往前传的部分”和“算参数梯度但不急着传的部分”拆开。
然后把后者尽量塞进原本 idle 的 bubble 里。

大白话：

有些活是“必须现在做，不然下家接不上”；
有些活是“晚一点做也行”。
那就把“晚一点做也行”的活塞进空档里，
把闲着的时间利用起来。

# 十五、Tensor Parallel（张量并行）

这是另一条非常重要的路，而且在大模型里特别常见。

### 1）核心思想

大模型里很多计算都是矩阵乘法。
那与其按“层”切，不如按“矩阵的宽度”切。
也就是把一个大矩阵拆成几块，让不同 GPU 同时算，然后再合并。

大白话：

不是把做饭分成“切菜站、炒菜站”，
而是把一大块肉切成几份，
几个人同时剁，最后再拼起来。

### 2）它在前向里怎么干

课程里讲得很具体：
比如 Y = X @ A，把 A 切成 A1, A2，
每张 GPU 都拿到 X，各自算 X@A1、X@A2，
然后再通过同步把结果拼起来。
在 MLP 这种结构里，这样的切法可以自然嵌进去。

大白话

输入大家都看一眼，
但参数每人只管一块。
最后再把局部答案合成总答案。

### 3）它的好处

不需要靠更大的 batch 来填 bubble

没有 pipeline 那种明显空转

只要你能找到大矩阵乘法，就容易套进去

大白话：

tensor parallel 比较像“多人同时抬同一张桌子”，
而不是“桌子从一个人传到另一个人”。

### 4）它的坏处：特别吃高速互连

课程里非常强调：
tensor parallel 的通信频率高，而且常常每层都要同步。
所以它非常依赖低延迟、高带宽连接。
经验法则是：

tensor parallel 通常优先用在单机内的多张 GPU 上。
比如一台 8-GPU 机器内。

大白话

tensor parallel 像多人一起托举一张很大的木板，
你们必须靠得很近、配合很快。
如果人都分散在不同楼层，那就别这么干了。

### 5）为什么大家常说 TP 到 8 张卡很常见

课程里给出的经验是：
tensor parallel 在单机内到 8 张卡常常比较合适；
再往外扩到更慢互连时，吞吐下降会明显加重。

所以行业里常见模式就是：

机内做 tensor parallel，跨机再叠别的并行。

# 十六、Sequence Parallel / Activation Sharding（序列并行 / 激活切分）

这一段很关键，因为很多人前面学完会以为：

“参数、梯度、优化器都分了，问题差不多解决了吧？”

课程说：还没。
activation memory 还是会继续长。

### 1）为什么 activation 这么烦

训练时前向要存很多激活，
反向时再逐步释放。
所以显存峰值往往出现在：
激活还没完全释放，梯度又开始累积的时候。

大白话：

你一边在房间里堆前向留下的东西，
一边又开始生成反向要用的新东西，
最挤的时候不是最开始，也不是最后，
而是中间那段“旧东西没清完，新东西又进来”的时刻。

### 2）为什么 tensor parallel 也没完全搞定 activation

课程里说得很清楚：
大矩阵乘法相关的 activation 好切，
但像 layer norm、dropout、一些 pointwise 操作的输入输出，
并不会因为 tensor parallel 自动被均匀切掉。
所以会残留一些“straggler term”。

大白话：

大件家具你已经拆开分给几个人搬了，
但还有很多小零碎没拆，
最后这些零碎也会把房间塞满。

### 3）sequence parallel 的核心办法

既然 layer norm、dropout 这类操作在不同 token / sequence position 之间常常互不依赖，
那就沿着序列维度把它们切开。
比如 1024 长度的序列，分成几段，每张 GPU 处理其中一段。

大白话

不是按“参数块”切，
而是按“句子的位置”切。
甲管前 256 个 token，乙管中间 256 个，丙管后面 256 个。

### 4）为什么还要同步

切完之后，有些后续层还是需要更完整的视图，
所以前向里会有 all-gather / reduce-scatter，
反向里则做相反方向的同步。

大白话

大家先各做各的一小段，
但到了要拼成完整句子的时候，还是得对一下答案。

### 5）它和 FlashAttention / recomputation 的关系

课程里把 activation memory 讲成两部分：

一部分是和 MLP / pointwise 操作有关

一部分是 attention 里更贵的那块，尤其和 S^2 有关

如果你用 FlashAttention / activation recomputation，
就能把 attention 那块 activation 压下去很多。
再叠加 sequence parallel，整体 activation memory 才会更像你能承受的样子。

大白话：

activation 节省不是靠一招，
而是几招一起上：

大矩阵部分靠 tensor parallel

pointwise 部分靠 sequence parallel

attention 最贵那块靠 FlashAttention / 重计算

回顾：什么是flash attention？Flash Attention improves attention efficiency by reducing memory IO. 不存整个超级大attention matrix。 It 分块计算 computes attention in a tiled and （融合步骤把QKᵀ，softmax，×V合在一起一次做完）
 fused manner, avoiding materializing the full attention matrix and keeping computation within fast GPU memory.

# 十七、这一讲最后顺带提到的两种并行

课程后面还顺带提了两类，没深讲，但你要知道名字：

### 1）Context Parallel / Ring Attention

把长上下文 attention 的计算再沿序列 / KV 流动方式拆开。
每台机器负责一部分 query，keys/values 以 ring 方式传递。
这和你前面学的 FlashAttention 分块思路是连着的。

### 2）Expert Parallel

MoE 里把不同 expert 分散到不同机器。
概念上有点像 tensor parallel，但多了 routing 不均衡的问题，因为不同 expert 负载可能不一样。

大白话：

context parallel：把“长句子的注意力”也拆着算

expert parallel：把“不同专家老师”分到不同办公室

# 十八、这一章最值钱的实战结论：怎么组合这些并行

课程最后给了一个非常实用的 rule of thumb。
我把它翻译成大白话给你。

### 第一步：先让模型能放得下

如果模型根本放不下，别谈别的。
优先用能省显存的手段，让模型先 fit in memory：

tensor parallel 先用到单机内合理上限

跨机后再考虑 FSDP / pipeline parallel

activation 太大再上 sequence parallel / recomputation

### 第二步：放得下之后，再追求更多总算力

当模型已经能跑起来了，剩下 GPU 想继续加速，
这时通常继续往外扩的是 data parallel。
因为 data parallel 对慢网络更友好，也更通用。

### 第三步：batch size 不够时，用 gradient accumulation

如果通信太频繁、全局 batch 又受限，
就用 gradient accumulation 等办法，
把 batch 这个资源“攒起来再同步”。

大白话总规则

先解决“能不能装下”，再解决“能不能更快”。
别反过来。

# 十九、你可以把这章记成一句超级口语化总结

数据并行是“每人拿一整本教材，批不同作业”；
ZeRO/FSDP 是“教材和账本别全员复制，谁负责哪页谁拿哪页”；
pipeline parallel 是“按流水线分层传激活”；
tensor parallel 是“同一层的大矩阵大家一起抬”；
sequence parallel 是“连那些零碎的激活也按 token 切开”。

# 二十、学习笔记版
CS336 Lecture 7: Parallelism 1 学习笔记
1. 本章目标

理解为什么训练大模型必须用多 GPU / 多机并行，并学会区分不同并行方式各自解决什么问题：算力扩展、显存扩展、通信代价。

2. 网络与通信基础

单机内 GPU 互连快，跨机器慢

并行策略必须考虑硬件层级

关键 collective：all-reduce、all-gather、reduce-scatter

重要等价：all-reduce ≈ reduce-scatter + all-gather

3. Data Parallel

每张 GPU 复制完整模型

batch 按样本切分

各自算梯度，再同步

优点：算力扩展直接

缺点：参数、梯度、优化器状态都复制，显存浪费大

4. ZeRO / FSDP

Stage 1：分片 optimizer states

Stage 2：再分片 gradients

Stage 3：连 parameters 也分片

FSDP ≈ ZeRO Stage 3

关键思想：边通信边计算，减少显存峰值

5. Batch size 是资源

数据并行最多扩到 batch size 的量级

batch 太大也会收益递减

batch 可以“花”在数据并行，也可以“花”在 pipeline parallel 的 microbatch 上

6. Pipeline Parallel

按层切模型

GPU 间传 activations

naive pipeline 会有大 bubble

microbatch 可降低 bubble

优点：省模型和激活显存、点对点通信适合慢链路

缺点：工程实现复杂，调度难

7. Tensor Parallel

按矩阵宽度切模型

每层的大矩阵乘法分摊到多 GPU

每层通常有同步开销

优点：没有 pipeline bubble，不吃 batch size

缺点：非常吃高速互连

经验法则：常用于单机内多卡

8. Sequence Parallel / Activation Sharding

tensor parallel 不能自动消灭所有 activation memory

layer norm / dropout / pointwise 输入输出等可按 sequence 切

前向和反向需要 gather / scatter 配合

常与 recomputation / FlashAttention 一起用来压 activation memory

9. 其他并行

context parallel / ring attention：面向长上下文 attention

expert parallel：MoE 的专家分布式放置与路由

10. 实战 rule of thumb

先想办法 fit in memory

机内优先 tensor parallel

跨机再考虑 FSDP 或 pipeline parallel

模型放得下后，再用 data parallel 扩总吞吐

batch 不够时，用 gradient accumulation 提升通信效率

二十一、你真正要背下来的 8 句话

多机训练的核心矛盾是：显存、带宽、batch size 三者互相拉扯。

DDP 好懂，但复制太多，显存贵。

ZeRO 的本质是：别把参数相关状态傻乎乎复制到每张卡。

FSDP 是把参数也分掉，再按层临时 gather。

pipeline parallel 按层切，优点是省显存，缺点是 bubble 和工程复杂。

tensor parallel 按矩阵切，优点是没有 pipeline bubble，缺点是很吃高速互连。

activation 也会爆显存，所以还要 sequence parallel 和 recomputation。

实际训练通常不是单用一种，而是 3D/4D 组合并行。
