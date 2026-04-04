# 这一章到底在讲什么

你可以把第 8 章理解成一句话：

一张 GPU 里的问题，很多时候是“显存和带宽”；多张 GPU 之后，又多了一个更大的问题：GPU 和 GPU 之间怎么沟通。 讲义开头就把这个统一主题说得很清楚：不管是单 GPU 还是多 GPU，本质都是“计算离数据很远”，所以要想办法编排计算，减少数据传输瓶颈；上周是减少 GPU 内部 memory access，这周是减少 GPU / 节点之间的 communication。

大白话版：

第 5、6 章是在研究“厨房里怎么炒菜更快”；第 8 章是在研究“有很多厨房时，菜怎么在厨房之间传，才不会堵死”。

这讲最重要的一句话，其实讲义第一页就已经暗示了：

多 GPU 训练的核心，不是“多几张卡一起算”这么简单，而是“怎么算的同时，尽量少传数据，或者把必须传的数据传得更聪明”。 讲义把多层级硬件列成了一个 hierarchy：单 GPU 里有 L1 / shared memory、HBM；多 GPU 里有 NVLink；跨节点还有更慢的互连。整个并行策略就是在这个层级上做权衡。

大白话：

你可以把它想成搬家：

房间里挪东西，快

同一套房子里搬东西，稍慢

跨楼搬东西，最麻烦

所以你做并行，不是只想“找更多人干活”，而是要想：

这些人之间递东西贵不贵。

# 第一部分：distributed 的“积木”是什么

第 8 章先不急着讲训练，而是先讲 collective operations。讲义明确说，这些是分布式编程的“概念原语”，是并行编程里很经典的一套抽象，比你自己手写点对点通信更清楚也更高效。它还先定义了两个词：<u> world size 是设备总数，rank 是某一个设备的编号。</u> 

1）为什么先学这些

因为多 GPU 训练，表面上看是在“同步梯度”或者“传激活”。
但底层真正发生的，其实就是几种固定模式的通信：

broadcast

scatter

gather

reduce

all-gather

reduce-scatter

all-reduce

大白话：

这些就像物流公司的几种固定路线：

一个人发给所有人

一堆货拆开分给大家

大家把货交给一个人

大家把结果汇总

大家互相拼完整

先汇总再拆分

先汇总再所有人都拿一份

你后面看到 DDP、TP、PP，本质都只是这些路线的组合。

四、这些 collective operation 用大白话怎么理解
1）Broadcast

一个 rank 有数据，发给所有 rank。讲义直接列了 broadcast，并配了 PyTorch 教程图。

大白话：

班长把通知发到每个人手里。

2）Scatter

一个完整数据块，被拆成几份，发给不同 rank。讲义把 scatter 和 gather 放在一起讲，还特别提醒：broadcast/scatter 和 gather 互为逆操作。

大白话：

把一大箱苹果拆成四袋，四个人一人拎一袋。

3）Gather

每个 rank 各拿一小块，集中到一个地方。

大白话：

大家把手里的拼图交给一个人，拼成整张图。

4）Reduce

大家各有一份值，做一个可交换可结合的操作，比如 sum、min、max，最后汇总。讲义专门提醒 reduce 的“reduce”指的是做这种结合操作。

大白话：

四个人各报一个数字，最后把它们加起来。

5）All-gather

每个 rank 各有一小块，最后 所有 rank 都得到拼好的完整结果。讲义列了 all-gather 的图，也在后面的 tensor parallel 里真的用到它。

大白话：

每个人手里只有自己那块拼图，最后每个人都拿到整张图。

6）Reduce-scatter

先 reduce，再 scatter。也就是大家先汇总，但不是每个人拿完整汇总结果，而是每个人只拿其中一块。讲义既列了 reduce-scatter 图，也在代码里做了示例。

大白话：

大家一起做总账，但最后甲只拿工资表那页，乙只拿报销表那页。

7）All-reduce

先 reduce，再让所有人都拿到结果。讲义非常明确地写了一句：

all-reduce = reduce-scatter + all-gather。
后面的代码也专门演示了这一点。

大白话：

先把总分算出来，然后每个人都拿一份总分。

这句话很重要，因为后面你理解 FSDP、ZeRO、梯度同步时，脑子里基本都离不开它。

五、第二部分：硬件到底长什么样，为什么它会决定并行策略

讲义接着讲硬件。它先拿家用环境做对比：同一台机器里的 GPU 可能走 PCIe，总线带宽比跨机器网络高很多；现代数据中心则是 GPU 之间用 NVLink 直接连，跨更多 GPU / 节点还有更复杂互连。讲义还给出一个对比：每块 H100 的 NVLink 总带宽大约 900 GB/s，而 HBM 内存带宽更高，大约 3.9 TB/s。

大白话：

GPU 之间虽然已经很快了，但还是没有 GPU 自己访问显存那么快。
这意味着：

你只要一跨 GPU，基本就得更节省地传东西。

为什么这一点这么关键

因为你最后选哪种并行方式，很大程度上取决于：

你的 GPU 是不是在同一台机器里

它们之间是不是有高速互连

你传的是不是很大的 tensor

你是每层都要传，还是每 step 才传一次

大白话：

近距离频繁说话可以，远距离频繁说话就容易拖慢全队。

六、NCCL 和 torch.distributed 在干嘛

讲义接着介绍 NCCL 和 PyTorch distributed。
它说得很直接：

NCCL 负责把 collective operation 翻译成真正的 GPU 间通信，检测硬件拓扑、找更好的传输路径，并且用 CUDA kernels 去执行发送/接收。

torch.distributed 则是在 PyTorch 里提供一个比较干净的接口，比如 all_gather_into_tensor、all_reduce，GPU 上常用的 backend 是 nccl，CPU 上常用 gloo。

大白话：

NCCL 像物流调度系统

torch.distributed 像你写代码时用的快递下单接口

你平时写：

“帮我 all-reduce 一下这个梯度。”

真正底层发生的是：

“系统先看这些 GPU 怎么连着，再决定怎么发包、走哪条路、谁先传给谁。”

七、讲义代码为什么要先 spawn 多个进程

讲义里用 spawn(...) 跑示例，每个进程对应一个 rank，然后在每个 rank 上初始化 process group，再调用 collective operations。它还专门用 dist.barrier() 让所有进程在某个点对齐。

大白话：

你不能一边三个人已经开始开会，第四个人还没进会议室。
所以 barrier 的意思就是：

大家先等等，齐了再往下走。

八、为什么这一讲要做 benchmark

讲义专门有一个 benchmarking()，对 all-reduce 和 reduce-scatter 测带宽。它做了 warmup，然后同步 CUDA，再测真实时间，最后按传输字节数估算有效带宽。它还引用了 NCCL performance 文档来说明如何理解这些操作的代价。

1）为什么要测这个

因为“通信慢不慢”不能靠猜。
你得知道：

一次 all-reduce 到底花多少时间

reduce-scatter 和 all-reduce 差多少

在你现在这台机器上，通信已经占了多少比例

大白话：

这就像你不能只说“这条路感觉堵”，
你得真去看：

从 A 到 B 到底几分钟。

2）为什么这一步很有现实意义

因为你后面看到的 data parallel、tensor parallel、pipeline parallel，本质差别很大一部分就在于：

谁传得更频繁，谁传的数据更大。
所以先知道“传一次大概有多贵”，后面很多判断就不再是拍脑袋。

九、第二大部分：distributed training 真正开始了

讲义在第二部分写得很明确：
它会用深层 MLP 的简化例子，分别演示三种分布式训练策略：

data parallelism：沿 batch 维切

tensor parallelism：沿 width 维切

pipeline parallelism：沿 depth 维切

这一句特别重要，因为它其实给了你一个很好的记忆法：

data parallel = 切数据

tensor parallel = 切每层宽度

pipeline parallel = 切层数 / 深度

大白话：

同一个模型，你可以从三种不同角度拆：

作业分给不同人批

一道大题拆成几个人一起做

整个流程拆成流水线几站

十、Data Parallelism（数据并行）到底是什么

讲义先给出 data parallelism 的图，然后在代码里做了一个很朴素的版本：

全部数据 batch 被均匀切给不同 rank

每个 rank 都有完整的一套参数

每个 rank 在自己那份数据上前向、反向

然后对每个参数的梯度做 all_reduce(..., AVG)

再各自 optimizer.step()

1）它的本质

每张卡模型都一样，数据不一样。

大白话：

4 个老师拿着同一本教材，
但每个人批改不同学生的作业。
批完以后，4 个人把意见平均一下，
于是下一轮他们手里的教材还是一致的。

2）讲义特别强调的三个点

讲义在代码后面的 notes 里写了三句很关键的话：

各个 rank 的 loss 不一样，因为是各自本地数据算出来的

梯度被 all-reduce 后会变得一样

所以参数也会保持一样

这三句其实就是 DDP 的灵魂。

3）为什么它好懂

因为它对原来单卡训练改动最小。
除了“把 batch 切了”和“反向后同步梯度”，逻辑基本没变。

大白话：

它像是把“一个老师批 128 份作业”改成“4 个老师各批 32 份，最后统一评分标准”。

4）它的代价是什么

这一讲的代码没有展开 optimizer state 和显存冗余的细节，但它已经明显呈现出一个事实：每个 rank 都有完整参数，也各自有 optimizer。 这意味着数据并行最直接的问题就是复制多份模型状态。

大白话：

好懂归好懂，但代价就是：

每个人都得背一整本书。

十一、Tensor Parallelism（张量并行）到底是什么

讲义第二个示例是 tensor parallelism。它的 sharding strategy 写得非常直接：

每个 rank 拿每一层参数的一部分，数据 / 激活要在 rank 之间传。
代码里，它把特征维度 num_dim 按 world size 切开，每个 rank 拿到 num_dim / world_size 那一部分参数；前向时先算自己这份局部输出，再通过 all_gather 把所有 rank 的输出拼回来，恢复成完整激活。

1）它的本质

模型不再完整复制给每个 GPU，而是每层被横着切开。

大白话：

不是 4 个老师各拿一本完整教材，
而是这本教材太宽了，
甲只拿第 1 章的一半，乙拿另一半，
每一题都得大家一起做完再拼起来。

2）代码里真正发生了什么

讲义代码逻辑是：

所有 rank 都看到完整输入 x

每个 rank 用自己那部分参数，算出局部激活

通过 dist.all_gather(...) 收集所有 rank 的局部激活

torch.cat(..., dim=1) 拼成完整激活，供下一层继续算

3）为什么这很重要

因为它说明 tensor parallel 的关键代价不是“最后才同步一次”，而是：

每一层几乎都要有同步 / 拼接。

大白话：

这像一道特别大的题，4 个人每一步都各做一小块，
但每做完一步都要把答案拼起来，
然后才能继续下一步。

4）它的优点是什么

因为参数被切开了，所以单个 rank 上放的参数更少。
也就是说，它能帮你把更宽、更大的层塞进多张卡里。

5）它的缺点是什么

通信频繁。
讲义虽然没有在这一页直接写“每层都很贵”，但代码已经非常清楚地展示了：每层 forward 都要 all-gather activations。

大白话：

tensor parallel 像多人抬同一张大桌子。
桌子是抬起来了，
但你们必须一直对节奏，
所以特别怕人和人之间离太远、沟通太慢。

十二、Pipeline Parallelism（流水线并行）到底是什么

讲义第三个示例是 pipeline parallelism。它的 sharding strategy 写得也很清楚：

每个 rank 拿一部分层，数据 / 激活在 rank 之间传。
代码里 world size=2、num_layers=4，于是每个 rank 负责 2 层。输入 batch 会再切成多个 micro-batches，依次流过各个 rank。rank 之间不是用 all-gather，而是 dist.send / dist.recv 点对点传输激活。

1）它的本质

模型按深度切。
前几层在 GPU0，后几层在 GPU1，数据像流水线一样往后流。

大白话：

工厂有两站：

第一站做前处理

第二站做后处理

第一站处理完一小批，就把半成品送到第二站。

2）为什么要切 micro-batches

讲义代码里专门把 batch 切成 num_micro_batches 份，并写明原因：

Break up into micro batches to minimize the bubble。
也就是为了减少流水线空泡。

大白话：

如果你一次只送一大坨货进去，
前站干活时后站在等，
后站干活时前站又在等。
所以最好把大批货切成很多小批，
前一小批刚离开第一站，第二小批立刻补上。

3）代码里真正发生了什么

rank 0 拿到真实数据并切成 micro-batches

rank 1 先分配好空 tensor，等着接收来自 rank 0 的激活

每个 micro-batch 到来后，本 rank 只算自己负责的层

算完就 send 给下一个 rank

4）讲义明确说了什么还没处理

讲义在 pipeline parallelism 后面专门写了一句：

Not handled: overlapping communication/computation to eliminate pipeline bubbles。
也就是说，这个例子只是一个最基础版本，还没有做“通信和计算重叠”来进一步减少空泡。

大白话：

现在这个演示版已经像流水线了，
但还不是最先进的工厂。
真正高级的工厂会做到：

你还在这一站加工时，下一站的准备工作已经在悄悄开始。

5）它的优点是什么

参数是按层切开的，所以每张卡只保存部分层。
相比 data parallel，它不用每张卡都放完整模型。

6）它的缺点是什么

有 bubble，而且调度复杂。
就连这份讲义也只给了 forward，backward 直接写成 homework exercise。

大白话：

pipeline parallel 省模型显存，但容易出现“有的站在忙，有的站在等”。

十三、第 8 章真正想让你看懂的，不只是三种并行名字

这一讲最值钱的地方，不是记住名词，而是看清楚：

data parallel 在同步什么
→ 梯度 all_reduce。

tensor parallel 在同步什么
→ 每层的局部激活 all_gather 后拼完整。

pipeline parallel 在同步什么
→ 相邻 stage 之间点对点 send/recv 激活。

大白话：

这三种并行的真正区别，不只是“切哪里”，而是：

它们把什么东西递给别人。

十四、把三种并行放在一起比较，你就更容易懂了
1）Data Parallel

切的是 batch。
每张卡都有完整模型。
主要通信是 梯度 all-reduce。

大白话：
每个人拿全套教材，各批各的卷子，最后统一分数。

2）Tensor Parallel

切的是 层的宽度 / 参数矩阵。
每张卡只有部分参数。
主要通信是 每层 all-gather 激活。

大白话：
每道题大家一起做，每一步都要拼答案。

3）Pipeline Parallel

切的是 层的深度。
每张卡负责几层。
主要通信是 stage 之间传激活。

大白话：
不同人负责不同工序，半成品一站一站往后传。

十五、为什么讲义说“this week: parallelism across multiple GPUs”

因为这讲本质上是在把第 5、6、7 讲的思想落实到代码层面。官方课程表显示：第 7 讲也是 Parallelism，第 8 讲继续 Parallelism，而第 8 讲的可执行讲义明确写着：“上周：single GPU 内并行；这周：across multiple GPUs。” 它其实就是把“分布式训练”从概念变成了代码化、原语化的演示。

大白话：

第 7 讲更像“原理课”。
第 8 讲更像“上机课”。
它在用最小代码告诉你：

多 GPU 并行最后真的就是这些 primitive 拼出来的。

十六、这一讲最后的总结，其实非常值得背

讲义最后 summary 里直接写了几句特别重要的话：

并行有很多种：data（batch）、tensor/expert（width）、pipeline（depth）、sequence（length）

可以选择 re-compute、存在本地 memory、或者存在其他 GPU 的 memory 然后 communicate

硬件会越来越快，但模型也会越来越大，所以这种层级式结构会一直存在

大白话翻译一下就是：

以后硬件再升级，也不可能让“通信问题”消失。
因为你总会想训练更大的模型、塞更长的上下文、用更多卡。
所以“怎么拆、怎么传、怎么少传”会一直是大模型系统里的核心功课。

十七、给你的学习笔记版
CS336 第 8 章：Parallelism（Lecture 8）学习笔记
1. 本章目标

从“单 GPU 内部优化”转向“多 GPU / 多节点之间的并行”。核心目标是减少跨 GPU / 节点通信瓶颈。讲义分为两部分：分布式通信原语与分布式训练策略。

2. 核心总思想

单 GPU 优化：减少 memory access

多 GPU 优化：减少 communication

硬件是分层的：shared memory / HBM / NVLink / 更远互连

并行设计本质是沿着这个层级做权衡

3. 分布式通信原语

world size：设备总数

rank：某一个设备编号

常见 collective：

broadcast

scatter

gather

reduce

all-gather

reduce-scatter

all-reduce

重要关系：all-reduce = reduce-scatter + all-gather

4. NCCL 与 torch.distributed

NCCL 负责把 collective operation 变成实际的 GPU 间通信，并根据硬件拓扑优化路径

torch.distributed 提供高层接口；GPU 常用 backend 是 nccl，CPU 常用 gloo

5. 为什么要 benchmark 通信

不能靠感觉判断通信是否贵

讲义对 all-reduce 和 reduce-scatter 做了实际 benchmark

通过 warmup、同步 CUDA、统计字节数来估算有效带宽

6. Data Parallelism

按 batch 切数据

每个 rank 持有完整参数和本地 optimizer

本地 forward / backward 后，对梯度做 all_reduce(AVG)

同步后各 rank 参数保持一致

优点：概念简单，接近单卡训练

代价：模型状态会被复制到每个 rank 上

7. Tensor Parallelism

按 width / 特征维度 切每层参数

每个 rank 只持有局部参数

前向时先算局部激活，再 all_gather 拼成完整激活

优点：能把更宽的大层拆到多卡

代价：每层都要通信，通信频繁

8. Pipeline Parallelism

按 depth / 层数 切模型

每个 rank 负责一段层

使用 micro-batches 减少 pipeline bubble

rank 之间通过 send/recv 传激活

优点：不需要每卡放完整模型

代价：有 bubble，调度复杂；讲义示例还没处理通信与计算重叠

9. 三种并行的本质差别

data parallel：切数据，传梯度

tensor parallel：切宽度，传并拼激活

pipeline parallel：切深度，传半成品激活

10. 本章总哲学

多 GPU 训练不是简单“多卡一起算”，而是：
把模型或数据拆开，同时设计好通信模式，让“必须传的数据”尽量少、尽量高效。

十八、你最该背下来的 8 句话

第 8 章的主题是：从单 GPU 优化转到多 GPU / 多节点并行。

多 GPU 的瓶颈常常不是算力，而是通信。

collective operations 是分布式训练的基本积木。

all-reduce 可以理解成 reduce-scatter + all-gather。

data parallel 是切 batch，最后同步梯度。

tensor parallel 是切每层宽度，几乎层层都要拼激活。

pipeline parallel 是切层深度，用 micro-batches 减少 bubble。

并行策略的本质区别，在于“切哪里”和“传什么”。
