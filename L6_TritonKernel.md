# 一、这一章到底在干嘛

第 5 章更像是在讲：

GPU 为什么有时候快，有时候慢。

第 6 章升级成：

既然知道 GPU 的脾气了，那我要怎么真的写出更快的代码？

课程材料里这一讲的主线非常清楚：先复习 GPU 的执行模型，然后强调 benchmark / profile，再拿 GeLU 这个例子展示 手写普通 PyTorch → fused PyTorch → CUDA/C++ kernel → Triton kernel → torch.compile 的差别，最后再做一个更复杂一点的 softmax Triton kernel。 ￼

大白话说，这一章其实在回答一个很现实的问题：

“我写的数学明明没变，为什么代码速度能差 5 倍、10 倍？”
答案通常不是“你数学错了”，而是：

你让 GPU 搬了太多数据，或者你根本没搞清楚 GPU 到底在跑什么 kernel。  ￼

⸻

# 二、第一部分：GPU 复习，但这次是为了“写代码”

这部分不是重复第 5 章，而是把第 5 章里那些概念，重新变成“写 kernel 时必须记住的规则”。讲义里先复习了 SM、thread block、grid、shared memory、waves、arithmetic intensity。A100 的例子里也直接给了数量级：很多 SM，DRAM 很大但慢，L1/L2/register 更快更近。 ￼

1）SM 到底要怎么想

你可以把 SM 想成一个“车间”。

一个 GPU 不是一个大脑袋统一干活，而是很多很多小车间同时开工。每个车间里有计算单元，也有比较快的本地存储。一个 thread block 会被放到某一个 SM 上执行，而 block 里面的线程能共享这块比较快的本地资源。 ￼

大白话：

SM = 车间
block = 分给这个车间的一整包活
thread = 车间里具体干活的小工人

所以你写 kernel 时，不是在想“一个大 GPU 干一件事”，而是在想：

我要怎么把大任务切成很多包，让每个车间都能拿到一包合适的活。  ￼

2）为什么要有 thread block

讲义强调 thread block 很重要，因为 同一个 block 里的线程可以共享 shared memory，也可以同步；但跨 block 基本不能这么方便地合作。 ￼

大白话：

你可以把 block 理解成一个“项目小组”。
组内人能互相喊、互相传纸条、一起看桌上的资料。
组和组之间就没这么方便了。

所以如果一堆数据需要频繁共享，最好让它们落在 同一个 block 里。
这也是为什么矩阵乘法、softmax 这种操作常常会围绕“block 里怎么复用数据”来设计。 ￼

3）waves 是什么，为什么会偷偷拖慢你

课程转录里特别提醒了 waves：thread blocks 会一波一波地被发到 SM 上。理想情况是，每一波都尽量装满；否则最后一波如果 block 不够，部分 SM 就会闲着。讲义里给的经验法则是：thread blocks 数量最好至少是 SM 数量的 4 倍，还提到 wave quantization。 ￼

大白话：

你有 108 个车间，但最后一趟只送来 50 包活，那就意味着有 58 个车间在发呆。
GPU 没坏，代码也没错，但吞吐量已经悄悄掉了。

所以你写 kernel 时，不只是要“能跑”，还要想：

我切出来的 block 数量够不够多？最后一波会不会浪费很多 SM？  ￼

4）arithmetic intensity 为什么是这讲的灵魂之一

讲义里把 arithmetic intensity 定义为 FLOPs / bytes moved，也就是“每搬多少字节数据，做了多少计算”。并且直接给了一个很强的经验判断：矩阵乘法如果实现得好，往往是 compute-bound；除此之外很多东西都更容易 memory-bound。  ￼

大白话：

GPU 不是不会算，它通常是 搬东西太累。
如果你搬 10 箱货，只做 1 次加法，那肯定亏。
如果你搬 1 箱货，能在本地反复加工很多次，那才划算。

这就是为什么后面 kernel fusion、Triton softmax、甚至 torch.compile 都围着同一个核心打转：

能不能让同一批数据，少进出显存，多在片上做完。  ￼

⸻

# 三、第二部分：为什么老师反复说“先 benchmark，再优化”

这部分我觉得特别重要，因为它其实是在纠正很多人的直觉。

讲义里明说：如果你想写高性能代码，最该记住的一件事就是 benchmark 和 profile。 课程里甚至专门用一个 MLP 示例，做 benchmark 和 profile，说明“猜瓶颈”常常不靠谱。 ￼

1）benchmark 是看“总共用了多久”

讲义把 benchmark 说得很直接：
它测的是 wall-clock time，也就是“这件事从头到尾花了多久”；它适合做两件事：比较不同实现谁更快，以及看性能如何随规模变化。 ￼

大白话：

benchmark 像你看外卖总共多久送到。
你能知道 A 店比 B 店快，但你不知道慢在哪一步：是做饭慢，还是骑手慢，还是堵车。 ￼

2）为什么 GPU benchmark 很容易测错

讲义的 benchmark helper 明确做了两件事：
先 warmup，再 torch.cuda.synchronize()。原因也写得很清楚：第一次运行可能更慢，因为编译或缓存原因；而 CUDA 默认是异步的，不同步就可能只量到“发命令的时间”，没量到真正执行时间。 ￼

大白话：

GPU 很像你把任务甩给一个厨房，厨房说“收到”，但其实菜还没炒完。
如果你在“收到”那一刻就停表，你测到的是“下单成功时间”，不是“上菜时间”。

所以：

不 synchronize()，很多 GPU benchmark 都是在自欺欺人。  ￼

3）profile 是看“时间花在哪”

讲义说 profiling 不只是帮你看哪里慢，更深一层是让你知道 底下到底调用了什么。它用 PyTorch profiler 去看 add、matmul、cdist、gelu、softmax，然后展示：表面上看似简单的一行代码，底下可能对应完全不同的 CUDA kernels。 ￼

大白话：

benchmark 只告诉你“这顿饭慢”。
profile 会告诉你：
   •   前菜等了多久
   •   主菜等了多久
   •   哪个厨师最忙
   •   到底是哪道菜拖后腿

这就是为什么 profile 比 benchmark 更接近“理解系统内部”。 ￼

4）为什么 tensor shape 会影响底层 kernel

讲义里专门比较了 matmul(dim=2048) 和 matmul(dim=128)，并指出：不同 tensor 维度会调用不同 CUDA kernels；kernel 名里甚至会直接透露实现信息，比如 CUTLASS、tile size 等。 ￼

大白话：

你以为你只是把矩阵从 128 改成了 2048。
GPU 看来却是：“这不是一个更大的活而已，这可能是另一种施工方案。”

所以以后看到“同样一段代码，大 shape 很快、小 shape 很怪”，先别怀疑人生。
很多时候是底层 dispatch 的 kernel 已经不一样了。 ￼

⸻

# 四、第三部分：kernel fusion 为什么这么值钱

这部分讲义用了 Horace He 的那个经典类比：warehouse : DRAM :: factory : SRAM。课程代码也明确写了：每个 operation 都需要 read / compute / write；如果 fuse 起来，只需要读写一次。 ￼

1）先别急着想“算式”，先想“搬运”

GeLU 这个例子非常经典。讲义对比了两种写法：

第一种是手工展开 GeLU，用很多基础运算拼出来。
第二种是 PyTorch 默认的 fused 实现。 ￼

数学上它们等价。
但 profile 显示，手工那种会拆成很多小操作，而 fused 版本本质上更像“一口气做完”。讲义直接总结：PyTorch 那边基本就是一个 kernel，而手工版是很多 atomic pieces。  ￼

2）为什么 fused 版本更快

因为不 fused 的版本，每一步都会：
   •   从显存读中间值
   •   做一点点计算
   •   再把中间结果写回去

这就像半成品每做一步都送回仓库，下一道工序再去仓库取。
fused 版本则是把多道工序尽量放在同一个 kernel 里，半成品尽量留在片上寄存器或本地更快的地方。 ￼

大白话：

fuse 不是“更会算”，而是“不瞎折腾搬货”。

3）这一段你要学会的真正直觉

很多人第一次看会以为：

“既然公式一样，那性能应该差不多吧？”

这正是 GPU 编程最容易踩的坑。
在 GPU 世界里，“公式一样” 绝不代表 “执行代价一样”。
很多时候真正贵的是：中间结果一遍又一遍地进出 global memory。 ￼

⸻

# 五、第四部分：CUDA kernel 到底是什么，为什么写起来痛苦但很重要

讲义接下来“开盒”，自己写 CUDA/C++ 的 GeLU kernel。它把 CUDA 描述成 C/C++ 的扩展和 API，并强调一个简化心智模型：你写一个 f(i)，launch kernel 后，GPU 会并行地对很多元素去执行这个逻辑。 ￼

1）CUDA kernel 的心智模型

课程转录里说得很清楚：
你会有一个 grid，grid 里有很多 thread blocks，block 里有很多 threads。每个 thread 都会拿到类似：
   •   block index
   •   block dimensions
   •   thread index

然后根据这些坐标，算出“我该处理哪一个元素”。 ￼

大白话：

你写的不是“整个数组怎么处理”，而是：

“如果我是其中一个线程，我该干哪一小格？”

这个思维切换非常关键。

2）wrapper 和真正的 kernel 是两层东西

课程转录专门讲了这一点：
CUDA 代码一般有两层：

第一层是 CPU 端的 wrapper，负责检查输入、分配输出、计算 grid / block 参数、发起 kernel launch。
第二层才是真正跑在 GPU 上的 kernel，负责单个线程的逻辑。 ￼

大白话：

wrapper 像项目经理，做排班、分配材料、喊大家开工。
kernel 才是工人在现场真正干的活。 ￼

3）为什么要检查 contiguous

转录里提到，他们会检查输入是不是 CUDA tensor、是不是 contiguous；因为后面会做大量索引计算，默认假设数据在内存里是规规矩矩连续排好的。 ￼

大白话：

你以为拿的是一排整齐堆好的砖，结果砖头散落在院子各处。
那你原本设计好的“按编号去拿第 i 块砖”的方法就不成立了。

所以 contiguous 本质上是在保证：

你对地址的数学推导，真的对应物理内存的排布。  ￼

4）为什么要算 num_blocks = ceil(num_elements / block_size)

课程转录里也特别解释了 cdiv / 向上取整：因为最后一批元素可能不足一个 block，但你还是得有人处理它们。 ￼

大白话：

你有 1000 件货，每车装 256 件。
你不能说 1000 / 256 = 3 车多一点，于是只派 3 车。
你必须派 4 车，不然最后那点货没人管。 ￼

5）为什么老师让你开 CUDA_LAUNCH_BLOCKING=1

课程里明确说，debug CUDA 时要把这个打开，不然报错会很痛苦。 ￼

大白话：

CUDA 平时像一个“异步流水线”，你下发任务后它不一定立刻报错。
等你发现错的时候，现场已经过了好几步，你都不知道是哪里先崩的。

把它 blocking 之后，更像“做一步、验一步”。
调试会慢，但更容易定位问题。 ￼

6）为什么这个 CUDA GeLU 没赢过 PyTorch

课程代码和讲课里都说了：
他们自己写的 CUDA 版比手工 naive 版快，但还不如 PyTorch 的 fused 版本。 ￼

这特别有教育意义。

大白话：

“我都手写 CUDA 了，怎么还没赢？”
因为你写的是“能跑的 CUDA”，不一定是“顶级优化的 CUDA”。
PyTorch 底层很多 kernel 已经被专业团队优化很久了。
所以你手写 CUDA 的意义，不只是为了今天赢 PyTorch，而是为了：

理解 GPU 真正在做什么，知道什么时候值得自己下场。  ￼

⸻

# 六、第五部分：Triton 到底厉害在哪

讲义里对 Triton 的介绍非常直接：OpenAI 2021 推出；目标是让 GPU 编程更 accessible；核心点是 用 Python 写，并且更偏 block-level thinking，而不是 thread-level thinking。它还列了一个很关键的对比：memory coalescing、shared memory management、SM 内调度这些，Triton 编译器会替你多做很多。 ￼

1）Triton 的真正卖点不是“Python”，而是“抽象层次变高了”

很多人第一反应是：
“哦，Triton 就是 Python 写 kernel。”

这只说对一半。更重要的是：

CUDA 让你更直接面对 thread；Triton 让你更直接面对 block。  ￼

大白话：

CUDA 更像你亲自安排每个工人站哪、拿哪把螺丝刀。
Triton 更像你说：“这一组人负责这一块区域，具体组内怎么更高效地展开，编译器帮我做很多事。”

2）Triton wrapper 和 CUDA wrapper 很像

转录里明确说，Triton GeLU 的外层 wrapper 跟前面 CUDA 的 wrapper 结构几乎一样：
   •   检查输入
   •   分配输出
   •   算 block / num_blocks
   •   launch kernel ￼

大白话：

变的不是“整个思路”，而是“你往下写的时候不用那么细抠线程级细节”。
所以 Triton 是一种很好的桥梁：
它不会让你完全失去硬件感，但也不会像裸写 CUDA 那么累。 ￼

3）Triton kernel 里面最关键的几行怎么理解

课程转录里把这部分讲得很清楚：
   •   program/block 先确定“我是哪一块”
   •   再算当前 block 的起点
   •   再生成一串 offsets
   •   offsets 在 Triton 里往往是一个向量，而不是单个 thread index
   •   超出边界时用 mask 处理 ￼

大白话你可以这么记：

第一步：先找到我这一包活从哪开始

比如我负责第 3 块，每块 1024 个元素，那我的起点就是 3 * 1024。 ￼

第二步：再把这一包里的位置展开

不是只看“第几个 thread”，而是直接把这一整块的 offsets 向量拿出来。 ￼

第三步：load、算、store

把这一块读进来，做向量化计算，再写回去。 ￼

你会发现这非常像 NumPy 式思维。
所以很多人第一次上手 Triton 会觉得：

“这终于像是在写我看得懂的 GPU 代码了。”

4）为什么 Triton 里 mask 这么常见

因为最后一个 block 很可能“不满”，也就是 offsets 会有一部分越界。转录里明确说了：需要 mask 去处理落在数组边界外的那些位置。 ￼

大白话：

最后一车货可能只剩 100 件，不是满车 256 件。
你不能让另外 156 个“位置”瞎读内存。
mask 就是在说：

这些位置虽然形式上存在，但别真去碰无效地址。  ￼

5）为什么 Triton 很适合做“块内向量化”

课程里还提到一个很关键的观察：在这个 GeLU 例子里，Triton 的心智模型里 offsets 是一个向量，所以它很自然地支持编译器做诸如 thread coarsening 之类的优化。 ￼

大白话：

CUDA 更像你手动安排一个个工人。
Triton 更像你拿着一整排工人一起发指令。
这让编译器更容易做整体优化。 ￼

⸻

# 七、第六部分：PTX 这段到底想让你学什么

课程里专门提到会往下看到 PTX，也就是更接近机器的低层表示。讲课转录里说得很直白：看 PTX 是为了理解 GPU 真正在做什么。 ￼

1）为什么要看 PTX

不是为了让你以后天天手写 PTX。
而是为了让你明白：

高级代码最后会被翻译成非常具体的 load / compute / store 指令，寄存器数量、访存方式、每个 thread 一次处理几个值，这些都是真实存在的。  ￼

2）讲课里从 PTX 里看到了什么

转录里提到，Triton 编译出来的 PTX 里可以看到：
   •   先准备一些临时存储
   •   再做具体的 floating-point 运算
   •   最后把寄存器中的结果写回输出地址
而且能看出 每个线程一次处理四个值，临时结果存在寄存器里。 ￼

大白话：

你平时写 gelu(x) 很优雅。
但到 PTX 这层，它其实变成：

“把这几个值读进寄存器 → 做加减乘除和 tanh 相关计算 → 把结果写回内存。”

这能帮助你建立一种非常重要的工程感：

GPU 优化最终都会落回到：读了几次、算了几步、写了几次。  ￼

⸻

# 八、第七部分：torch.compile 为什么值得你认真看待

这部分我觉得非常实用，因为很多人学完 CUDA / Triton 后会陷入一种错觉：
“以后我要自己手写一切 kernel。”
课程其实是在故意帮你打破这个错觉。 ￼

1）torch.compile 在干什么

转录里说得很明确：
它会拿普通、没优化过的 PyTorch 代码，自动尝试做更优化的版本，尤其包括 kernel fusion。并且在这个 GeLU 例子里，它的性能甚至比课程里手写的 Triton / CUDA 示例还更好一些。 ￼

2）为什么它能这么强

因为很多简单模式，其实编译器已经很会了：
   •   哪些操作可以 fuse
   •   某些已知 shape 用什么 kernel 更合适
   •   怎么自动生成 Triton 风格的优化代码 ￼

大白话：

以前你可能要亲自搬砖。
现在很多“标准户型”的房子，施工队已经有成熟流水线了。
你非要自己一砖一瓦砌，不一定更快，还可能更差。 ￼

3）那什么时候还值得自己写 Triton / CUDA

课程给的判断很成熟：

对于简单 operator fusion 或常见 matmul 调度，torch.compile 已经很强。
但像 FlashAttention 这种带有更复杂 IO 设计、甚至跟特定硬件特性深度绑定的优化，就不是普通 JIT 一下子总能自动想到的。 ￼

大白话：

如果问题是“常规优化”，先信编译器。
如果问题是“新结构、新算子、特殊硬件、特别卡脖子的热点”，那才值得自己下场。 ￼

4）这一段最重要的思想

你不应该学完这一章就得出结论：

“我要写 CUDA kernels for everything.”

课程转录里几乎是反过来在提醒你：不要这样。
更好的结论是：

先 benchmark / profile，能靠现成库和编译器解决就先别手写；只有当你知道它不够好，且你知道热点在哪，才考虑手写 Triton / CUDA。  ￼

⸻

# 九、第八部分：softmax 为什么是这一章最后的“升级题”

前面 GeLU 主要是 elementwise，比较简单。
softmax 不一样，因为它涉及 一行内部的 reduction：要先找 max，再 exp，再 sum，再除。课程转录里明确说，到这里就不再是简单 elementwise 了。 ￼

1）为什么 softmax 比 GeLU 难

GeLU 的每个元素基本只看自己。
softmax 的一个元素，归一化时要依赖同一行的其他元素。 ￼

大白话：

GeLU 像每个人填自己的表格。
softmax 像全班成绩要先求全班总分、最高分，再回头算每个人比例。
它天然需要“组内合作”。 ￼

2）课程里给的 naive 但合理设计：一行一个 block

转录里老师直接给了设计直觉：
如果矩阵的每一行能放进一个 SM 的本地工作范围里，那最简单的设计就是：

每个 block 负责一整行。
grid 的大小就是行数。 ￼

大白话：

因为 softmax 是“按行归一化”，那最顺手的做法就是：

一整个小组，负责一整行。

这样这一行的 max、sum、normalize 都能在“组内”完成，不用行和行之间乱沟通。 ￼

3）这个 softmax kernel 实际做了什么

转录里概括得很直接：
   •   找到当前 row
   •   生成 column offsets
   •   把这一整行 load 进来
   •   row - max(row)
   •   exp
   •   sum
   •   除掉分母
   •   写回 global memory ￼

大白话：

就是把“这一行的数据”先拉到车间里，
然后在车间里把这一整套 softmax 流程一次做完，
最后只把最终结果送出去。 ￼

4）为什么这种写法比 naive PyTorch 拼算子强很多

课程转录里说 naive softmax “kind of a disaster”，因为你会看到一堆 max、sum、exp 等操作零零碎碎发生，到处是 memory reads/writes。相反，compiled / PyTorch / Triton 的版本都更接近“一个 fused kernel”。 ￼

大白话：

naive 写法像：
   •   先把一行送去做 max
   •   再送回来
   •   再送去做 exp
   •   再送回来
   •   再送去做 sum
   •   再送回来
   •   最后再送去除法

你光在路上就累死了。
fused softmax 则像：

整行拿进车间，一次加工完。  ￼

5）这一段真正想教你的不是 softmax 本身

不是为了让你死记 softmax 代码。
而是让你学会一种 kernel 设计题的思路：

先问自己：
   •   reduction 维度是什么
   •   哪些值必须一起看
   •   哪一组线程 / 一个 block 最自然地覆盖这块数据
   •   能不能在 block 内部就把中间步骤做完 ￼

这套思路往后看 RMSNorm、LayerNorm、attention、FlashAttention 都很有用。 ￼

⸻

# 十、把整章串起来：你现在应该形成的“脑内地图”

如果我把这一章压缩成一个脑内流程图，它其实是这样：

1）先知道 GPU 的脾气

block、SM、shared memory、waves、arithmetic intensity。 ￼

2）先量，再猜

benchmark 看总时间，profile 看热点和底层 kernel。 ￼

3）发现很多慢，不是因为算得慢，而是因为拆得太碎、搬得太多

这就是 kernel fusion 的动机。 ￼

4）如果要手写 kernel，先学 CUDA 心智模型

grid / block / thread，wrapper vs kernel，本质是“每个线程负责哪一块”。 ￼

5）如果不想太底层，就用 Triton

仍然保留 block-level 思维，但更像 Python / vectorized programming。 ￼

6）别迷信手写

torch.compile 对很多标准模式已经很强。 ￼

7）真正复杂的问题，要从“数据怎么分块、怎么在块内做完整流程”出发

softmax 就是第一个示范题。 ￼

⸻

# 十一、这一章你最该背下来的 10 句大白话
	1.	GPU 优化很多时候不是数学题，是搬运题。  ￼
	2.	benchmark 告诉你慢不慢，profile 告诉你为什么慢。  ￼
	3.	不做 cuda synchronize，很多 benchmark 都是假的。  ￼
	4.	一个 PyTorch 表达式，底下可能是很多 kernel，也可能是一个 fused kernel。  ￼
	5.	kernel fusion 的本质是少写中间结果、少回显存。  ￼
	6.	写 CUDA kernel 时，你想的是“如果我是一个线程，我该算哪一格”。  ￼
	7.	写 Triton kernel 时，你更像是在想“如果我是一个 block，我该处理哪一块”。  ￼
	8.	mask 的意义就是别让最后那点不满块的数据越界乱读。  ￼
	9.	torch.compile 已经能自动做很多你原本想手写的融合优化。  ￼
	10.	softmax 这种题，关键不是公式会不会写，而是“按什么维度分块最自然”。  ￼

⸻

# 十二、给你的复习笔记版

CS336 Lecture 6：Kernels, Triton 详细笔记

1. 本章目标

从“理解 GPU 为什么快/慢”过渡到“如何写高性能 GPU 代码”，主要通过 benchmark/profile、kernel fusion、CUDA kernel、Triton kernel、torch.compile、softmax kernel 来展开。 ￼

2. GPU 执行模型回顾
   •   SM 是执行 block 的硬件单元
   •   block 内线程可共享 shared memory、可同步
   •   grid 是 block 的集合
   •   block 会按 waves 调度到 SM
   •   arithmetic intensity 高更容易跑得好 ￼

3. Benchmark vs Profile
   •   benchmark：看总时间，适合比较实现和看 scaling
   •   profile：看时间花在哪、实际调用了哪些 kernels
   •   GPU benchmark 需要 warmup + torch.cuda.synchronize() ￼

4. Kernel Fusion
   •   多个小操作分开做会产生很多中间读写
   •   fused kernel 把多步放在一次 kernel launch 中
   •   核心收益来自减少显存往返，不是数学变少 ￼

5. CUDA Kernel
   •   CUDA 是 C/C++ 扩展，用来写 GPU kernels
   •   wrapper 负责检查输入、分配输出、设置 grid/block、launch
   •   kernel 负责线程级计算逻辑
   •   必须思考 index mapping 和边界处理 ￼

6. Triton
   •   用 Python 写 GPU kernels
   •   更偏 block-level 抽象
   •   编译器帮助做 memory coalescing、shared memory 等优化
   •   Triton kernel 往往围绕 block 起点、offsets、mask、load/store 展开 ￼

7. PTX
   •   更低层的 GPU 指令表示
   •   用来观察 Triton/CUDA 最终如何被翻译成 load/compute/store
   •   有助于理解寄存器使用和 thread-level 实际行为 ￼

8. torch.compile
   •   自动优化普通 PyTorch 代码
   •   重要能力是 kernel fusion
   •   对很多标准模式已经很强，不必盲目手写 CUDA/Triton ￼

9. Triton Softmax
   •   softmax 不是纯 elementwise，而是 row-wise reduction
   •   一个自然设计是“一行一个 block”
   •   block 内完成 max / exp / sum / divide / write-back
   •   关键思想是把需要一起看的数据放在同一个 block 范围内 ￼

10. 本章总哲学

高性能 GPU 编程 = 先找热点，再决定是否融合、分块、手写 kernel 或交给编译器。核心永远是减少数据搬运，并让需要合作的数据在 block 内完成计算。  ￼
