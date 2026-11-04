# 分布式训练（Distributed Training）

> **最后更新：** 2026-11-04
> **相关文件：** [计算与内存效率](05_compute_and_memory_efficiency.md) · [混合精度与量化](04_mixed_precision_and_quantization.md) · [混合专家](../06_efficiency/05_mixture_of_experts.md)
> **关键论文：** Shoeybi et al. 2019 Megatron-LM（[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)）· Rajbhandari et al. 2019 ZeRO（[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)）· Narayanan et al. 2021 Efficient Large-Scale Training（3D）（[arXiv:2104.04473](https://arxiv.org/abs/2104.04473)）· Zhao et al. 2023 PyTorch FSDP（[arXiv:2304.11277](https://arxiv.org/abs/2304.11277)）

## 概述
前沿模型拥有数千亿到数万亿参数（仅开放权重的 MoE 总参数就已达到 1–2.8T），并在数万到数十万块加速器上训练——远超任何单个设备的内存和算力。分布式训练是一门**在设备之间切分模型、数据和计算**、同时尽量减少侵蚀效率的通信与空闲时间的学问。核心挑战是**内存墙**（模型 + 优化器状态 + 激活放不下）和**通信墙**（在网络间同步梯度/激活）。掌握并行方式的分类以及 ZeRO/FSDP，是任何大规模训练岗位的基本功。

## 核心概念
**内存预算。** 对于一个以混合精度（mixed precision）和 Adam 训练的 $\Psi$ 参数模型，稳态内存大致为：fp16 参数（$2\Psi$）+ fp16 梯度（$2\Psi$）+ fp32 优化器（optimizer）状态（主权重 $4\Psi$ + 动量 $4\Psi$ + 方差 $4\Psi$）$\approx 16\Psi$ 字节——*还要再加上激活*。这就是为什么一个 70B 模型仅状态就需要约 1.1TB，因而必须切分。

**数据并行（data parallelism，DP）。** 复制模型、切分批次；每步对梯度做 all-reduce。简单且每步通信量小，但会复制全部状态——受内存限制。

**张量并行（tensor parallelism，TP，Megatron）。** 把单个矩阵切分到多个设备上。对于 FFN $Y=\text{GeLU}(XA)B$：$A$ 按**列**切分（之后无需通信），$B$ 按**行**切分（之后做 all-reduce）。注意力按头切分。TP 需要**高带宽的纵向扩展（scale-up）**互联（NVLink），因为它在*每一层内部*都要通信；通常限制在一个 NVLink 域内（HGX 节点上 TP=8；NVL72 等机柜级域可以放宽这一限制）。

**流水线并行（pipeline parallelism，PP）。** 把层划分为若干阶段分布到不同设备上；微批次像装配线一样依次流过。代价是**气泡**（填充/排空时的空闲时间）：对于 $p$ 个阶段和 $m$ 个微批次，气泡占比 $\approx \frac{p-1}{m}$。**GPipe** 与 **1F1B**（交错式，PipeDream）调度可以减小气泡和激活内存。

**序列/上下文并行（SP）。** 沿序列维度切分（以及注意力/LayerNorm 中 TP 未覆盖的部分），对长上下文（long context）至关重要；**Ring Attention** 是上下文并行的一种形式。

**ZeRO（零冗余优化器，Zero Redundancy Optimizer）。** 通过在数据并行各 rank 之间*分片（sharding）*状态，消除 DP 的状态复制：**Stage 1** 分片优化器状态（这部分节省 $4\times$），**Stage 2** 再分片梯度，**Stage 3** 再分片参数（按需用 all-gather 收集，梯度用 reduce-scatter）。ZeRO-3 ≈ PyTorch **FSDP**：每个 rank 持有一个分片，即时物化完整层，用完即释放。ZeRO-Offload/Infinity 把状态下放到 CPU/NVMe。

**3D 并行。** 组合 **DP × TP × PP**（以及 SP）把模型映射到集群上：节点内用 TP（NVLink），跨节点用 PP，副本之间用 DP。MoE 还增加了**专家并行**（专家分布到不同设备上，并通过 all-to-all 分发 token）。

**分片与矩阵优化器。** 逐元素的优化器（Adam）在 ZeRO/FSDP 下很容易分片。矩阵优化器（Muon）在做 Newton–Schulz 正交化时需要*完整*的动量矩阵，迫使每个权重矩阵做一次 all-gather——这正是 Dion 这类原生支持分片的设计的动机。

## 关键挑战
- **通信开销。** all-reduce/all-gather 的数据量可能占主导；带宽和拓扑（节点内 vs 节点间）决定了哪些配置可行。
- **流水线气泡。** 除非微批次数量很多且调度巧妙，否则空闲的阶段会浪费算力。
- **激活内存。** 在长上下文下常常超过参数内存；需要检查点/下放（见[计算与内存效率](05_compute_and_memory_efficiency.md)）。
- **掉队者与容错。** 在 1 万块以上 GPU 时硬件故障很频繁；单个慢节点/坏节点就会拖住整个同步训练。
- **极端的专家稀疏度。** 数百个细粒度专家（例如 Kimi K3 有 896 个路由专家、每次激活 16 个）使 all-to-all 数据量和每个专家的负载倾斜成为瓶颈约束。

## 解决方案与当前最佳实践
标准映射：**NVLink 域内用 TP**，**跨节点用 PP** 并采用交错 1F1B，其余部分用 **ZeRO/FSDP 数据并行**，MoE 层用**专家并行**，长上下文再加上**序列并行**。框架：**Megatron-LM**、**DeepSpeed**、**PyTorch FSDP**、**Megatron-DeepSpeed**，以及较新的 **TorchTitan**/**nanotron**。让通信与计算重叠，使用 bf16/fp8（在 Blackwell 级硬件上现在还有 NVFP4）、梯度检查点，并调优微批次大小以掩盖气泡。目标是高 **MFU**（大规模下 35–55% 已属良好）。对于 RL 后训练，要把 **rollout（推断引擎）工作节点与训练器（trainer）工作节点分离**，并把权重同步纳入整体并行方案。

## 各实验室观点
- **NVIDIA/Microsoft** 主导 Megatron-LM 和 DeepSpeed/ZeRO——事实上的开源技术栈；Microsoft 还提出了 Dion，用于对分片友好的正交化更新。
- **Meta** 主导 **PyTorch FSDP**（ZeRO-3 风格）和 TorchTitan；在 1.6 万块 H100 上训练了 Llama 3。
- **Google DeepMind** 使用 TPU pod，配合 **GSPMD/Pathways** 和 JAX（`jax.pmap`/`shard_map`），这是一套不同但类似的并行模型；Ironwood（TPU v7）于 2026-03 正式可用（GA）。
- **DeepSeek** 在 H800（互联受限）上通过定制流水线调度（DualPipe）和感知通信的 MoE（V3）实现了极致效率，现已扩展到 1.6T 参数的 MoE（V4-Pro）。
- **月之暗面（Moonshot）**用分布式 Muon（MuonClip）在 15.5T token 上训练了 1T 参数的 MoE（Kimi K2）；**智谱（Zhipu）**构建了异步智能体式（agentic）RL 基础设施（slime），据报道 GLM-5 完全在华为昇腾 + MindSpore 上训练（Z.ai 未证实）。

## 最新进展（2023–2026年11月）
**FSDP2**、**DualPipe**（DeepSeek-V3 近乎零气泡的调度）、Hopper 上的 **fp8 训练**（Transformer Engine），以及面向超大 MoE 模型的**专家并行 + 通信重叠**。面向 1 万–10 万 GPU 规模训练的容错/弹性训练（异步检查点、热备）日趋成熟。随着前沿集群逼近电力/网络极限，跨数据中心和低带宽训练（DiLoCo、分布式低通信方法）开始出现。

**2025年中 → 2026年11月更新：**
- **万亿规模的开放 MoE 对专家并行施压（2025-07 → 2026-08）。** Kimi K2（1T / 32B 激活，384 个专家，激活 8 个）、DeepSeek-V4-Pro（1.6T / 49B，2026-04）、Kimi K3（2.8T / 104B，896 个路由专家激活 16 个 + 2 个共享专家，2026-07）以及 Qwen3.8（2.4T / 95B，512 个专家，2026-08）。NVIDIA 的服务指南把 Qwen3.8 部署在 GB300 NVL72 上——机柜级 NVLink 域正在成为专家并行的基本单元。
- **原生支持分片的矩阵优化器（2025-04，2025-09 修订）。** Dion 用对动量的摊销幂迭代取代 Newton–Schulz，只对秩为 $r$ 的子空间做正交化；它兼容 FSDP 与 TP 分片，无需收集完整矩阵（[arXiv:2504.05295](https://arxiv.org/abs/2504.05295)）。Kimi K2 证明了分布式 Muon 在 1T 参数规模上可行（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）。
- **机柜级纵向扩展域（2025-12 → 2026）。** Trainium3 UltraServer 容纳 144 颗芯片（362 PF FP8，全互连的 NeuronSwitch 结构，集群最多 100 万颗芯片）；NVIDIA 的 Vera Rubin NVL144 机柜在 72 个封装上提供约 3.6 EF NVFP4；Google 把第 8 代 TPU 拆分为训练（8t）和推断（8i）两款芯片（2026-04）。更大的纵向扩展域让 TP/EP 可以跨越远多于 8 个设备。
- **吉瓦级集群。** 算力现在以 GW 为单位签约（OpenAI–AMD 自 2026 年下半年起 6 GW 的 MI450；Anthropic 最多 100 万颗 TPU、超过 1 GW；Project Rainier 约 50 万颗 Trainium2）。据报道 OpenAI 的 GPT-6 Astra（2026-09）在超过 10 万块 GPU 上训练（单一来源）。约束同步训练规模的越来越是电力，而不是芯片。
- **RL 系统也是分布式训练（2025-05 → 2026-02）。** 完全异步的 RL 将 rollout 与训练工作节点解耦，并采用有界陈旧度的 PPO（AReaL，最高快 2.77 倍，[arXiv:2505.24298](https://arxiv.org/abs/2505.24298)）；PipelineRL 在生成过程中把**飞行中权重更新（in-flight weight update）**推送给生成器（长 CoT 上约快 2 倍，[arXiv:2509.19128](https://arxiv.org/abs/2509.19128)）；slime（Megatron + SGLang）支撑了 GLM-5 的智能体式 RL。
- **跨越训练器/推断边界的数值问题（2025-09/10）。** 非批次不变（batch-invariant）的内核和 BF16 舍入使 rollout 引擎的对数概率与训练器的对数概率产生偏离；批次不变内核（Thinking Machines）以及把 RL 切换到 FP16（[arXiv:2510.26788](https://arxiv.org/abs/2510.26788)）可恢复同策略（on-policy）性——这是解耦式 RL 栈的一项新的正确性要求。
- **面向非 NVIDIA 芯片的硬件感知格式（2025-08 起）。** DeepSeek 的 UE8M0（2 的幂）FP8 缩放因子被定位为面向国产芯片；DeepSeek-V4 的训练硬件仍有争议。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"解释 ZeRO 的 stage 1/2/3，以及 stage 3 与 FSDP 的关系。"** 分片优化器/梯度/参数状态；ZeRO-3 ≈ FSDP 的 all-gather/reduce-scatter。
- **"为一个 400B 模型在 8k 块 GPU 上设计并行方案。"** 节点内 TP，节点间 PP，外层 DP/FSDP，上下文用 SP；用带宽来论证。
- **"什么是流水线气泡，如何减小？"** $\frac{p-1}{m}$；增加微批次，使用交错 1F1B。
- **"为什么要把 TP 限制在节点内？"** 它每一层都要通信；需要 NVLink 级带宽。
- **"如何在 FSDP 下运行 Muon？"** Newton–Schulz 需要完整矩阵 → 每个矩阵做一次 gather（或把整个矩阵分配给某个 rank），或使用原生支持分片的变体（Dion：在分片动量上做低秩幂迭代）。
- **"为一个 1T 的 MoE 设计 RL 后训练系统。"** 把 rollout 引擎与训练器分离，采用有界陈旧度的异步方式、快速权重广播，并匹配数值（批次不变内核 / FP16 / 路由一致性），使"同策略"真正保持同策略。

## 开放问题
把同步训练扩展到约 10 万块加速器以上会给电力、网络和可靠性带来压力；**低通信/去中心化**训练（跨数据中心、联邦式）在前沿质量上尚未得到证明。最优的自动并行化（相对于手工调优的配置）以及不损失吞吐量（throughput）的稳健容错仍是开放的工程前沿，万亿规模下通信高效的矩阵优化器、以及异步 RL 中吞吐量与陈旧度的权衡同样如此。

## 参考文献
- Shoeybi, M. et al. (2019). *Megatron-LM.* arXiv:1909.08053.
- Rajbhandari, S. et al. (2019). *ZeRO.* arXiv:1910.02054.
- Narayanan, D. et al. (2021). *Efficient Large-Scale LM Training on GPU Clusters.* arXiv:2104.04473.
- Zhao, Y. et al. (2023). *PyTorch FSDP.* arXiv:2304.11277.
- DeepSeek-AI (2024). *DeepSeek-V3 (DualPipe).* arXiv:2412.19437.
- Ahn, K. et al. (2025). *Dion: Distributed Orthonormalized Updates.* arXiv:2504.05295.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Fu, W. et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* arXiv:2505.24298.
- Piché, A. et al. (2025). *PipelineRL.* arXiv:2509.19128.
