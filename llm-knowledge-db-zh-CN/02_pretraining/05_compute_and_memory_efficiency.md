# 计算与内存效率（Compute and Memory Efficiency）

> **最后更新：** 2026-11-04
> **相关文件：** [分布式训练](01_distributed_training.md) · [注意力机制](../00_foundations/01_attention_mechanisms.md) · [混合精度与量化](04_mixed_precision_and_quantization.md)
> **关键论文：** Dao et al. 2022 FlashAttention（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）· Chen et al. 2016 Gradient Checkpointing（[arXiv:1604.06174](https://arxiv.org/abs/1604.06174)）· Korthikanti et al. 2022 Activation Recomputation / Sequence Parallel（[arXiv:2205.05198](https://arxiv.org/abs/2205.05198)）· Chowdhery et al. 2022 PaLM（MFU）（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)）

## 概述
在并行（如何把工作*切分*到多个设备）之外，还有一个正交的问题：如何高效地使用每一个设备——把模型及其激活放进内存，并让昂贵的张量核心保持忙碌，而不是在内存访问或通信上空转。核心指标是 **MFU（模型 FLOPs 利用率，Model FLOPs Utilization）**——训练实际为*有用的*模型计算所达到的算力占硬件峰值 FLOPs 的比例。前沿训练的目标是 35–55% 的 MFU；与 100% 之间的差距来自内存受限、通信、流水线气泡和非矩阵乘法开销。自 2025 年以来，最大的效率杠杆越来越多地来自*架构*层面——次二次注意力、混合 SSM 层、极端的 MoE 稀疏度和 4 比特运算——而不仅仅是内核调优。

## 核心概念
**6N FLOPs 规则。** 对于 $N$ 参数的稠密模型（dense model），每个 token 的训练计算量 $\approx 6N$ FLOPs：前向 $2N$ + 反向 $4N$（反向约为前向的 2 倍）。$D$ 个 token 的总训练 FLOPs $\approx 6ND$——这是算力预算和缩放定律的基础（见[缩放定律](../00_foundations/04_scaling_laws.md)）。（注意力会额外增加一个与序列长度相关的项；MoE 使用*激活*参数；线性注意力/SSM 层用固定大小状态的递推取代了二次项。）

**MFU。** $\text{MFU}=\frac{6ND/T}{\text{peak FLOP/s}\times \text{#devices}}$，其中 $T$ 为墙钟训练时间。它衡量端到端的真实效率（不同于把重计算也计入的"硬件 FLOPs 利用率"）。PaLM 在 540B 规模报告了约 46% 的 MFU——一个标杆数字。注意分母取决于峰值所依据的精度（bf16、fp8 还是 NVFP4），因此跨训练比较 MFU 时必须固定精度。

**激活内存。** 为反向传播保存的前向激活按 $O(\text{batch}\times \text{seq}\times d\times \text{layers})$ 缩放，在长上下文下经常*超过*参数内存——这是长上下文训练的瓶颈约束。

**梯度检查点（激活重计算）。** 不保存全部激活；只保存一部分（"检查点"），其余在反向时**重计算**。以约 33% 的额外计算换取 $O(\sqrt{L})$（或更好）的激活内存。**选择性重计算**（Korthikanti et al.）只重算廉价但占内存多的操作（如注意力），而保留昂贵矩阵乘法的激活——是更好的帕累托点。

**FlashAttention。** 通过分块（tiling）+ 在线 softmax，彻底消除 $O(n^2)$ 的注意力激活内存，使长上下文变得可行并加快训练（见[注意力机制](../00_foundations/01_attention_mechanisms.md)）。它并未消除 $O(n^2)$ 的*计算量*，而这正是稀疏注意力和线性注意力（sparse attention / linear attention）所针对的。

**下放（offloading）。** 当 GPU 内存耗尽时，把优化器状态/参数/激活移到 CPU 内存或 NVMe（ZeRO-Offload/Infinity）——以 PCIe/NVMe 带宽换容量；适用于受内存限制或预算受限的场景。

## 关键挑战
- **受内存带宽限制的操作。** 注意力、LayerNorm、softmax 和逐元素操作受 HBM 带宽而非 FLOPs 限制——让张量核心闲置。
- **长上下文下的激活爆炸。** 长序列会使激活内存暴涨；检查点有帮助但增加计算。
- **内核启动与非矩阵乘法开销。** 大量小操作无法充分利用 GPU；需要算子融合。
- **重计算与内存的权衡。** 检查点过多浪费 FLOPs，过少则 OOM——最佳点取决于具体负载。
- **1M 上下文下的二次注意力。** 即使有 FlashAttention，在极长序列上注意力 FLOPs 仍占主导，而如今长上下文预训练阶段和长程 RL rollout 都需要这样的长度。

## 解决方案与当前最佳实践
**FlashAttention(-2/3) + 选择性梯度检查点 + bf16/fp8（Blackwell 级硬件上用 NVFP4）+ 算子融合**（通过 `torch.compile`、Triton、CUDA graphs 或融合内核）是标准的效率技术栈。**序列并行**分摊长上下文的激活内存。调优**微批次大小**以平衡内存与流水线气泡。通过性能剖析找出受内存限制的瓶颈并加以融合。把 **MFU** 作为北极星指标来报告和优化。仅在必要时使用 **CPU/NVMe 下放**（受带宽限制）。在架构层面，先选择**稀疏或混合线性注意力**和**高稀疏度 MoE** 来削减每个 token 的 FLOPs 和 KV，再去调内核。

## 各实验室观点
- **Google** 让 **MFU** 成为标准效率指标（PaLM），并针对 TPU 脉动阵列做了大量优化；其第 8 代 TPU 拆分为训练（8t）和推断（8i）两款芯片。
- **NVIDIA/Meta** 在 GPU 生态中推动 FlashAttention、选择性重计算和基于 `torch.compile` 的融合；NVIDIA 的 Nemotron 系列推动以 NVFP4 训练的混合 Mamba-Transformer MoE。
- **DeepSeek** 通过定制内核、fp8 和通信重叠调度，在 H800（互联受限）上取得了出色的 MFU/成本效率，随后又用 DSA（V3.2）和 CSA/HCA（V4）削减了长上下文 FLOPs。
- **月之暗面（Moonshot）/ Qwen** 让混合线性注意力（KDA、Gated DeltaNet，3:1 比例）达到生产级水准。
- **OpenAI/Anthropic** 不公开细节，但显然在激进地优化 MFU 和内存以控制训练成本。

## 最新进展（2023–2026年11月）
**FlashAttention-3** 利用了 Hopper 的异步特性和 fp8。`torch.compile`/Triton 融合和 **CUDA graphs** 降低了启动开销。**fp8** 提高了可达吞吐量（也让 MFU 的计算更复杂）。**MoE** 把重心转向*激活*参数的效率和通信重叠。跨集群和受电力约束的训练推动了对计算/通信协同优化的兴趣，每 token 能耗正成为新兴指标。

**2025年中 → 2026年11月更新：**
- **可学习的稀疏注意力（2025-09/12）。** DeepSeek Sparse Attention 加入一个 FP8 的"闪电索引器"（lightning indexer，少量头、ReLU）为历史 token 打分，使主 MLA 注意力只在 top-$k$ 上运行；索引器本身仍是 $O(L^2)$ 但很廉价，训练方式是先用 KL 对齐的稠密预热，再做稀疏的继续训练（[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）。GLM-5（744B）采用了它（[arXiv:2602.15763](https://arxiv.org/abs/2602.15763)）。
- **压缩稀疏注意力（2026-04）。** DeepSeek-V4 对 KV 块进行压缩（CSA），并增加重度压缩的稠密注意力（HCA）；据报道在 1M 上下文下，V4-Pro 每 token 的 FLOPs 约为 V3.2 的 27%，KV 缓存约为其 10%（DeepSeek-V4 技术报告：arXiv:2606.19348，据其 Hugging Face 模型卡）。
- **混合线性注意力（2025-09/10）。** Qwen3-Next 在一个 80B-A3B 的 MoE 中以 3:1 交错 Gated DeltaNet 与门控注意力，报告在 32K 以上的预填充吞吐量是 Qwen3-32B 的 10 倍以上；Kimi Linear（KDA + MLA，3:1）报告在 1M 下 KV 缓存最多减少 75%、解码吞吐量最高 6 倍（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）。
- **以 SSM 为主的混合架构（2025-08 → 2026-03）。** Nemotron Nano 2 在 62 层中只保留 6 个注意力层（28 个 Mamba-2、28 个 FFN），报告在推理负载上吞吐量最高为 Qwen3-8B 的 6.3 倍（[arXiv:2508.14444](https://arxiv.org/abs/2508.14444)）；Mamba-3 的 MIMO 形式提高了解码的算术强度，并以一半的状态大小达到 Mamba-2 的困惑度（[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)）。
- **MoE 内部的算力分配（2026-03）。** 专家与注意力之间的最优计算比随总计算量呈幂律增长，并取决于稀疏度（[arXiv:2603.10379](https://arxiv.org/abs/2603.10379)）。
- **硬件专用化（2025-09 → 2026-04）。** NVIDIA 的 Rubin CPX（GDDR7，约 30 PF NVFP4）把计算受限的预填充与带宽受限的解码分开处理；Rubin GPU 的指标为 50 PF NVFP4、288 GB HBM4；Trainium3 单芯片为 2.52 PF FP8。峰值 FLOPs 如今常以 4 比特报出，这会抬高 MFU 分母的表观数值。
- **反例（2025-10）。** MiniMax 把 M2 改回了全注意力：混合架构在标准基准上与全注意力持平，但在多跳推理上落败，而且线性注意力的状态对低精度很敏感——效率型架构必须在高难度推理上验证，而不能只看困惑度。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"推导 N 参数模型的训练 FLOPs。"** $6ND$：每个 token 前向 $2N$、反向 $4N$；注意力项单独计算。
- **"什么是 MFU，为什么通常低于 60%？"** 有用 FLOPs / 峰值；内存受限、通信、气泡、非矩阵乘法操作。
- **"梯度检查点如何以计算换内存？"** 少存激活，在反向中重算（约 33% 额外 FLOPs）；选择性重计算对其进行优化。
- **"你的长上下文训练因激活而 OOM——怎么办？"** FlashAttention、选择性检查点、序列并行、更小的微批次、下放。
- **"DSA 的索引器仍是 $O(L^2)$——为什么还能节省计算？"** 索引器只用少量低维 FP8 头和 ReLU 打分，二次项的常数极小；昂贵的 MLA 注意力随后只在 top-$k$ 个 token 上运行，使主要开销变为 $O(Lk)$。

## 开放问题
随着模型越来越受通信限制（MoE、长上下文、多数据中心），进一步提高 MFU 是一个开放的系统挑战。重计算、下放与并行之间的最优自动权衡（相对于手工调优）尚未解决，而前沿的每 token 能耗/成本效率越来越受限于电力和网络，而不是原始 FLOPs。稀疏或混合线性注意力能否在 1M 上下文的多跳推理上追平全注意力，以及如何在不同精度和架构之间公平地报告效率，仍是开放问题。

## 参考文献
- Dao, T. et al. (2022). *FlashAttention.* arXiv:2205.14135.
- Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174.
- Korthikanti, V. et al. (2022). *Reducing Activation Recomputation.* arXiv:2205.05198.
- Chowdhery, A. et al. (2022). *PaLM (MFU).* arXiv:2204.02311.
- Kaplan, J. et al. (2020). *Scaling Laws (6N rule).* arXiv:2001.08361.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- NVIDIA (2025). *NVIDIA Nemotron Nano 2.* arXiv:2508.14444.
