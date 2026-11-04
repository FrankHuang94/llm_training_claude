# 混合专家（Mixture of Experts, MoE）

> **最后更新：** 2026-11-04
> **相关文件：** [模型架构变体](../00_foundations/05_model_architecture_variants.md) · [分布式训练](../02_pretraining/01_distributed_training.md) · [DeepSeek 路线图](../08_lab_roadmaps/04_deepseek_roadmap.md)
> **关键论文：** Shazeer et al. 2017 Sparsely-Gated MoE（[arXiv:1701.06538](https://arxiv.org/abs/1701.06538)）· Fedus et al. 2021 Switch Transformer（[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)）· Jiang et al. 2024 Mixtral（[arXiv:2401.04088](https://arxiv.org/abs/2401.04088)）· DeepSeek-AI 2024 DeepSeekMoE/V3（[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)）

## 概述
混合专家（mixture of experts，MoE）把模型的**总参数**（容量/知识）与**每个 token 的激活参数**（计算成本）解耦。MoE 层不再使用单个稠密 FFN，而是拥有许多专家（expert）FFN 和一个**路由器（router）**，把每个 token 只发送给少数几个专家。这使得模型可以拥有稠密模型 10–30 倍的参数，而每个 token 的 FLOPs 大致不变——这是作用于缩放定律的一个强力杠杆（固定计算下获得更多容量）。MoE 从研究上的奇思（2017）成长为前沿默认选择：Mixtral、DeepSeek-V3/V4、Llama 4、Qwen3/3.5/3.8、Kimi K2/K3、GLM-5、MiniMax-M2、Mistral Large 3/4 以及 OpenAI 的 gpt-oss 都是 MoE（GPT-4 也被普遍认为是）。到 2026 年，开放 MoE 的总参数达到 **1–2.8T**，每个 token 只激活约 3–5%。理解路由、负载均衡（load balancing）以及系统/训练层面的挑战，如今已是核心知识。

## 核心概念
**MoE 层。** 用 $E$ 个专家 FFN $\{f_1,\dots,f_E\}$ 和一个门控网络替换 FFN。路由器计算得分 $g(x)=\text{softmax}(W_r x)$，选出 **top-$k$** 个专家，并以（重新归一化的）门控值对它们的输出加权求和：
$$y=\sum_{i\in \text{TopK}(g(x))} g_i(x)\, f_i(x)$$
在 $E=8$ 个专家中取 $k=2$（Mixtral）时，每个 token 只运行 2 个 FFN。**激活参数** ≈ 嵌入 + 注意力 + $k$ 个专家；**总参数**则包含全部 $E$ 个专家。Mixtral 8×7B：总参数约 47B，激活约 13B。DeepSeek-V3：总参数 671B，**激活 37B**（256 个路由专家中取 8 个 + 1 个共享专家）。

**路由。** 通常是 **token 选择的 top-k**（每个 token 挑选自己的专家）。路由器与模型联合训练；它必须学会有用的专精分工。替代方案：**专家选择**（每个专家挑选自己的 top token——保证均衡）、哈希（固定）以及**软 MoE**（对所有专家加权组合，无硬路由）。

**负载均衡。** 不加干预时，路由器会**坍缩**——少数专家拿走所有 token（富者愈富），浪费容量并使**专家容量**（单个专家能处理的最大 token 数；多出的 token 会被*丢弃*）溢出。解决办法：一个鼓励专家利用率均匀的**辅助负载均衡损失**：
$$\mathcal{L}_{aux}=\alpha\, E \sum_{i=1}^{E} f_i\, P_i$$
其中 $f_i$ 是路由到专家 $i$ 的 token 比例，$P_i$ 是专家 $i$ 的平均路由概率。DeepSeek 提出了**无辅助损失**的均衡方法（动态调整每个专家的偏置），以避免辅助损失带来的质量代价。

**细粒度 + 共享专家（DeepSeekMoE）。** 把专家拆分成*更多、更小*的专家（更细的专精），并加入始终激活的**共享专家**（承载通用知识），改善专精与冗余之间的权衡。这一设计如今占据主导：

| 模型（发布时间） | 总参数 / 激活参数 | 路由专家（激活 / 总数）+ 共享 |
|---|---|---|
| gpt-oss-120b (2025-08) | 116.8B / 5.1B | 4 / 128 |
| Kimi K2 (2025-07) | ~1T / 32B | 8 / 384 |
| Qwen3-Next-80B-A3B (2025-09) | 80B / ~3B | 10 / 512 + 1 |
| MiniMax-M2 (2025-10) | ~230B / 10B | 256 个专家（全注意力） |
| GLM-5 (2026-02) | 744B / 40B | —（采用 DSA） |
| DeepSeek-V4-Pro (2026-04) | 1.6T / 49B | — |
| Kimi K3 (2026-07) | 2.8T / 104B | 16 / 896 + 2 |
| Qwen3.8 (2026-08) | 2.4T / 95B | 10 / 512 + 1 |
| Mistral Large 4 (2026-10) | ~1.05T / 52B | — |

**系统：专家并行。** 专家被分片到不同设备上；路由需要 **all-to-all** 通信（token → 其专家所在设备，再返回）——这是主要的系统成本，需要通信/计算重叠（DeepSeek-V3 的 DualPipe），并越来越依赖机柜级 NVLink/纵向扩展域。

## 关键挑战
- **训练不稳定。** 路由器是离散/有噪声的；MoE 训练更容易不稳定，需要 **z-loss**/路由器调优；负载均衡损失以质量换取均衡。
- **负载不均衡与 token 丢弃。** 坍缩和容量溢出浪费计算并损害质量；容量因子必须调优。
- **通信开销。** 在大规模下 all-to-all 占主导；对互联带宽敏感。
- **内存占用。** 尽管只有少数专家被激活，所有专家都必须被*存储*（且常常需要被加载）——服务时需要大量内存/显存（一个 2.8T 的 MoE 即使用 MXFP4 发布也约有 1.56 TB）。
- **推断批处理。** 不同 token 命中不同专家；高效的批量服务和专家放置并不简单。
- **MoE 上的 RL。** rollout 引擎和训练器之间的路由可能不同，token 级重要性比率变得嘈杂，如果没有序列级比率或路由回放，大型 MoE 的 RL 会崩溃。
- **微调/泛化。** MoE 的微调可能更棘手且可能过拟合；专家的专精往往不如期望的那样可解释。

## 解决方案与当前最佳实践
**细粒度路由（从数百个专家中取 top-8 到 top-16）+ 共享专家、无辅助损失（或低系数）的负载均衡、用 z-loss 保稳定**，并调优容量因子以尽量减少丢弃。训练时采用**专家 + 张量 + 流水线并行**并配合 **all-to-all 重叠**；服务时在机柜级域内做**专家并行**放置、使用 **4 比特专家权重**（MXFP4/INT4/FP4 QAT），并采用预填充/解码或注意力/FFN 解耦。DeepSeek-V3 的配方（细粒度专家、共享专家、无辅助损失均衡、MTP、fp8、DualPipe）仍是开源参考，并在稀疏注意力（V3.2/V4）和 Muon 训练方面得到了扩展。对于 RL，使用**序列级重要性比率（GSPO）**或路由回放。当你想要**在固定推断 FLOPs 下获得容量**且能负担内存时选择 MoE；在内存受限或希望微调更简单时选择稠密模型。

## 各实验室观点
- **Google** 开创了大规模的现代 MoE（Shazeer 2017、GShard、Switch、GLaM、ST-MoE）——大量奠基性工作；Gemini 使用 MoE；Gemma 4 包含一个 26B-A4B 的 MoE。
- **Mistral** 以 **Mixtral 8×7B/8×22B** 让开放 MoE 成为主流，随后推出"细粒度"的 **Mistral Large 3**（675B / 41B）和 **Large 4**（约 1.05T / 52B，2026-10 开放权重）。
- **DeepSeek** 在架构上推进得最为显著：**DeepSeekMoE**（细粒度 + 共享专家）、无辅助损失均衡、V3 在硬件受限下的 671B/37B 激活效率，以及 1.6T / 49B 的 V4-Pro。
- **月之暗面（Moonshot）/ Qwen** 推向极端稀疏度：Kimi K3 从 896 个专家中激活 16 个；Qwen3-Next/3.8 从 512 个中激活 10 个。
- **Meta** 以 **Llama 4**（Scout/Maverick/Behemoth）转向 MoE；其 2026 年的 Muse 模型是闭源的，架构未公开。
- **OpenAI：**gpt-oss 公开确认采用 MoE（128 个专家，top-4）；GPT-4 被普遍认为是 MoE（未证实）。
- **NVIDIA：**Nemotron 3 交错使用 Mamba-2 层和 MoE 层（Nano 30B-A3B：128 个专家 + 1 个共享专家）。

## 最新进展（2023–2026年11月）
MoE 成为**前沿默认选择**（DeepSeek-V3、Llama 4、Qwen-MoE、Mixtral、Grok）。关键进展：**细粒度 + 共享专家**、**无辅助损失负载均衡**、**fp8 MoE 训练**，以及通信最优的调度。**MoE 缩放定律**（将最优专家数量/粒度与计算量联系起来）开始出现。关于**升级改造（upcycling）**（把稠密模型转换为 MoE）以及面向内存受限部署的服务效率（专家下放、预测）的研究也在进行。

**2025年中 → 2026年11月更新：**
- **万亿规模的开放 MoE（2025-07 → 2026-10）。** 最大的开放权重从 1T（Kimi K2）增长到 1.6T（DeepSeek-V4-Pro）、2.4T（Qwen3.8）和 2.8T（Kimi K3），而激活参数保持在约 13–104B——稀疏度不断上升，激活计算量则保持对服务友好。Mistral Large 4（约 1.05T）是第一个接近 1T 的西方开放模型。
- **算力分配定律（2026-03）。** 专家与注意力之间的最优计算比随总计算量呈幂律增长，并取决于稀疏度（[arXiv:2603.10379](https://arxiv.org/abs/2603.10379)）；据报道，一项相关的 ICLR 2026 结果发现，记忆与推理的最优稀疏度不同，推理有时更偏好更稠密的模型。
- **MoE 的 RL 稳定性（2025-07）。** GSPO 的序列级、长度归一化的重要性比率配合序列级裁剪，修复了曾导致大型 MoE RL 崩溃的 token 级重要性采样噪声，并免去了路由回放的需要；用于 Qwen3（[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）。
- **极端稀疏度下的均衡（2026-07）。** 据报道，Kimi K3 的报告把 896 个专家下的训练稳定性归功于"Quantile Balancing"（以及 SiTU-GLU）。
- **4 比特专家成为默认（2025-08 → 2026-07）。** gpt-oss 以 MXFP4 发布 MoE 权重；Kimi K2 Thinking 对 MoE 组件使用 INT4 QAT；DeepSeek-V4 对路由专家使用 FP4 QAT；Kimi K3 以 MXFP4 发布。专家占据大部分参数却只占少量 FLOPs，因此是天然的量化目标。
- **机柜级服务与解耦（2026）。** NVIDIA 的指南在 GB300 NVL72 上服务 Qwen3.8；Trn3 UltraServer（144 颗芯片）面向类似负载。一项注意力–FFN 解耦研究发现，聚合式服务在吞吐量上胜出，而 AFD 有助于交互性；注意力廉价的模型可以把 ≥90% 的 GPU 分配给 FFN/专家（[arXiv:2605.28302](https://arxiv.org/abs/2605.28302)）。
- **MoE + 高效注意力。** 大多数新 MoE 把稀疏专家与次二次注意力搭配使用（GLM-5 用 DSA，Qwen3-Next/3.5 用 Gated DeltaNet 混合，Kimi Linear/K3 用 KDA + MLA，Nemotron 3 用 Mamba-2）；MiniMax-M2 是一个值得注意的全注意力例外。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么用 MoE——它能带来什么？"** 把总容量与每 token 激活 FLOPs 解耦 → 在大致固定的计算量下获得更多参数/知识。
- **"什么是负载均衡问题，如何解决？"** 路由器坍缩 → 辅助负载均衡损失（$\sum f_i P_i$）或无辅助损失的偏置均衡；容量因子 + token 丢弃。
- **"Mixtral 与 DeepSeek-V3 的架构差异？"** 少量大专家（8 取 2）vs 大量细粒度专家 + 共享专家 + 无辅助损失均衡。
- **"MoE 训练的主要系统瓶颈是什么？"** 专家路由的 all-to-all 通信；需要重叠和高互联带宽。
- **"推断时 MoE 与稠密模型——内存上的权衡？"** 激活 FLOPs 更少，但必须存储所有专家 → 内存占用高。
- **"为什么大型 MoE 上的 RL 会崩溃，GSPO/路由回放如何修复？"** 当采样器和训练器的路由不同时，token 级重要性比率很嘈杂；序列级比率/裁剪（GSPO）或回放采样器的路由决策可以稳定更新。
- **"为什么稀疏度一直在上升（例如 896 个专家中激活 16 个）？"** 容量随总参数增长，而服务成本取决于激活参数和内存带宽；机柜级域和 4 比特专家使存储庞大的专家池变得负担得起。

## 开放问题
最优的**专家粒度、数量与路由**缺乏定论（而且对知识与推理可能不同）；极端稀疏度下的**训练稳定性**以及专家专精的**可解释性（interpretability）**仍不完善。数万亿参数 MoE 的**内存高效服务**、MoE 策略上稳定的 **RL**、稳健的**微调**以及可靠的**稠密→MoE 升级改造**都是活跃的研究方向。在最大规模上 MoE 与稠密（或混合）模型孰优孰劣，仍有争论。

## 参考文献
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks (Sparsely-Gated MoE).* arXiv:1701.06538.
- Fedus, W. et al. (2021). *Switch Transformer.* arXiv:2101.03961.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Dai, D. et al. (2024). *DeepSeekMoE.* arXiv:2401.06066.
- Zoph, B. et al. (2022). *ST-MoE.* arXiv:2202.08906.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
