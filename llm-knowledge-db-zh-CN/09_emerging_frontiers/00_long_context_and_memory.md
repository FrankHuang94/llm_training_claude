# 长上下文与记忆（Long Context and Memory）

> **最后更新：** 2026-11-04
> **相关文件：** [上下文窗口扩展](../06_efficiency/06_context_window_extension.md) · [非 Transformer 架构](04_neuromorphic_and_non_transformer.md) · [扩散与混合语言模型](06_diffusion_and_hybrid_language_models.md) · [智能体框架](../04_reasoning_and_agents/04_agentic_frameworks.md)
> **关键论文：** Liu et al. 2023 Lost in the Middle ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) · Packer et al. 2023 MemGPT ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)) · Behrouz et al. 2024 Titans ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) · Gemini Team 2024 (1M+ context) ([arXiv:2403.05530](https://arxiv.org/abs/2403.05530)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692))

## 概述
随着模型开始处理整本书长度的文档、长达数小时的视频、持续多天的智能体（agent）任务以及个性化助理，**上下文**（当前窗口中的信息）与**记忆**（跨窗口/跨会话持久保存的信息）之间的区分正变得至关重要。上下文窗口（context window）已达到 1–10M token——到 2026 年，**约 1M token 已成为闭源前沿模型与开放权重旗舰模型的默认规格**——但仍有两个问题：模型**并不能均匀地利用长上下文**（"lost in the middle"、"上下文腐化（context rot）"），而且固定窗口无论多大都不是真正的**持久记忆**。这一前沿方向既探索延长上下文长度，*也*探索构建外部/参数化记忆，使模型能够随时间积累和回忆信息。它连接了效率工程（KV 缓存管理、稀疏/混合注意力）与架构研究（记忆模块、状态空间模型（SSM）、测试时训练得到的记忆）。

## 核心概念
**上下文 vs 记忆。**
- **上下文：** Transformer 的注意力窗口——快速、可完全访问，但代价是二次复杂度且是短暂的（每次调用后清空）。
- **记忆：** 在窗口*之外*存储和检索信息的机制：外部的（检索/数据库——检索增强生成（RAG）、向量库），或*内部/参数化*的（把信息向前传递的权重或循环状态）。

**Lost in the middle（Liu et al.）。** 即使在支持的窗口之内，检索/利用准确率随位置呈 **U 形**——开头和结尾强，中间弱。名义上的长上下文 ≠ 均匀可用的上下文；这推动了更好的位置方法、长文档训练以及检索重排序。

**外部记忆架构。**
- **MemGPT：** 把 LLM 当作带虚拟内存的操作系统——固定的"主上下文"加上"外部存储"，由模型发出函数调用把信息**换页**进出，从而实现事实上无界的记忆。
- **RAG / 向量记忆：** 存储过往交互/文档的嵌入；每次查询检索 top-k 个相关片段（关于溯源与依据，见[幻觉](../05_alignment_and_safety/01_hallucination_causes_and_mitigations.md)）。
- **智能体记忆：** 短期草稿区 + 长期情景/语义存储 + 习得技能（见[智能体框架](../04_reasoning_and_agents/04_agentic_frameworks.md)）。**上下文工程（context engineering）**（压缩（compaction）、结构化笔记、拥有干净窗口的子智能体、通过工具即时检索）如今已成为智能体的实用学科。

**内部/学习式记忆（2024–2026 年的前沿）。**
- **Titans**（Behrouz et al.）：为注意力增加一个**神经长期记忆**模块，它*在测试时学习如何记忆*（基于"惊讶度"的门控），把注意力的精确性与压缩式循环记忆结合起来——可扩展到极长序列。其后继者（ATLAS、HOPE）以及 **TTT-E2E** 把长上下文视为*测试时学习*（见[测试时训练](03_test_time_training.md)）。
- **压缩式/循环记忆**（Infini-attention、RMT、RWKV/SSM 状态、Gated DeltaNet/KDA）：把过去的上下文压缩成一个有界状态向前传递——次二次复杂度但有损。

**KV 缓存即记忆。** KV 缓存*就是* Transformer 的工作记忆；对其进行管理（量化、驱逐非"重要命中（heavy-hitter）"token、分页、前缀缓存、MLA 潜空间压缩、学习式 top-k 稀疏选择）是长上下文记忆的实践层面（见[上下文窗口扩展](../06_efficiency/06_context_window_extension.md)）。对于全注意力模型，KV 占用为 $2\,n_{\text{layers}}\,n_{kv}\,d_{\text{head}}\,L$ 个元素，随上下文长度 $L$ 线性增长——这正是 2025–26 年混合式与压缩-稀疏设计所要攻克的项。

## 关键挑战
- **有效上下文 ≪ 名义上下文。** 模型无法在整个 1M token 窗口上可靠地推理（lost-in-the-middle；多跳推理退化；随着无关 token 累积出现的"上下文腐化"）。
- **成本。** 长上下文的预填充（prefill）计算与 KV 内存都很大；持久记忆又增加了存储/检索开销。
- **记忆管理策略。** *存储、检索、压缩、遗忘什么*以及何时进行，是一个尚未解决的控制问题（"记忆层级"问题）；2026 年的工作开始用强化学习（RL）来*学习*这一策略。
- **灾难性干扰。** 参数化记忆的更新有覆盖旧知识的风险（关联[持续学习](01_continual_learning.md)）。
- **检索质量。** RAG 的效果取决于检索；错误会向下游传播。
- **评测。** 大海捞针测试不够；多跳、聚合和长时程记忆基准（RULER、∞-Bench、LongMemEval、MRCR 多针）仍不成熟。

## 解决方案与当前最佳实践
**混合方案：长上下文 + 检索 + 结构化记忆。** 用长上下文做稠密的任务内推理；用 **RAG/向量记忆**承载大规模外部知识；用 **MemGPT 式分页**、压缩和智能体笔记实现跨会话持久化。用 **MLA/GQA + 量化 + 驱逐/分页**管理 KV；在 1M 规模上，开放模型通过 **约 3:1 的线性:全注意力混合**（Gated DeltaNet/KDA + 全注意力/MLA 层）或**在（压缩后的）KV 上做学习式 top-k 稀疏注意力**（DSA → CSA/HCA）让上下文变得可负担。用**长文档**训练，并用**多跳**长上下文基准（而不仅是大海捞针）评测。新兴的**学习式记忆**模块（Titans/ATLAS/HOPE）、**测试时训练得到的权重**（TTT-E2E）以及 **SSM/注意力混合**（见[非 Transformer](04_neuromorphic_and_non_transformer.md)、[扩散与混合语言模型](06_diffusion_and_hybrid_language_models.md)）瞄准的是次二次复杂度的持久记忆。

## 各实验室观点
- **Google DeepMind：** 率先在生产中提供 1–2M token 上下文（Gemini 1.5 起），并开展面向 10M+ 的研究（ATLAS、Nested Learning/HOPE）；如今已不再独占 1M。
- **OpenAI/Anthropic：** 从 128K–200K 扩展到 **约 1M** 窗口（Claude Opus 4.6 于 2026-02 推出 1M beta 并配有上下文压缩；GPT-5.4 和 GPT-6 Astra 约 1.05M），强调*可靠*利用（多针 MRCR）以及智能体层面的记忆（压缩、并行子智能体）。
- **中国开放权重实验室（DeepSeek、Moonshot、Qwen、MiniMax）：** 把 1M 上下文当作架构问题——DSA/CSA/HCA（DeepSeek）、KDA+MLA 混合（Kimi）、Gated DeltaNet 混合（Qwen）、块稀疏 MSA（MiniMax M3）。
- **学术界/初创公司：** MemGPT（Letta）、Titans、Infini-attention、RMT、TTT-E2E——外部与学习式记忆架构；用 RL 训练的记忆工具（AgeMem、Memory-R1）。
- **SSM 实验室 / NVIDIA：** 以 Mamba 系列混合模型（Nemotron 3，1M 上下文）作为长上下文的基底。

## 最新进展（2023–2026年11月）
**1–2M token** 上下文已上线（Gemini）；研究朝 10M+ 推进。**学习式测试时记忆**（Titans）和**压缩式循环**记忆取得进展。**智能体/持久记忆**系统（MemGPT/Letta、助理中的记忆功能）在个性化和长时程智能体场景中走向成熟。新的**长上下文推理**基准（RULER、∞-Bench、LongMemEval）暴露了有效与名义之间的差距，使研究重心转向*可用*记忆。争论焦点：扩展上下文 vs 构建显式记忆 vs 更换架构（SSM）。

**2025 年中 → 2026 年 11 月更新：**
- **1M 成为默认规格，竞争转向 KV/FLOP 成本（2025-09 → 2026-10）。** DeepSeek-V4（2026-04）、Kimi K3（2026-07）、MiniMax M3（约 2026-06）、MiMo-V2.5 和 Nemotron 3 都提供 1M 上下文；Claude Opus 4.6（2026-02）是首个支持 1M（beta）的 Opus，在 1M 下的 MRCR v2 8-needle 上得分 76%，而 Sonnet 4.5 为 18.5%（厂商数据）——这表明竞争维度是*可用*上下文而非名义长度。
- **混合线性注意力（2025-09/10）。** Qwen3-Next 以 3:1 交错 Gated DeltaNet 与门控 softmax 注意力（在 32K 以上相对 Qwen3-32B 预填充吞吐量 >10×）；**Kimi Linear**（KDA，通道级门控的 delta 规则，与 MLA 3:1 混合）报告在 1M 下 KV 缓存最多减少 75%、解码吞吐量最高提升 6×（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）。MiniMax 的 M2 以多跳推理损失为由回退到全注意力——这是一个关键的反例。
- **在压缩 KV 上的学习式稀疏注意力（2025-09 → 2026-04）。** DSA（lightning indexer + top-k，[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）演化为 DeepSeek-V4 的 CSA（先把每 $m$ 个 token 的块压缩，再取 top-k）+ HCA（在高度压缩的 KV 上做稠密注意力）；评述者报告在 1M 下每 token FLOPs 约为 V3.2 的 27%、KV 约为其 10%（报告列为 arXiv:2606.19348；编号/日期不一致，需核实）。
- **把长上下文视为测试时学习（2025-05 → 2025-12）。** ATLAS 在过去 token 的滑动窗口上优化记忆（Omega 规则），报告在 10M token 的 BABILong 上提升 80%（[arXiv:2505.23735](https://arxiv.org/abs/2505.23735)）；**TTT-E2E** 在上下文上执行下一 token 梯度步，并使用元学习得到的初始化，在 3B 规模上像全注意力一样随上下文扩展，同时保持每 token 延迟恒定，在 128K 下提速 2.7×（[arXiv:2512.23675](https://arxiv.org/abs/2512.23675)）。
- **门控注意力修复注意力汇聚点并改善长度泛化（2025-05）。** 在 SDPA 之后加一个逐头 sigmoid 门，把首 token 注意力从 46.7% 降到 4.8%，RULER 提升 >10 分；已在 Qwen3-Next 中采用（[arXiv:2505.06708](https://arxiv.org/abs/2505.06708)）。
- **面向智能体的上下文工程与学习式记忆（2025-09 → 2026-01）。** Anthropic 的指南把上下文视为受"上下文腐化"制约的有限注意力预算（压缩、笔记、子智能体、即时检索）；**AgeMem** 把存储/检索/更新/丢弃/总结变为策略的工具动作，并用逐步 GRPO 训练（[arXiv:2601.01885](https://arxiv.org/abs/2601.01885)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"上下文和记忆有什么区别？"** 窗口内注意力（短暂、二次复杂度）vs 跨会话持久化/可检索的信息。
- **"什么是 'lost in the middle'，如何缓解？"** U 形的位置利用；长文档训练、检索重排序、位置编码修正。
- **"你会如何为智能体提供长期记忆？"** MemGPT 式分页 + 向量库 + 情景/语义记忆 + 压缩；管理存储/检索/遗忘什么（越来越多地用 RL 来学习）。
- **"长上下文 vs RAG vs 学习式记忆——如何权衡？"** 成本/召回/持久性；实践中采用混合方案。
- **"2026 年的开放模型如何让 1M 上下文变得可负担？"** 采用状态有界的约 3:1 线性（GDN/KDA）:全注意力混合，或在压缩 KV 上做 top-k 稀疏注意力（DSA/CSA）；要会 KV 大小公式，以及 MiniMax 关于多跳推理的反驳论点。

## 开放问题
缩小**有效与名义**上下文之间的差距（在数百万 token 上进行真正的多跳推理）、有原则的**记忆管理**策略、**足够无损的压缩式记忆**、**无干扰**的参数化记忆，以及可靠的**长时程记忆评测**都尚未解决。混合/稀疏注意力是否会丢失长链推理所必需的东西（MiniMax 的观点），还是在公平比较下能与全注意力持平（Kimi Linear 的观点），仍无定论。未来究竟是"无限上下文"、显式记忆系统、测试时训练的权重，还是新架构，是一个开放且高风险的争论。

## 参考文献
- Liu, N. et al. (2023). *Lost in the Middle.* arXiv:2307.03172.
- Packer, C. et al. (2023). *MemGPT.* arXiv:2310.08560.
- Behrouz, A. et al. (2024). *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663.
- Munkhdalai, T. et al. (2024). *Infini-attention.* arXiv:2404.07143.
- Gemini Team (2024). *Gemini 1.5.* arXiv:2403.05530.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- Stanford/NVIDIA et al. (2025). *End-to-End Test-Time Training for Long Context (TTT-E2E).* arXiv:2512.23675.
