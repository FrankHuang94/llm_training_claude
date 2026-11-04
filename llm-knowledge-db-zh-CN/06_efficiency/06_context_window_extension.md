# 上下文窗口扩展（Context Window Extension）

> **最后更新：** 2026-11-04
> **相关文件：** [位置编码](../00_foundations/03_positional_encoding.md) · [注意力机制](../00_foundations/01_attention_mechanisms.md) · [长上下文与记忆](../09_emerging_frontiers/00_long_context_and_memory.md)
> **关键论文：** Chen et al. 2023 Position Interpolation（[arXiv:2306.15595](https://arxiv.org/abs/2306.15595)）· Peng et al. 2023 YaRN（[arXiv:2309.00071](https://arxiv.org/abs/2309.00071)）· Xiao et al. 2023 StreamingLLM（[arXiv:2309.17453](https://arxiv.org/abs/2309.17453)）· Liu et al. 2023 Lost in the Middle（[arXiv:2307.03172](https://arxiv.org/abs/2307.03172)）· DeepSeek-AI 2025 DeepSeek Sparse Attention（[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）

## 概述
上下文窗口（context window）已从 2K（GPT-3）增长到 128K–1M+ token，使整库代码推理、长文档分析和多样本提示成为可能。到 2026 年末，**约 1M token 已是事实上的前沿规格**：Gemini 自 1.5 起就提供 1–2M，Claude Opus 4.6 推出了 1M（beta），GPT-5.4 和 GPT-6 Astra 提供约 1.05M，而开放模型（DeepSeek-V4、Kimi K3、MiniMax-M3、MiMo-V2.5、Nemotron 3）原生支持 1M。竞争焦点已经从名义长度转向 **1M 下的每 token 成本**（FLOPs 和 KV 缓存）以及对上下文的**有效**利用。扩展上下文是一个多方面的问题：**位置编码（positional encoding）**必须泛化到训练长度之外，**注意力成本**（$O(n^2)$）必须被驯服，**KV 缓存**内存必须被管理，而且模型必须真正*用好*远处的信息（"迷失在中间"问题）。本文聚焦于扩展的*技术*；记忆架构和超长上下文的*前沿*见[长上下文与记忆](../09_emerging_frontiers/00_long_context_and_memory.md)。

## 核心概念
**位置扩展（RoPE 拉伸家族）。** 直接让 RoPE 模型运行到训练长度之外会失败。修正方法（数学推导见[位置编码](../00_foundations/03_positional_encoding.md)）：
- **位置插值（Position Interpolation，PI）：**把位置线性缩放到训练范围内，再做简短的微调——简单有效。
- **NTK-aware / "theta" 缩放：**调整 RoPE 的基频，在不同维度上做不均匀的插值（高频维度与低频维度需要不同处理）。
- **YaRN：**NTK-by-parts + 注意力温度校正——以极少的微调把 4K 扩展到 128K；被广泛使用。
- **LongRoPE：**搜索非均匀的逐维重缩放，以达到 2M+ token。
长上下文配方还会在一个专门的训练阶段中**在长文档上退火**。混合设计绕开了部分问题：据报道，在 Kimi K3 中全局 MLA 层不使用位置编码（NoPE），位置信息交由递推的 KDA 层承载。

**注意力成本与流式处理。**
- **FlashAttention** 消除了 $O(n^2)$ 的*内存*；**环形/序列并行**把长序列分布到多个设备上（见[注意力机制](../00_foundations/01_attention_mechanisms.md)）。
- **稀疏/局部模式**（滑动窗口、空洞、全局+局部）降低计算量；**StreamingLLM** 表明，保留少量开头的"**注意力汇（attention sink）**"token + 一个最近窗口，即可在无需重新训练的情况下实现（近乎）无限的流式生成（模型会把多余的注意力倾泻到开头的 token 上）。
- **可学习的稀疏注意力**（可训练的稀疏性）：NSA（2025）、MoBA（在 KV 块上做 top-$k$），以及 **DeepSeek Sparse Attention**——一个廉价的"闪电索引器"为历史 token 打分，完整注意力只在 top-$k$ 上运行；DeepSeek-V4 在选择之前增加了 KV 压缩（CSA），外加一个重度压缩的稠密分支（HCA）。
- **混合线性注意力：**每个全注意力层配约 3 个线性/delta 规则层（Gated DeltaNet、KDA），使大多数层拥有固定大小的状态，把 KV 缓存削减约 75%。

**KV 缓存管理。** KV 缓存随上下文长度 × 批大小 × 层数线性增长，在长上下文中主导内存。缓解手段：**GQA/MLA**（更少的 KV 头 / 潜在 KV）、**KV 量化**（fp8/int8/int4）、**KV 逐出/压缩**（H2O、SnapKV——保留"重要命中"token；变换编码）、**架构层面的 KV 缩减**（混合架构、压缩稀疏注意力），以及 **PagedAttention**（虚拟内存式分页、前缀共享）。

**检索作为替代方案。** **检索增强生成（RAG）**绕开了长上下文：检索相关片段，只把它们喂给模型，以召回率换成本。长上下文与 RAG 的权衡是一个反复出现的设计问题；智能体式的**上下文工程**（通过工具即时检索、压缩（compaction）、使用干净上下文窗口的子智能体）是第三种选择。

## 关键挑战
- **迷失在中间（Liu et al. 2023）。** 模型对长上下文**开头和结尾**信息的利用远好于**中间**——呈 U 形的准确率曲线；长上下文 ≠ 均匀利用。
- **扩展与质量的权衡。** 激进的位置拉伸可能降低短上下文质量和推理能力。
- **有效上下文与名义上下文。** 一个"支持"1M token 的模型未必能在全部上下文上可靠地*推理*（通过检索式的"大海捞针"测试 ≠ 在全部上下文上的多跳推理）。
- **效率与多跳推理。** 在 MiniMax 的实验中，混合/线性注意力在标准基准上与全注意力持平，但在多跳推理上落败，这就是 MiniMax-M2 回到全注意力的原因。
- **成本。** 长上下文预填充计算量大，KV 内存也大；服务 1M token 的上下文代价高昂。
- **评测。** 大海捞针测试是必要的但不充分；更难的长上下文推理基准（RULER、LongBench、∞-Bench、MRCR）揭示了性能退化。

## 解决方案与当前最佳实践
位置扩展用 **RoPE theta 缩放 / YaRN + 长文档退火阶段**；计算用 **FlashAttention + 序列/环形并行**；内存用 **GQA/MLA + KV 量化 + 分页**。在 1M 规模下，采用**架构层面**的解决方案：在（压缩后的）KV 上做可学习的 top-$k$ 稀疏注意力（DSA/CSA、MSA），或采用约 3:1 的线性/全注意力混合（Gated DeltaNet、KDA），先在短上下文上稠密训练，在后期长上下文阶段再切换为稀疏。使用**输出门控注意力**或 **StreamingLLM 式的注意力汇**来保证稳定性和无界流式处理。用 **RULER/∞-Bench/MRCR** 和多跳任务评测，而不只做大海捞针测试。按任务决定**长上下文还是 RAG**：大型、稀疏的知识库用 RAG（更便宜、可更新）；稠密的跨文档推理用长上下文。前沿系统把长上下文、检索和压缩结合在一起。

## 各实验室观点
- **Google DeepMind：**长上下文先驱（Gemini 1.5 的 1–2M token），大海捞针测试结果强劲；架构保密。关于测试时记忆的研究（Titans、ATLAS、Nested Learning/HOPE）瞄准窗口之外的记忆。
- **Anthropic：**Claude 的长上下文（100K→1M；Opus 4.6 把 1M 带到了 Opus 上（beta），在 1M 下的 MRCR v2 8-needle 上得分 76%，而 Sonnet 4.5 为 18.5%）；强调可靠的利用和上下文工程（压缩、子智能体）。
- **OpenAI：**128K–1M（GPT-4.1），GPT-5.4 和 GPT-6 Astra 约 1.05M；把长上下文产品化。
- **DeepSeek：**用 **MLA** 提升 KV 效率；**NSA** → **DSA**（V3.2）→ **CSA/HCA**（V4，默认 1M）实现可训练的长上下文效率；DSA 被 GLM-5 采用。
- **月之暗面（Moonshot）/ Qwen / MiniMax：**MoBA 和 KDA 混合架构（Kimi Linear、1M 的 K3）；Gated DeltaNet 混合架构（Qwen3-Next、Qwen3.5）；MiniMax 从闪电注意力混合（M1）转向全注意力（M2），再到块稀疏的 MSA（M3，1M）。
- **Meta/Mistral：**开放模型中的 RoPE 缩放（Llama 3.1 128K）和滑动窗口（Mistral）方案。

## 最新进展（2023–2026年11月）
**1M–2M token** 的上下文已在前沿可用（Gemini），研究正向 10M+ 推进。**可训练的稀疏注意力**（NSA、MoBA）端到端地瞄准长上下文效率。**KV 压缩/逐出**（SnapKV、H2O）和 **MLA** 应对了内存墙。更难的**长上下文推理**基准（RULER、∞-Bench）暴露出名义上下文远大于有效上下文，使研究重新聚焦于*可用*的长上下文以及**长上下文 + 检索 + 记忆**的混合系统（见[长上下文与记忆](../09_emerging_frontiers/00_long_context_and_memory.md)）。

**2025年中 → 2026年11月更新：**
- **DeepSeek Sparse Attention（2025-09/12）。** 一个 FP8 闪电索引器（少量头、ReLU）为 MLA 选出 top-$k$ token；训练方式是先做稠密预热、用 KL 把索引器与主注意力分布对齐，再做稀疏的继续训练（[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）；GLM-5 在 744B 规模上采用了它（[arXiv:2602.15763](https://arxiv.org/abs/2602.15763)）。
- **1M 下的压缩稀疏注意力（2026-04）。** DeepSeek-V4 的 CSA 先把每个 $m$-token 的 KV 块压缩为一个条目，再由索引器做 top-$k$ 选择，而 HCA 在压缩程度高得多的 KV 上运行稠密注意力；据报道在 1M 上下文下，V4-Pro 每 token 的 FLOPs 约为 V3.2 的 27%、KV 缓存约为其 10%，并在从 4K→16K 的稠密训练之后、于 64K 阶段开启稀疏注意力（报告编号列为 arXiv:2606.19348；编号/日期不一致——请核实）。
- **混合线性注意力成为主流（2025-09/10）。** Qwen3-Next 以 3:1 混合 Gated DeltaNet 与门控注意力（32K 以上的预填充吞吐量是 Qwen3-32B 的 10 倍以上）；Kimi Linear 采用逐通道门控的 KDA，以 3:1 与 MLA 交错，报告在 1M 下 KV 缓存最多减少 75%、解码吞吐量最高 6 倍，并声称在公平比较下胜过全注意力（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）。
- **面向 1M 的块稀疏注意力（2026-06）。** MiniMax-M3 的 MSA 对 128-token 的块打分，并为每个 GQA 组保留 top-16 个块，声称在 1M 下预填充比 M2 快 9 倍、解码快 15 倍。
- **门控修复注意力汇并有助于长上下文（2025-05）。** 在 SDPA 之后加一个按头的 sigmoid 门可以消除注意力汇，并使 RULER 提升超过 10 分（[arXiv:2505.06708](https://arxiv.org/abs/2505.06708)）。
- **把上下文当作测试时学习（2025-05 → 2025-12）。** TTT-E2E 把长上下文视为持续学习——一个滑动窗口 Transformer 在测试时对上下文做梯度更新，在每 token 延迟恒定的情况下达到与全注意力相当的缩放表现，并在 128K 时加速 2.7 倍（[arXiv:2512.23675](https://arxiv.org/abs/2512.23675)）；ATLAS 报告在 10M token 的 BABILong 上提升 80%（[arXiv:2505.23735](https://arxiv.org/abs/2505.23735)）。
- **KV 压缩（2026）。** KV 变换编码（KVTC，ICLR 2026；最高 20 倍）和量化或逐出策略（ARKV，[arXiv:2603.08727](https://arxiv.org/abs/2603.08727)）报告了大幅节省（摘要层面）；DSA 式的索引器会额外引入一个"索引器 K 缓存"，使预填充/解码之间的 KV 传输变得复杂。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"如何把一个 4K 训练的模型扩展到 128K？"** RoPE PI/NTK/YaRN 的 theta 缩放 + 长文档微调；用 GQA/MLA + 量化管理 KV。
- **"什么是'迷失在中间'？"** U 形利用——模型对上下文两端的利用好于中间；名义长度 ≠ 均匀利用。
- **"长上下文与 RAG——各在何时使用？"** RAG 用于大型稀疏知识（便宜、可更新）；长上下文用于稠密的跨文档推理；常常两者结合。
- **"StreamingLLM 如何实现无限生成？"** 保留开头的注意力汇 token + 最近窗口；把多余的注意力卸载到汇上。
- **"推断时长上下文最大的成本是什么？"** KV 缓存内存（以及预填充计算）→ 量化/分页/压缩 KV，使用 GQA/MLA。
- **"要让 1M 上下文负担得起：用可学习的稀疏注意力还是线性混合？"** 稀疏（DSA/CSA）：在选中的 token 上保留全注意力的表达力，但 KV 仍需（压缩后）存储，索引器是"廉价的 $O(L^2)$"；混合（GDN/KDA 3:1）：大多数层是固定大小的状态，KV 节省巨大，但 MiniMax 发现其多跳推理较弱且状态对低精度敏感。要在多跳长上下文评测上验证，而不是只看困惑度。

## 开放问题
让**有效**上下文追上**名义**上下文——在数百万 token 上进行可靠的多跳推理，而不只是检索——仍未解决。消除"迷失在中间"、在不损失多跳质量的前提下实现高效/可训练的长上下文注意力（稀疏 vs 混合 vs 压缩），以及固定窗口之外**持久记忆**的正确架构（测试时训练、记忆层，还是 RAG 与智能体式上下文工程），仍是开放问题（见[长上下文与记忆](../09_emerging_frontiers/00_long_context_and_memory.md)）。

## 参考文献
- Chen, S. et al. (2023). *Position Interpolation.* arXiv:2306.15595.
- Peng, B. et al. (2023). *YaRN.* arXiv:2309.00071.
- Xiao, G. et al. (2023). *StreamingLLM.* arXiv:2309.17453.
- Liu, N. et al. (2023). *Lost in the Middle.* arXiv:2307.03172.
- Yuan, J. et al. (2025). *Native Sparse Attention (NSA).* arXiv:2502.11089.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
