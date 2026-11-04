# 非 Transformer 架构（Non-Transformer Architectures）

> **最后更新：** 2026-11-04
> **相关文件：** [注意力机制](../00_foundations/01_attention_mechanisms.md) · [扩散与混合语言模型](06_diffusion_and_hybrid_language_models.md) · [测试时训练](03_test_time_training.md) · [长上下文与记忆](00_long_context_and_memory.md)
> **关键论文：** Gu & Dao 2023 Mamba ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) · Dao & Gu 2024 Mamba-2 / SSD ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)) · Peng et al. 2023 RWKV ([arXiv:2305.13048](https://arxiv.org/abs/2305.13048)) · Sun et al. 2023 RetNet ([arXiv:2307.08621](https://arxiv.org/abs/2307.08621)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) · Mamba-3 2026 ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569))

## 概述
Transformer 的 $O(n^2)$ 注意力和庞大的 KV 缓存，促使人们寻找质量与之相当的**次二次复杂度**序列模型。主要挑战者是**状态空间模型（SSM）**——尤其是 **Mamba**——以及各种**线性注意力**变体（RWKV、RetNet、Hyena）。它们追求的"圣杯"是：**达到 Transformer 的质量，同时具备类 RNN 的线性时间推断和恒定内存**（KV 缓存不随长度增长）。在线性注意力方法多年表现不佳之后，Mamba（2023）弥合了大部分质量差距，重新点燃了这一领域；2024–2026 年，注意力-SSM / 注意力-线性的**混合**模型进入生产——到 2026 年，多款开放*旗舰*模型（Qwen3.5、Kimi K3、Nemotron 3）都是混合架构。理解 SSM 的形式化、注意力与 SSM 的权衡以及混合架构为何占主导，越来越成为必备知识。

## 核心概念
**状态空间模型（SSM）。** 连续线性系统 $h'(t)=Ah(t)+Bx(t)$，$y(t)=Ch(t)$，离散化为递推 $h_t=\bar A h_{t-1}+\bar B x_t$，$y_t=\bar C h_t$。关键的对偶性：它既可以作为**递推**计算（线性时间、恒定内存的推断，如同 RNN），**也**可以作为**卷积**计算（可并行训练）。早期 SSM（**S4**）利用巧妙的结构化/HiPPO 初始化来捕获长程依赖。

**Mamba（选择性 SSM）。** 关键突破：让 SSM 参数 $(\bar B, \bar C, \Delta)$ **依赖于输入**（"选择性"），使模型能够根据内容*选择*记住或遗忘什么——克服了线性时不变 SSM 的关键弱点（它们无法像注意力那样做基于内容的推理）。代价是失去卷积视角，因此 Mamba 使用**硬件感知的并行扫描**来高效训练。Mamba 在中小规模上以线性时间、恒定内存的推断匹配了 Transformer。

**Mamba-2 / SSD。** 建立了理论桥梁（"**状态空间对偶性**"），表明 SSM 与（某种形式的）注意力密切相关，从而实现更快、更硬件友好的实现以及更大的状态规模。

**线性注意力家族。**
- **RWKV：** 一种 RNN/Transformer 混合体，将线性注意力重新表述以支持并行训练和循环推断；"可扩展的 RNN"，RWKV-7 加入了表达力强的动态状态。
- **RetNet：** "保留（retention）"机制，具有并行（训练）、循环（推断）和分块三种形式。
- **Hyena：** 用长隐式卷积 + 门控替代注意力（次二次复杂度）。
- 它们的共同目标是 $O(n)$ 推断；近期变体（**DeltaNet、Gated DeltaNet**）加入 delta 规则/门控更新以增强联想召回，与 TTT 相互重叠（见[测试时训练](03_test_time_training.md)）。**Kimi Delta Attention（KDA）**用通道级（而非头级）门控改进了 Gated DeltaNet——概念上，标量的逐头衰减 $\alpha_t$ 变成逐通道的对角阵 $\mathrm{Diag}(\alpha_t)$——从而更精细地控制每个状态维度遗忘什么。

**Mamba-3。** 以推断为先的重新设计：指数-梯形离散化、**复数值**状态转移（恢复了实数、非负衰减所缺乏的状态追踪能力），以及在 GPU 上提高解码算术强度的 **MIMO** 更新。

**根本性权衡。** 注意力保留*所有* token（完美召回，二次成本）；SSM 把历史压缩成*固定大小的状态*（线性成本，有损召回）。SSM 擅长长序列和高吞吐量，但在**精确召回/复制/上下文内检索**方面**较弱**——而这恰恰是注意力最擅长的。

## 关键挑战
- **召回/复制弱点。** 固定状态模型在需要从长上下文中精确检索的任务上表现吃力（联想召回、大海捞针测试）——这正是注意力的强项。
- **上下文学习差距。** 某些上下文学习（ICL）/归纳头行为对纯 SSM 而言更难。
- **规模化验证。** 大多数决定性比较都在中小规模上进行；前沿规模上的对等性尚未充分确立（而且各实验室仍在使用注意力或混合架构）。
- **生态成熟度。** 数十年来对注意力的优化（FlashAttention、kernel、工具链）使 Transformer 占优；SSM 的 kernel/工具链较新。
- **混合设计。** 注意力:SSM 的比例和放置位置依赖经验；据报道 GDN 混合模型中的大幅激活（massive activations）聚集在全注意力层之前（单一来源）。
- **数值问题。** 循环状态对低精度存储敏感（这是 MiniMax 在 M2 中放弃线性注意力的原因之一）；即使在 NVFP4 预训练下，Nemotron 3 也把 Mamba 输出投影保留在 MXFP8。
- **长链推理。** 混合模型在标准基准上可以与全注意力持平，但在多跳/长链推理上落后（MiniMax M2 报告）。

## 解决方案与当前最佳实践
**实践中混合架构胜出。** 将*少量*全注意力层（负责召回/ICL）与大量 SSM/线性层（负责效率）交错排列——例如 **Jamba**（Mamba-Transformer-MoE）、**Zamba**、**Nemotron-H / Nano 2 / Nemotron 3**、**Samba**，以及 2025–26 年的生产配方 **约 3:1 线性:全注意力**（Qwen3-Next/3.5：Gated DeltaNet + 门控注意力；Kimi Linear/K3：KDA + MLA）。这样在关键位置保留注意力的召回能力，其他位置则获得 SSM 的吞吐量/长上下文效率。对于召回需求不高的纯长序列吞吐场景，Mamba-2/3 和 RWKV-7 表现强劲。另一条与之竞争的次二次路线保留 *softmax* 注意力但使其**稀疏化**（学习式 top-k：DSA/CSA、MiniMax MSA）——见[扩散与混合语言模型](06_diffusion_and_hybrid_language_models.md)。预训练好的 Transformer 也可以**通过继续训练实现线性化/混合化**（MiniCPM-SALA、STILL），成本只是从头训练的一小部分。纯 SSM 在前沿仍属小众；闭源实验室不披露架构（gpt-oss 作为 OpenAI 唯一公开的设计，采用滑动窗口/稠密注意力交替并带有学习式注意力汇聚点（sink））。

## 各实验室观点
- **学术界（CMU/Princeton——Gu & Dao）：** Mamba/Mamba-2，领先的 SSM 路线。
- **AI21：** **Jamba**——首个生产规模的 SSM-Transformer-MoE 混合模型。
- **NVIDIA：** Nemotron-H → Nemotron Nano 2（仅约 8% 为注意力层）→ **Nemotron 3**（Mamba-2 + MoE + 少量注意力层；Nano 30B-A3B，Super 120B-A12B 以 NVFP4 预训练），连同数据和配方完全开放发布。
- **Qwen / Moonshot：** 生产级线性混合模型（Qwen3-Next、Qwen3.5；Kimi Linear、Kimi K3）。
- **MiniMax：** 反例——M1 采用 lightning attention 混合，M2 **回到全注意力**，M3 再采用块稀疏注意力（MSA）。
- **RWKV/开源社区：** RWKV（线性注意力 RNN），广泛的开放生态并关注端侧。
- **美国前沿实验室（OpenAI/Anthropic/Google/Meta）：** 闭源模型的架构未公开；公开信号以注意力为中心（gpt-oss 的滑动窗口 + sink），其中 Google 发表的循环记忆研究最多（Titans/ATLAS/HOPE）。

## 最新进展（2023–2026年11月）
**Mamba（2023）**复兴了这一领域；**Mamba-2/SSD** 在理论上统一了 SSM 与注意力。**混合架构**（Jamba、Zamba、Nemotron-H、Samba）成为务实的最佳平衡点并达到生产规模。**表达性循环状态**（Gated DeltaNet、RWKV-7、Titans）使 SSM 与**测试时学习/快速权重**之间的界限变得模糊。共识逐渐清晰：**是混合，而非纯粹替代**——注意力负责召回，SSM 负责效率——尽管最优配比以及纯 SSM 能否在前沿完全匹敌注意力仍是开放问题。

**2025 年中 → 2026 年 11 月更新：**
- **Nemotron Nano 2（2025-08）→ Nemotron 3（2025-12/2026-03）。** 62 层 = 28 个 Mamba-2 + 28 个 FFN + 仅 6 个注意力层；以 FP8 训练 20T token；在推理负载上吞吐量最高为 Qwen3-8B 的 6.3×（[arXiv:2508.14444](https://arxiv.org/abs/2508.14444)）。Nemotron 3 将 Mamba-2 与 MoE 交错（[arXiv:2512.20856](https://arxiv.org/abs/2512.20856)）；Super（120B-A12B）以 NVFP4 在约 25T token 上预训练（[arXiv:2604.12374](https://arxiv.org/abs/2604.12374)）。
- **Qwen3-Next（2025-09）与 Qwen3.5（2026-02）。** 48 层 = 12 × [3 个 Gated DeltaNet + 1 个门控注意力]，用于一个 80B-A3B、512 专家的 MoE；在 32K 以上相对 Qwen3-32B 预填充吞吐量 >10×。据报道 Qwen3.5-397B-A17B 保留了 GDN 混合结构（无技术报告；二手来源）。
- **Kimi Linear（2025-10）→ Kimi K3（2026-07）。** KDA 与 MLA 以 3:1 交错（48B-A3B）——声称是首个在短上下文、长上下文和 RL 三种情形的公平比较中胜过全注意力的线性混合模型，在 1M 下 KV 最多减少 75%、解码提速 6×（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）；K3 将其扩展到总参数 2.8T / 激活 104B，含 69 个 KDA 层 + 24 个 MLA 层。
- **MiniMax 的异议（2025-10）。** M2 放弃了 M1 的 lightning attention 混合：在标准基准上持平，但在多跳/长链推理上更差，对低精度状态敏感，基础设施不成熟（MiniMax 博客）——这是面试中的经典反方论据。
- **Mamba-3（2026-03，ICLR 2026 oral）。** 复数值状态转移、指数-梯形离散化、MIMO 解码；以一半的状态规模达到 Mamba-2 的困惑度（[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)）。
- **改造既有模型（2026-02）。** MiniCPM-SALA 通过持续训练把预训练 Transformer 转换为 1:3 稀疏:线性混合（比从头训练便宜约 75%；二手来源），STILL 则在层内把显著 token 路由到 softmax、其余由线性注意力处理，仅需约 0.04B 训练 token（[arXiv:2602.02180](https://arxiv.org/abs/2602.02180)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"相对于 Transformer，SSM/Mamba 解决了什么问题？"** 针对长序列的次二次、恒定内存推断（KV 缓存不增长）。
- **"是什么让 Mamba 在早期 SSM 失败的地方取得成功？"** 依赖输入（"选择性"）的参数实现基于内容的记忆/遗忘；硬件感知的并行扫描。
- **"SSM 与注意力的根本权衡是什么？"** 固定状态压缩（线性、有损召回）vs 保留所有 token（二次、完美召回）。
- **"为什么混合架构占主导？"** 少量注意力层恢复召回/ICL；SSM 层提供效率。
- **"SSM 在哪些方面吃力？"** 精确复制/联想召回/大海捞针。
- **"线性混合（Kimi Linear）vs 稀疏注意力（DSA）vs 全注意力（MiniMax M2）——为一个 1M 上下文的智能体模型你会选哪个？"** 有界状态与 3:1 配比 vs 在压缩 KV 上精确但经过选择的 softmax vs 简单性与多跳鲁棒性；讨论 KV 成本、精度敏感性、kernel 成熟度和 RL 行为。

## 开放问题
**纯**次二次架构能否在**前沿规模上的召回/ICL** 方面匹敌注意力，尚无定论。最优**混合配比**（为何约 3:1 反复出现）、混合架构是否会削弱长链推理（MiniMax）或不会（Kimi）、弥合召回差距、成熟的 kernel/工具链，以及连接 SSM、注意力、快速权重与测试时学习的更深层理论，都是活跃的研究方向。该领域近期的可能答案是*高效混合*，但一个干净的 Transformer 继任者仍是悬而未决的大奖。

## 参考文献
- Gu, A., Dao, T. (2023). *Mamba.* arXiv:2312.00752.
- Dao, T., Gu, A. (2024). *Mamba-2 / State Space Duality.* arXiv:2405.21060.
- Peng, B. et al. (2023). *RWKV.* arXiv:2305.13048.
- Sun, Y. et al. (2023). *RetNet.* arXiv:2307.08621.
- Lieber, O. et al. (2024). *Jamba.* arXiv:2403.19887.
- NVIDIA (2025). *Nemotron Nano 2.* arXiv:2508.14444.
- NVIDIA (2025). *Nemotron 3.* arXiv:2512.20856.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- CMU/Princeton/Together AI (Gu, Dao et al.) (2026). *Mamba-3.* arXiv:2603.15569.
