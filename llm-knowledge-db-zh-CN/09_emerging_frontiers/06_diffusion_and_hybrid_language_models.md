# 扩散与混合语言模型（Diffusion & Hybrid Language Models）

> **最后更新：** 2026-11-04
> **相关文件：** [非 Transformer 架构](04_neuromorphic_and_non_transformer.md) · [长上下文与记忆](00_long_context_and_memory.md) · [注意力机制](../00_foundations/01_attention_mechanisms.md) · [投机解码](../06_efficiency/04_speculative_decoding.md) · [混合专家](../06_efficiency/05_mixture_of_experts.md)
> **关键论文：** Nie et al. 2025 LLaDA ([arXiv:2502.09992](https://arxiv.org/abs/2502.09992)) · inclusionAI 2025 LLaDA2.0 ([arXiv:2512.15745](https://arxiv.org/abs/2512.15745)) · Inception Labs 2025 Mercury ([arXiv:2506.17298](https://arxiv.org/abs/2506.17298)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) · DeepSeek-AI 2025 DeepSeek-V3.2 / DSA ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556))

## 概述
2023 年那一代大语言模型（LLM）的两个假设正在生产中被放宽。（1）**从左到右、一次一个 token 的生成**：*离散扩散语言模型*（dLLM）通过对掩码序列进行迭代去噪来生成，每一步预测多个 token；*多 token 预测*（multi-token prediction，MTP）则为自回归（AR）模型加入并行的未来 token 预测头。（2）**每一层都使用稠密 softmax 注意力**：*混合*模型把少数全注意力层与线性循环层（Gated DeltaNet、KDA、Mamba）混合，或者用对（压缩后的）KV 做**学习式 top-k 稀疏注意力**来替代稠密注意力。两条路线瞄准的是同一个瓶颈——解码延迟和长上下文成本——到 2026 年两者都已有 100B+ 参数的实例（LLaDA2.0-flash；Qwen3-Next/3.5、Kimi K3、DeepSeek-V4、Nemotron 3）。本文件涵盖其机制、扩展证据以及反对意见。

## 核心概念
**掩码（吸收态）扩散语言模型。** 前向过程：以概率 $t\sim U[0,1]$ 独立地掩码每个 token。一个双向 Transformer 预测所有被掩码的 token；损失 $\mathcal{L}=\mathbb{E}_{t,x_0,x_t}\big[\tfrac{1}{t}\sum_{i:\,x_t^i=[M]}-\log p_\theta(x_0^i\mid x_t)\big]$ 是负对数似然（NLL）的上界（D3PM → SEDD → MDLM 一脉）。采样从全掩码开始，在 $S$ 步内逐步*解除掩码*（通常按置信度），因此 $S\ll L$ 步即可输出 $L$ 个 token——这是速度优势的来源。**LLaDA**（8B，从头训练）表明，这种方法可以与同等规模的 AR 模型相竞争，包括在上下文学习和"反转诅咒（reversal curse）"上。

**块扩散/半自回归。** 按块从左到右生成，在每个块*内部*做扩散：恢复了 KV 缓存与可变长度，在 AR（块大小为 1）和完全扩散（块 = 整个序列）之间插值。LLaDA2.0 和 BlockGen 都利用了这一连续谱。

**AR→扩散转换。** 不从头预训练，而是转换一个预训练好的 AR 模型：先用逐渐增大块大小的块扩散做预热，再训练全序列扩散，最后衰减到小块以实现高效推断（LLaDA2.0 的类 WSD 调度）。

**混合线性注意力。** 用固定大小的循环状态替代大部分注意力层，例如门控 delta 规则 $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$（Gated DeltaNet；**KDA** 把衰减变为通道级），同时保留约四分之一的层为全 softmax 注意力（或 MLA）以实现精确召回。这样 KV 缓存只随全注意力层增长。

**学习式稀疏注意力。** 保留 softmax，但只关注一个被选出的子集：**DSA** 用一个廉价的"lightning indexer"（少量头、ReLU、FP8）为过去的 token 打分，并让 MLA 只在 top-$k$ 上计算；indexer 仍是 $O(L^2)$ 但很便宜，主注意力变为 $O(Lk)$。**CSA**（DeepSeek-V4）先把每 $m$ 个 token 的块压缩成一个 KV 条目，再选出 top-$k$ 个压缩条目；**HCA** 则在压缩程度高得多的 KV 上做稠密注意力。MiniMax 的 **MSA** 为每个 GQA 组选择 top-16 个 128-token 块。

**多 token 预测（MTP）。** 辅助头/模块预测 $t{+}2,\dots,t{+}k$ 位置的 token；既用作额外的训练信号，也用作内置的投机解码草稿（DeepSeek-V3 风格；如今已是 Qwen3-Next、GLM-4.5/5 的标配）。

## 关键挑战
- **dLLM 的质量/计算差距。** 早期的扩展研究报告称，掩码扩散需要比 AR 更多的计算才能达到同等似然，而且似然只是一个界；并行解除相互依赖的 token 的掩码会导致不连贯（即在一步内独立预测多个 token 带来的"因子分解"误差）。
- **dLLM 推断工程。** 双向注意力破坏了标准 KV 缓存；可变长度输出和提前终止都很别扭；速度优势取决于每一步能在不损失质量的前提下确定多少 token。
- **dLLM 的推理与 RL。** AR 的 RL 机制（token 对数概率、GRPO 比率）无法直接迁移，因为序列似然难以计算。
- **混合模型在多跳上失分。** MiniMax 发现线性混合模型在标准基准上与全注意力持平，但在多跳/长链推理上落后；线性状态对低精度存储敏感。
- **稀疏注意力的系统成本。** indexer 会增加一个"indexer K 缓存"，使预填充/解码分离式的 KV 传输变得复杂；训练需要一个稠密预热阶段，让 indexer 学到主注意力想要什么。

## 解决方案与当前最佳实践
**dLLM：** 从强大的 AR 检查点转换（LLaDA2.0）而非从头预训练；用**块扩散**实现缓存与可变长度；加入**置信度损失**，使模型每一步能确定多个高置信度 token；追求似然时优先掩码扩散，但在推理任务上要测试均匀状态变体。**混合模型：** 正在收敛的配方是 **约 3:1 线性:全注意力**（Qwen3-Next：12×[3 GDN + 1 门控注意力]；Kimi Linear/K3：3 KDA : 1 MLA；Nemotron：以 Mamba-2 为主，约 8% 为注意力层），使用输出门控注意力消除注意力汇聚点（sink），并对循环状态/投影采用更高精度存储。**稀疏注意力：** 分两阶段训练 DSA——先做稠密预热，用 KL 损失把 indexer 与主注意力分布对齐，再做稀疏继续训练——并与 KV 压缩（CSA/HCA）结合以支持 1M 上下文。**改造既有模型：** 通过继续训练把预训练 Transformer 转换为混合模型（MiniCPM-SALA、STILL），成本只是从头训练的一小部分。**MTP：** 保留 MTP 模块，同时用于训练信号和自投机解码。

## 各实验室观点
- **Google DeepMind：** Gemini Diffusion（2025-05，实验性，据报道与 Gemini 2.0 Flash-Lite 相当）；此后无公开更新（二手来源）。
- **Inception Labs：** 首批商用 dLLM（Mercury Coder Mini/Small）；Mercury 2（2026-02）号称首个扩散推理语言模型，在 Blackwell 上约 1,000 tok/s（自报）。
- **inclusionAI / 蚂蚁：** LLaDA 一脉；LLaDA2.0 把 AR→扩散转换扩展到 100B MoE。
- **Qwen / Moonshot：** 生产级线性混合模型（Qwen3-Next、Qwen3.5；Kimi Linear、以 KDA+MLA 3:1 构建的 Kimi K3）。
- **DeepSeek / 智谱：** 学习式稀疏注意力（V3.2 中的 DSA，被 GLM-5 采用；V4 中的 CSA+HCA）。
- **MiniMax：** M1 用混合架构 → M2 用全注意力 → M3 用块稀疏 MSA——这是业内最明确的一次公开回调。
- **NVIDIA：** Mamba-2/Transformer(/MoE) 混合模型（Nemotron-H、Nano 2、Nemotron 3），连同数据和配方一并发布。

## 最新进展（2023–2026年11月）
离散扩散从 D3PM 发展到 SEDD 和 MDLM（2023–24），随后 **LLaDA**（2025-02）把掩码扩散扩展到 8B。Mamba/Mamba-2 和 Gated DeltaNet 使线性循环层具备竞争力，Jamba/Nemotron-H 证明混合架构可以规模化；DeepSeek-V3 使 MTP 成为主流，NSA/MoBA 则使稀疏注意力变得可训练。

**2025 年中 → 2026 年 11 月更新：**
- **商用及 100B 规模的 dLLM（2025-05 → 2026-02）。** Gemini Diffusion（I/O 2025，二手来源）；Mercury 每步并行预测多个 token（[arXiv:2506.17298](https://arxiv.org/abs/2506.17298)）；**LLaDA2.0** 通过块扩散预热 → 全序列 → 小块衰减，把 AR MoE 模型转换为 16B（mini）和 100B（flash）规模的扩散模型，并用辅助置信度损失实现激进的并行解码（[arXiv:2512.15745](https://arxiv.org/abs/2512.15745)）。
- **扩散缩放定律（2026-02）。** "Scaling Beyond Masked Diffusion LMs"将掩码、均匀状态和插值扩散与 AR 进行比较：在扩散变体中掩码扩散的似然扩展性最好，但均匀状态扩散尽管困惑度更差，在 GSM8K 上表现更好（[arXiv:2602.15014](https://arxiv.org/abs/2602.15014)）——困惑度并不是衡量 dLLM 的合适标尺。BlockGen（2026-06）在混合块大小上训练，使似然在 AR 与扩散之间平滑插值（单一来源）。
- **线性混合模型进入生产（2025-09 → 2026-07）。** Qwen3-Next 80B-A3B（3 GDN : 1 门控注意力；32K 以上相对 Qwen3-32B 预填充吞吐量 >10×）；**Kimi Linear**（KDA，48B-A3B）声称在包括 RL 在内的公平比较下胜过全注意力，在 1M 下 KV 最多减少 75%、解码提速 6×（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）；**Kimi K3** 把 KDA+MLA 3:1 扩展到 2.8T/激活 104B（69 个 KDA + 24 个 MLA 层，据报道 MLA 层不使用位置编码（NoPE））。
- **MiniMax 回归全注意力（2025-10）。** M2 放弃了 M1 的 lightning attention 混合（多跳损失、低精度状态敏感、基础设施不成熟）；M3（约 2026-06）随后采用块稀疏 MSA，声称在 1M 下相对 M2 预填充提速 9×、解码提速 15×。
- **DSA → CSA/HCA（2025-09 → 2026-04）。** DeepSeek-V3.2-Exp 通过匹配 V3.1-Terminus 的训练配置来单独考察 DSA 的效果（质量大致持平，API 降价 >50%）；GLM-5 在 744B 规模上采用 DSA（[arXiv:2602.15763](https://arxiv.org/abs/2602.15763)）；据报道 DeepSeek-V4（1.6T/激活 49B）在 1M 下每 token FLOPs 约为 V3.2 的 27%、KV 约为其 10%（DeepSeek-V4 技术报告：arXiv:2606.19348，据其 Hugging Face 模型卡）。
- **SSM 混合模型（2025-08 → 2026-03）。** Nemotron Nano 2（28 Mamba-2 + 28 FFN + 6 注意力；相对 Qwen3-8B 吞吐量最高 6.3×；[arXiv:2508.14444](https://arxiv.org/abs/2508.14444)）；Nemotron 3 Mamba-2/MoE 家族，其中 Super 以 NVFP4 预训练（[arXiv:2512.20856](https://arxiv.org/abs/2512.20856)）；**Mamba-3** 加入复数值状态转移和 MIMO 解码（[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)）。
- **超越"预测后 k 个 token"的 MTP（2025-10 → 2026-02）。** Future Summary Prediction 预测长程未来的紧凑摘要，在 3B/8B 规模上优于 NTP 和 MTP（[arXiv:2510.14751](https://arxiv.org/abs/2510.14751)）；自蒸馏把 NTP 模型转换为独立的 MTP 解码器，在 GSM8K 上提速 >3×，准确率损失 <5%（[arXiv:2602.06019](https://arxiv.org/abs/2602.06019)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"写出掩码扩散的训练损失，并解释 $1/t$ 权重。"** 吸收态噪声下的 ELBO；每个被掩码位置贡献的交叉熵按掩码率的倒数加权。
- **"为什么扩散语言模型的解码可能更快，又受什么限制？"** 每个去噪步可确定多个 token；受限于同时解除掩码的 token 之间的依赖、缺乏 KV 缓存（块扩散可解决），以及保证质量所需的步数。
- **"Qwen3-Next vs DeepSeek-V3.2 vs MiniMax-M2——对长上下文的三种回答，请比较。"** 线性混合（有界状态）vs 学习式 top-k 稀疏 softmax vs 全注意力；KV、FLOPs、多跳鲁棒性、精度敏感性、kernel 成熟度。
- **"DSA 的 indexer 是如何训练的？"** 先做稠密预热，用 KL 损失把 indexer 打分与主注意力分布对齐，然后用 top-$k$ 做稀疏训练。
- **"MTP vs 投机解码 vs 扩散？"** MTP = 带额外预测头的 AR（自草稿，通过验证保持精确）；扩散 = 非自回归的并行去噪（相对于任何 AR 分布都不是精确的）。

## 开放问题
dLLM 能否在前沿规模上弥合质量差距（以及推理/RL 能否迁移到 dLLM）、当困惑度具有误导性时该用什么标尺、如何为扩散模型做精确似然的 RL、约 3:1 混合与稀疏 softmax 哪个更能保留长链推理（Kimi vs MiniMax）、面向 1M+ 上下文的最优压缩/选择层级，以及这些设计如何与 FP4 训练和分离式服务相互作用，都是开放问题。

## 参考文献
- Austin, J. et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).* arXiv:2107.03006.
- Lou, A. et al. (2023). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD).* arXiv:2310.16834.
- Sahoo, S. et al. (2024). *Simple and Effective Masked Diffusion Language Models (MDLM).* arXiv:2406.07524.
- Nie, S. et al. (2025). *Large Language Diffusion Models (LLaDA).* arXiv:2502.09992.
- Arriola, M. et al. (2025). *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.* arXiv:2503.09573.
- Inception Labs (2025). *Mercury: Ultra-Fast Language Models Based on Diffusion.* arXiv:2506.17298.
- inclusionAI (2025). *LLaDA2.0.* arXiv:2512.15745.
- Multiple authors (2026). *Scaling Beyond Masked Diffusion Language Models.* arXiv:2602.15014.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- NVIDIA (2025). *Nemotron Nano 2.* arXiv:2508.14444.
- Gloeckle, F. et al. (2024). *Better & Faster Large Language Models via Multi-token Prediction.* arXiv:2404.19737.
