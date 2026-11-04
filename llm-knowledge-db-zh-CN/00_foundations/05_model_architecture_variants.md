# 模型架构变体（Model Architecture Variants）

> **最后更新：** 2026-11-04
> **相关文件：** [Transformer 架构](00_transformer_architecture.md) · [混合专家](../06_efficiency/05_mixture_of_experts.md) · [注意力机制](01_attention_mechanisms.md)
> **关键论文：** Chowdhery et al. 2022 PaLM（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)）· Touvron et al. 2023 LLaMA 2（[arXiv:2307.09288](https://arxiv.org/abs/2307.09288)）· Jiang et al. 2024 Mixtral（[arXiv:2401.04088](https://arxiv.org/abs/2401.04088)）· DeepSeek-AI 2024 DeepSeek-V3（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）

## 概述
尽管几乎所有前沿大语言模型（LLM）都共享仅解码器的 Transformer 骨架，它们却在一组定义清晰的"旋钮"上各不相同：归一化的类型与位置、激活函数、注意力变体、位置方案、稠密 FFN 与混合专家（MoE）FFN，以及各种稳定性技巧。这些选择并非表面文章——它们实质性地影响训练稳定性、推断（inference）成本和单位 FLOP 的质量。本文梳理这一设计空间和已收敛的"现代配方"，再对照各旗舰模型家族如何具体实现。

2020→2026 年的故事是：先逐步标准化（前置归一化 + RMSNorm + RoPE + SwiGLU + GQA），随后在效率驱动下重新走向多样化：MoE FFN、潜在注意力（MLA）、注意力–状态空间模型（SSM）与线性注意力的混合、可学习的稀疏注意力，以及——到 2026 年——采用 4 比特权重的万亿参数开放 MoE。

## 核心概念
**归一化。** 原始 Transformer 使用**后置归一化**的 LayerNorm（在深层时不稳定）。现代模型使用**前置归一化**（$x+\text{Sublayer}(\text{Norm}(x))$）以稳定梯度，并几乎总是配合 **RMSNorm**（不做均值相减）。一些超大规模训练还会加入 **QK-norm**（在注意力之前对查询/键做归一化）和**三明治/双重归一化**以提升稳定性。

**激活函数。** ReLU → GeLU（BERT/GPT-2/3）→ **门控线性单元**，尤其是 **SwiGLU**（LLaMA、PaLM、Mistral）。门控激活一致地提升了每参数质量；由于多了一个门控矩阵，FFN 隐藏维度降为约 $\frac{8}{3}d$ 以保持参数量不变。

**注意力变体。** MHA → **MQA**（PaLM）→ **GQA**（LLaMA-2 70B+、Mistral），用于削减 KV 缓存；**滑动窗口**（Mistral）；**MLA**（DeepSeek）用于低秩 KV 压缩。见[注意力机制](01_attention_mechanisms.md)。

**稠密 FFN 与 MoE FFN。** 稠密模型（dense model）对每个 token 激活全部参数。**稀疏 MoE** 用 $E$ 个专家（expert）替换 FFN，并由路由器（router）选出 top-$k$（Mixtral 时代的设计中为 2；2025–26 年的细粒度设计中为数百个专家里选 8–16 个），因此激活参数 $\ll$ 总参数。Mixtral 8×7B（总参数 47B，约 13B 激活）、DeepSeek-V3（总参数 671B，37B 激活）、Llama 4（MoE）、Kimi K2（1T/32B，384 个专家，激活 8 个）以及 Kimi K3（2.8T/104B，896 个专家，激活 16 个 + 2 个共享专家）都是例子。MoE 把容量与计算解耦——见 [MoE](../06_efficiency/05_mixture_of_experts.md)。

**并行层。** PaLM 从*同一个*归一化输入出发并行计算注意力和 FFN（而不是顺序计算），省掉一次归一化并提高 TPU 利用率。

**深度与宽度。** 在给定参数预算下，纵横比（层数与 $d_{model}$）在表达力与并行性之间权衡；更深更窄有利于推理（reasoning），但损害硬件效率和稳定性。

## 关键挑战
- **大规模下的稳定性。** 某些选择（后置归一化、某些激活函数、大学习率）会引起损失尖峰（loss spike）；各种缓解手段之间存在复杂的相互作用。
- **MoE 的不稳定与负载失衡。** 路由器可能坍缩到少数几个专家；需要辅助负载均衡（load balancing）损失或无损失均衡。
- **推断最优与训练最优相背离。** GQA/MQA/MoE 有利于推断却使训练复杂化；最好的训练架构不等于最好的服务架构。
- **可复现性。** 许多设计选择只由未发表的内部消融实验支撑。

## 解决方案与当前最佳实践
**2026 年稠密默认配方**：仅解码器、前置 RMSNorm、SwiGLU、RoPE、GQA、无偏置、大规模下不共享嵌入、AdamW（矩阵参数越来越多地改用 Muon）+ WSD/余弦调度。**MoE 默认配方**（2026）：细粒度专家（128–896 个）加 1–2 个共享专家、top-4 到 top-16 路由、无辅助损失或低系数的均衡、专家/张量并行、一个 MTP 头，并越来越多地采用 Muon（配合 QK-Clip）以及 4 比特专家权重（MXFP4/NVFP4/INT4 QAT）。token 混合器如今按层选择：注意力–Mamba 混合块（Jamba、Zamba、Nemotron-H/Nemotron 3）、约 3:1 的 delta 规则混合，或可学习的稀疏注意力。

| 模型（日期） | 总参数 / 激活参数 | 专家（路由+共享） | token 混合器 |
|---|---|---|---|
| gpt-oss-120b（2025-08） | 116.8B / 5.1B | 128，top-4 | GQA；带状/稠密交替；可学习的注意力汇 |
| Kimi K2（2025-07） | ~1T / 32B | 384，top-8 | MLA |
| Qwen3-Next（2025-09） | 80B / ~3B | 512（10+1） | 3 Gated DeltaNet : 1 门控注意力 |
| MiniMax-M2（2025-10） | ~230B / 10B | 256 | 全注意力 |
| GLM-5（2026-02） | 744B / 40B | — | DSA（DeepSeek Sparse Attention） |
| DeepSeek-V4-Pro（2026-04） | 1.6T / 49B | — | CSA + HCA（压缩稀疏注意力） |
| Gemma 4（2026-04） | 26B-A4B MoE；31B 稠密 | — | — |
| Kimi K3（2026-07） | 2.8T / 104B | 896（16+2） | 69 KDA : 24 门控 MLA（3:1） |
| Qwen3.8 开放版（2026-08） | 2.4T / 95B | 512（10+1） | — |
| Mistral Large 4（2026-10） | ~1.05T / 52B | — | — |

（—：未公开说明或未经核实。多数数字为厂商自报；部分来自二手来源。）

## 各实验室观点
- **OpenAI：** 前沿模型闭源；GPT-4 被普遍认为是 MoE；o 系列加入了长推理 rollout（采样轨迹）。gpt-oss（Apache 2.0）是其开放参考：128 个专家、top-4 的 MoE，专家权重为 MXFP4（约 4.25 比特/参数），因此 120b 可以放进一张 80 GB GPU。
- **Google DeepMind：** Gemini 是原生多模态（multimodal）的 MoE；Gemma 1–3 遵循简洁的稠密配方（GQA + RoPE），Gemma 4（Apache 2.0）则在 31B 稠密模型之外增加了一个 26B-A4B 的 MoE，以及采用逐层嵌入的端侧模型。
- **Meta：** Llama 1–3 为稠密模型（树立标准的配方）；Llama 4 转向 MoE（Scout/Maverick/Behemoth）；自 2026 年 4 月起，其前沿的 Muse 模型为闭源，架构未公开。
- **DeepSeek：** 独特的 MLA + 细粒度 MoE + 多 token 预测（MTP）训练目标，之后是 DSA（V3.2）和 CSA/HCA（V4，1M 上下文），并采用 Muon 以及对路由专家的 FP4 QAT（据报道）。
- **Moonshot / Qwen / 智谱 / MiniMax：** 中国的开放前沿如今在规模和注意力设计上领先——KDA 混合（Kimi）、配合超稀疏 512 专家 MoE 的 Gated DeltaNet 混合（Qwen）、采用 DSA（GLM-5），以及明确回归全注意力（MiniMax M2）后再转向块稀疏 MSA（M3）。
- **NVIDIA：** Nemotron 3 是只含少量注意力层的 Mamba-2/MoE 混合模型，以 NVFP4 预训练（Super、Ultra），并连同数据和配方一起发布。
- **Mistral：** 滑动窗口 + GQA 的稠密模型（7B）和 MoE（Mixtral）；效率优先。Mistral Large 3（675B/41B 的"细粒度"MoE）和 Large 4（约 1.05T/52B，2026 年 10 月开放权重）是西方最大的开放 MoE；Medium 3.5 则是一个 128B 的*稠密*模型。

## 最新进展（2023–2026年11月）
MoE 已成为前沿主流（DeepSeek-V3、Llama 4、Qwen-MoE、Mixtral）。**多 token 预测**（预测多个未来 token）被用作辅助目标（DeepSeek-V3、Meta 的研究），以提升数据效率，并作为投机解码（speculative decoding）的基础。**注意力–SSM 混合**面向长上下文。**MLA** 作为一种 KV 高效的 GQA 替代方案得到推广。归一化研究（QK-norm、OLMo-2 的重排归一化）继续追求耗资数十亿美元的训练的稳定性。

**2025年中 → 2026年11月更新：**
- **开放 MoE 跨越 1T，稀疏度持续上升（2025-07 → 2026-10）。** Kimi K2（1T/32B；arXiv:2507.20534）→ DeepSeek-V4-Pro（1.6T/49B）→ Kimi K3（2.8T/104B）→ Qwen3.8（2.4T/95B）→ Mistral Large 4（约 1.05T/52B）。激活参数保持在约 10–104B，而专家数量增长到 512–896，路由为 top-10/16；Qwen3-Next 在 80B 中只激活约 3B。
- **三类相互竞争的 token 混合器。** (i) 约 3:1 的 delta 规则混合（Qwen3-Next、Qwen3.5、Kimi Linear arXiv:2510.26692、K3）；(ii) 可学习的 top-$k$ 稀疏注意力（DSA arXiv:2512.02556 → V4 的 CSA/HCA；GLM-5 arXiv:2602.15763；MiniMax M3 的 MSA，以 128 token 为块，每个 GQA 组取 top-16）；(iii) 滑动窗口混合（gpt-oss；据报道 MiMo-V2.5 为 5:1 的 SWA:全局）。MiniMax M2 回归全注意力（多跳推理受损、低精度状态脆弱）仍是一个有力的反例。
- **生产规模的 SSM 混合模型。** Nemotron Nano 2（2025-08；arXiv:2508.14444）：62 层 = 28 个 Mamba-2 + 28 个 FFN + 6 个注意力层，以 FP8 训练 20T token，从 12B 剪枝到 9B，在推理负载上吞吐量（throughput）最高为 Qwen3-8B 的 6.3 倍。Nemotron 3（arXiv:2512.20856）交错 Mamba-2 与 MoE；Mamba-3（arXiv:2603.15569，ICLR 2026 口头报告）加入复值状态转移和 MIMO 解码，以一半的状态大小达到 Mamba-2 的困惑度。
- **稳定性转移到门控、优化器和残差几何上。** 输出门控注意力（arXiv:2505.06708）消除了注意力汇；MuonClip 中的 QK-Clip 防止 logit 爆炸（K2：15.5T token 训练零损失尖峰）；mHC（arXiv:2512.24880）把拓宽的残差流约束为双随机混合；K3 将稳定性归功于"SiTU-GLU""Quantile Balancing"和 Attention Residuals（据报道，未经独立验证）。
- **数值格式如今是一项架构选择。** gpt-oss 以 MXFP4 发布 MoE 权重；Kimi K2 Thinking 采用仅权重的 INT4 QAT（生成速度约提升 2 倍）；据报道 DeepSeek-V4 对路由专家做 FP4 QAT；K3 以 MXFP4 发布权重；Nemotron 3 Super/Ultra 直接以 NVFP4 *预训练*（arXiv:2604.12374）。
- **非自回归变体。** LLaDA2.0（arXiv:2512.15745）把自回归 MoE 转换为最大 100B 的块扩散语言模型，采用"预热 → 全序列 → 小块"的调度；一项缩放对比研究（arXiv:2602.15014）发现，在各种扩散变体中，掩码扩散的似然缩放最好，但均匀状态扩散在 GSM8K 上表现更佳。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"'现代 LLM 配方'包含什么，为什么？"** 前置 RMSNorm、SwiGLU、RoPE、GQA——逐一说明理由。
- **"稠密还是 MoE：何时用、为什么？"** 把容量与激活计算解耦；服务吞吐量与内存占用、路由复杂度之间的权衡。
- **"MLA 是什么，与 GQA 有何不同？"** 低秩潜在 KV 压缩 vs 共享头。
- **"PaLM 为什么采用注意力+FFN 并行层？"** 更少的归一化、更好的硬件利用率，大规模下质量损失可忽略。
- **"为什么开放 MoE 从 8 个专家/top-2 走到了 512–896 个专家/top-10–16？"** 更细的粒度在每 FLOP 上提供更多专家组合和更好的专精；共享专家吸收通用知识；代价是 all-to-all 通信、路由稳定性以及总参数所需的内存（机架级 NVLink 域）。
- **"说出 2026 年实现可负担 1M token 上下文的三条路线及其失效模式。"** 3:1 线性混合（召回/多跳）、在压缩 KV 上做可学习稀疏（索引器开销、额外缓存）、SWA 混合（局部层的长程信息丢失）。

## 开放问题
这些选择大多缺乏第一性原理的理论——它们是靠消融实验胜出的。开放问题包括：MoE 中最优的专家粒度与路由（以及专家:注意力的计算分配，arXiv:2603.10379）；混合 SSM/线性注意力设计会取代纯注意力，还是会在多跳推理上遇到瓶颈；正确的深度/宽度规律；4 比特权重能走多远而不损失质量；以及如何在不依赖启发式归一化的情况下让数万亿参数的训练保持稳定。"最佳架构"很可能取决于部署场景，而业界才刚刚开始将其形式化。

## 参考文献
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.
- Lieber, O. et al. (2024). *Jamba.* arXiv:2403.19887.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- NVIDIA (2025). *NVIDIA Nemotron Nano 2.* arXiv:2508.14444.
- DeepSeek-AI (2025). *DeepSeek-V3.2*（DeepSeek Sparse Attention）. arXiv:2512.02556.
- Zhipu AI / Z.ai (2026). *GLM-5.* arXiv:2602.15763.
- Gu, A., Dao, T. et al. (2026). *Mamba-3.* arXiv:2603.15569.
