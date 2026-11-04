# 预训练目标（Pretraining Objectives）

> **最后更新：** 2026-11-04
> **相关文件：** [Transformer 架构](../00_foundations/00_transformer_architecture.md) · [信息论基础](../00_foundations/06_information_theory_basics.md) · [模型架构变体](../00_foundations/05_model_architecture_variants.md)
> **关键论文：** Radford et al. 2018 GPT（CLM）· Devlin et al. 2018 BERT（MLM）（[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)）· Raffel et al. 2020 T5（span corruption）（[arXiv:1910.10683](https://arxiv.org/abs/1910.10683)）· Tay et al. 2022 UL2（[arXiv:2205.05131](https://arxiv.org/abs/2205.05131)）· Bavarian et al. 2022 FIM（[arXiv:2207.14255](https://arxiv.org/abs/2207.14255)）

## 概述
预训练（pretraining）目标决定了模型从无标注文本中提取*什么信号*。对于生成式大语言模型（LLM），业界已压倒性地收敛到**因果语言建模（causal language modeling，CLM）**——即自回归的下一个 token 预测——因为它是一个通用的自监督任务，扩展起来干净利落，支持上下文学习，并把预训练与生成统一了起来。但整个设计空间（掩码语言模型、前缀语言模型、片段破坏、中间填充、多 token 预测、离散扩散）仍值得理解：一方面出于历史原因，另一方面也因为其中的若干元素正重新进入前沿训练配方。

## 核心概念
**因果语言模型（CLM）。** 用链式法则分解联合分布并最大化对数似然：
$$\mathcal{L}_{CLM}=-\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i})$$
这恰好就是交叉熵，即最小化与数据分布之间的 KL（见[信息论](../00_foundations/06_information_theory_basics.md)）。每个 token 既是输入又是标签，提供了稠密的监督；配合因果掩码，可以在整个序列上单次前向并行训练。

**掩码语言模型（masked LM，MLM）。** BERT 掩盖约 15% 的 token，并根据*双向*上下文预测它们：$\mathcal{L}_{MLM}=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})$。它非常适合学习表示/理解，但（a）浪费监督信号（只有被掩盖的 token 贡献损失），（b）不能直接用于生成。在 LLM 上 MLM 输给了 CLM，但对编码器/嵌入（embedding）模型依然重要。

**前缀语言模型（Prefix-LM）。** 一种混合方案：对前缀使用双向注意力，对后缀做因果生成（用于 UL2 以及部分 Gemini 时代的编码器）。它在理解与生成之间架起了桥梁。

**片段破坏（span corruption，T5）。** 掩盖连续的片段，用哨兵 token 替换，再让解码器生成缺失的片段。对编码器-解码器的迁移学习很有效。

**UL2 / 去噪器混合（Mixture-of-Denoisers）。** Tay et al.（2022）用一个*混合*目标训练单个模型——短片段（R-denoising）、长片段（X-denoising）和前缀语言模型（S-denoising）——通过模式 token 切换，力图兼得各目标之长。

**中间填充（Fill-in-the-Middle，FIM）。** Bavarian et al.（2022）：把文档随机切分为前缀/中间/后缀，并重排为 `<PRE> prefix <SUF> suffix <MID> middle`，使*因果*模型无需改动架构即可学会填充。这对代码补全（IDE 中的中间插入）至关重要。如今已是代码模型的标准做法，预训练中典型的 FIM 比例为 50–90%。

**多 token 预测（Multi-Token Prediction，MTP）。** 以辅助目标的形式（额外的预测头）预测接下来的 $k$ 个 token；可提升数据效率，并为投机解码（speculative decoding）提供内置的草稿模型（draft model）（DeepSeek-V3，Gloeckle et al. 2024）。

**掩码（离散）扩散语言模型（masked diffusion LM）。** 采样掩码比例 $t\sim U(0,1)$，以概率 $t$ 独立掩盖每个 token 得到 $x_t$，再训练一个*双向* Transformer 恢复被掩盖的 token：$\mathcal{L}=-\mathbb{E}_{t,x_t}\big[\tfrac{1}{t}\sum_{i:\,x_t^i=[\text{M}]}\log p_\theta(x_0^i\mid x_t)\big]$，它是负对数似然的一个上界。生成时每一步并行地揭开许多 token。它可以看作 MLM 的一种重新加权、多掩码比例的推广，而且*是*生成式的。

## 关键挑战
- **监督效率。** MLM 只在被掩盖的那部分 token 上训练；CLM 在每个 token 上训练——这是 CLM 扩展性更好的一个主要原因。
- **暴露偏差（exposure bias）。** Teacher forcing 在真实前缀上训练，但推断（inference）时模型以自己（可能出错）的输出为条件。
- **目标与能力错配。** 下一个 token 预测优化的是局部似然，而不是长程连贯性、规划或事实性。
- **填充与从左到右。** 纯 CLM 无法原生地做填充；FIM 必须在预训练时就加入（事后低成本添加很难）。
- **并行解码与似然的取舍。** 扩散语言模型每步可解码多个 token，但在同等计算量下似然仍落后于自回归模型，而且似然最好的扩散变体并不总是推理（reasoning）最好的变体。

## 解决方案与当前最佳实践
**CLM 是生成式 LLM 的通用默认选择**，代码场景再加上 **FIM**，并以 **MTP** 作为辅助预测头——DeepSeek-V3 式的 MTP 模块（既是训练信号也是投机草稿）现已成为 Qwen3-Next、GLM-4.5/5 等开放模型的标配。编码器/嵌入模型仍使用 MLM 类目标（或对比学习目标）。文档**打包（packing）**（把多个文档拼接以填满上下文，并配合恰当的注意力/文档掩码）可最大化 token 利用率。Z-loss（惩罚对数配分函数）可稳定 softmax（见[训练稳定性](02_training_stability.md)）。扩散语言模型越来越多地是通过**转换预训练好的自回归模型**获得的，而不是从零训练。

## 各实验室观点
- **OpenAI** 确立了 CLM 的标准地位（GPT 系列），并在 Codex/代码模型上开创了 FIM。
- **Google DeepMind** 探索了最广的目标空间（T5 片段破坏、UL2、前缀语言模型），并把其中的元素融入 Gemini；它还展示过一个实验性的文本扩散模型（Gemini Diffusion，2025-05），据报道水平与 Gemini 2.0 Flash-Lite 相当，此后再无公开更新。
- **Meta** 使用 CLM + FIM（Code Llama），并研究过 MTP。
- **DeepSeek** 在 V3 中把 MTP 作为核心训练目标（效率与投机解码的协同）；**Qwen** 和**智谱（Zhipu）**在 Qwen3-Next 与 GLM-4.5/5 中跟进。
- **Inception Labs** 和 **inclusionAI（蚂蚁）**是在商用规模 / 100B 规模上押注扩散语言模型最引人注目的两家。

## 最新进展（2023–2026年11月）
**MTP** 从研究走向了生产（DeepSeek-V3）。在数据墙之下，人们对用**目标混合**提升数据效率重新产生兴趣。推理时代的后训练（post-training）已把*能力前沿*转移到建立在 CLM 预训练之上的强化学习（RL），但**预训练目标本身仍是 CLM**——更丰富的预训练目标（填充、去噪、MTP）在前沿规模上是否有实质帮助，争论仍在继续。

**2025年中 → 2026年11月更新：**
- **MTP 成为默认管线（2025–26）。** DeepSeek-V3 式的串行 MTP 模块已在 Qwen3-Next 和 GLM-4.5/5 中落地，身兼二职：既提供更稠密的训练信号，又充当原生的投机草稿（见[投机解码](../06_efficiency/04_speculative_decoding.md)）。
- **超越"下 $k$ 个 token"目标（2025-10）。** *Future Summary Prediction*（ICLR 2026）训练一个辅助头去预测长程未来的紧凑摘要，而非接下来的 $k$ 个 token；作者报告在 3B/8B 规模的数学、推理和代码任务上同时优于 NTP 和 MTP（[arXiv:2510.14751](https://arxiv.org/abs/2510.14751)）。
- **通过自蒸馏获得 MTP（2026-02）。** 用在线蒸馏（distillation）目标把预训练好的 NTP 模型转换为独立的 MTP 解码器，报告在 GSM8K 上解码加速超过 3 倍，准确率损失低于 5%（[arXiv:2602.06019](https://arxiv.org/abs/2602.06019)）——MTP 可以事后加装，而不只能在预训练时获得。
- **扩散语言模型达到商用规模（2025-06 起）。** Mercury（Inception Labs）推出每步预测多个 token 的扩散语言模型（[arXiv:2506.17298](https://arxiv.org/abs/2506.17298)）；Mercury 2（2026-02）被定位为扩散*推理*语言模型，自报在 Blackwell 上约 1,000 tok/s。
- **100B 规模的自回归→扩散转换（2025-12）。** LLaDA2.0 用块扩散日程转换预训练好的自回归 MoE 模型（16B mini、100B flash）：块大小逐步增大（预热）→ 全序列扩散（稳定）→ 小块（衰减），并加入辅助置信度损失以支持激进的并行解码（[arXiv:2512.15745](https://arxiv.org/abs/2512.15745)）。
- **扩散模型的缩放定律（2026-02）。** 将掩码扩散、均匀状态扩散和插值扩散与自回归进行比较：在扩散变体中掩码扩散的似然缩放最好，但均匀状态扩散尽管困惑度更差，在 GSM8K 上表现更好——跨目标族比较时，似然是一个不完美的代理指标（[arXiv:2602.15014](https://arxiv.org/abs/2602.15014)）。
- **中期训练作为目标层面的杠杆（2025-10 / 2026-03）。** 专门数据引入的时机与其配比权重相互作用（[arXiv:2510.14865](https://arxiv.org/abs/2510.14865)）；PRISM 发现，在中期训练（mid-training）加入科学数据可在 RL 后带来 GPQA-Diamond +17–28 分的提升，而改变 RL 数据配比带来的变化不到 2 分（[arXiv:2603.17074](https://arxiv.org/abs/2603.17074)）——决定上限的主要是 CLM 阶段，而不是 RL。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么在 LLM 上 CLM 胜过了 MLM？"** 稠密监督（每个 token）、原生生成、上下文学习、扩展干净。
- **"FIM 如何在不改架构的情况下工作？"** 用哨兵 token 重排前缀/中间/后缀，使因果模型学会填充。
- **"什么是多 token 预测，为什么有用？"** 预测未来 $k$ 个 token；数据效率更高，并免费获得用于投机解码的草稿模型。
- **"什么是暴露偏差？"** 在真实前缀上训练、在自身输出上推断 → 误差累积。
- **"自回归与掩码扩散语言模型——你在权衡什么？"** 扩散：并行、双向解码和原生填充，但似然界更松，且每步提交的 token 越多质量下降越明显；自回归：精确似然、KV 缓存（KV cache）、成熟的服务栈。目前扩散语言模型达到大规模的途径是从自回归检查点（checkpoint）转换（LLaDA2.0）。

## 开放问题
下一个 token 预测对于稳健的推理/规划是否*足够*（还是需要 RL 或新的目标），是该领域的核心争论。目标混合与 MTP 在最大规模上的价值尚无定论，也还没有一个公认的、能在预训练阶段直接针对长程连贯性或事实性的目标。扩散语言模型能否在长推理上（而不只是速度上）达到自回归的质量，以及能力有多少由中期训练数据而非目标本身决定，都是开放问题。

## 参考文献
- Devlin, J. et al. (2018). *BERT.* arXiv:1810.04805.
- Raffel, C. et al. (2020). *T5.* arXiv:1910.10683.
- Tay, Y. et al. (2022). *UL2.* arXiv:2205.05131.
- Bavarian, M. et al. (2022). *Efficient Training of LMs to Fill in the Middle.* arXiv:2207.14255.
- Gloeckle, F. et al. (2024). *Better & Faster LLMs via Multi-Token Prediction.* arXiv:2404.19737.
- Nie, S. et al. (2025). *Large Language Diffusion Models (LLaDA).* arXiv:2502.09992.
- Inception Labs (2025). *Mercury: Ultra-Fast Language Models Based on Diffusion.* arXiv:2506.17298.
- inclusionAI (2025). *LLaDA2.0.* arXiv:2512.15745.
