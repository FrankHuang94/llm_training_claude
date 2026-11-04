# 分词（Tokenization）

> **最后更新：** 2026-11-04
> **相关文件：** [分词器训练](../01_data/02_tokenizer_training.md) · [Transformer 架构](00_transformer_architecture.md) · [信息论基础](06_information_theory_basics.md)
> **关键论文：** Sennrich et al. 2016 BPE（[arXiv:1508.07909](https://arxiv.org/abs/1508.07909)）· Kudo 2018 Unigram LM（[arXiv:1804.10959](https://arxiv.org/abs/1804.10959)）· Kudo & Richardson 2018 SentencePiece（[arXiv:1808.06226](https://arxiv.org/abs/1808.06226)）· Radford et al. 2019 GPT-2（字节级 BPE）· Xue et al. 2022 ByT5（[arXiv:2105.13626](https://arxiv.org/abs/2105.13626)）

## 概述
分词（tokenization）是原始字节与 Transformer 所消费的整数序列之间一个"有损但可逆"的接口。它决定了模型的有效词表、序列长度（进而决定计算成本），以及数量惊人的下游行为——算术、拼写、多语言公平性和代码处理都取决于分词器（tokenizer）的设计。由于语言模型在 token 之上运作，*分词器定义了概率模型进行因子分解的基本单元*，许多所谓的"模型"失败其实是分词器的产物。

现代大语言模型（LLM）使用**子词**分词，它是字符级（序列长、没有 OOV、单元语义弱）与词级（词表巨大、存在 OOV 问题）之间的折中。子词方法学习一个在序列压缩与词表大小之间取得平衡的词表，通常为 32K–256K 个 token。

## 核心概念
**字节对编码（BPE）。** 从字符/字节出发，贪心地合并出现频率最高的相邻对，重复直到达到目标词表大小。编码时按顺序应用学到的合并规则。BPE 是确定性的、由频率驱动的。**字节级 BPE**（从 GPT-2 开始）直接作用于原始 UTF-8 字节，保证任何输入都不会出现 OOV（任意 Unicode 字符串都可表示），代价是非拉丁文字的一个字符会被拆成多个 token。

**WordPiece**（BERT）与 BPE 类似，但合并的依据不是原始频率，而是在一元模型下使训练语料似然最大的那一对：选择使 $\frac{\text{count}(xy)}{\text{count}(x)\text{count}(y)}$ 最大的合并。

**Unigram LM**（Kudo 2018）采取相反的自顶向下方法：从一个大的候选词表开始，用 EM 迭代地剪掉那些在一元模型下对语料似然损害最小的 token。它产生概率化的切分，从而支持**子词正则化**（训练时采样不同的切分方式以提高鲁棒性）。

**SentencePiece** 是一个实现（而非算法），封装了 BPE/Unigram，直接在原始文本上运行而无需预分词，并把空白视为一个元符号（`▁`）。这使它与语言无关——对没有空格分词的语言（中文、日语、泰语）至关重要。

**繁殖率（fertility）** = 每个词（或每个字节）平均对应的 token 数。繁殖率高意味着一种语言"昂贵"——每单位语义需要更多 token、成本更高、有效上下文更短。英语通常约为 1.3 token/词；在以英语为中心的分词器下，许多低资源语言的繁殖率要高出 2–4 倍，这是公平性问题，也是成本问题。

## 关键挑战
- **多语言不公平。** 针对英语优化的词表会把其他文字切得过碎，抬高非英语用户的成本和延迟（latency），并降低质量。
- **算术/拼写的脆弱性。** 数字切分不一致（例如"327"是一个 token 还是"3""27"）会损害算术；由于 token 掩盖了字符，字符级任务很难。
- **"分词是瓶颈"之争。** 分词器是在模型之前训练好、随后冻结的产物，会产生故障 token（例如 `SolidGoldMagikarp`），并使领域适配变得复杂。
- **词表与序列长度的权衡。** 更大的词表能压缩序列（计算更少），但会扩大嵌入（embedding）/softmax，并带来罕见 token 训练不足的风险。

## 解决方案与当前最佳实践
当前做法：**通过 SentencePiece/`tiktoken` 实现的字节级 BPE 或 Unigram**，前沿多语言模型的词表为 100K–256K（GPT-4o 约 200K；Llama 3 为 128K，较 Llama 2 的 32K 大幅提升——这是有意为之的效率/多语言收益）。数字常被拆成单个数字 token 或固定的 3 位数字块，以帮助算术。保留/特殊 token 用于编码对话角色和工具调用。分词器的训练数据应与预训练（pretraining）配比保持一致，以平衡各语言与代码的繁殖率。

## 各实验室观点
- **OpenAI** 推出了 `tiktoken`（字节级 BPE）；GPT-4o 更大的多语言词表显著减少了非英语文本的 token 数。
- **Meta** 将 Llama 3 的词表扩展到 128K，理由是效率和多语言覆盖；其 2026 年的 Muse 模型为闭源，分词器未公开。
- **Anthropic** 随 Claude Opus 4.7（2026-04）推出了新的分词器，对同样的文本会产生约 1.0–1.35 倍的 token——这说明尽管有"冻结产物"的代价，前沿实验室仍会在模型版本之间修订分词器。
- **Google** 使用 SentencePiece（Gemini/Gemma），配合大型多语言词表。
- **DeepSeek/Mistral** 使用针对代码密集和多语言配比调优的字节级 BPE。
- 各实验室都存在一股研究暗流（Meta 的 BLT、DeepMind），推动**无分词器/字节级**模型。

## 最新进展（2023–2026年11月）
**Byte Latent Transformer（BLT）**（Meta，2024）用基于熵动态确定大小的字节块取代固定 token，目标是在保持 token 级效率的同时彻底去掉分词器。**MegaByte**、**MambaByte** 以及基于熵的分块在探索同一前沿。多语言词表扩展和逐语言繁殖率审计如今已是预训练的标准卫生流程。

**2025年中 → 2026年11月更新：**
- **分词器变化体现在价格上（2026-04）。** Claude Opus 4.7 的新分词器对完全相同的输入会产生约 1.0–1.35 倍的 token。因此，如果不归一化到字节或字符，跨分词器版本比较每 token 价格、上下文上限（"1M token"）和每任务 token 数指标都没有意义——这与困惑度和每字节比特数（bits-per-byte）之间的注意事项相同（见[信息论基础](06_information_theory_basics.md)）。
- **token 效率成为头条指标——但主要来自冗长度，而非繁殖率。** Opus 4.5 以少 76% 的输出 token 达到了 Sonnet 4.5 在 SWE-bench Verified 上的最好成绩（2025-11）；GPT-5.5 在每个 Codex 任务上用的 token 更少（2026-04）；据报道 Gemini 3.6 Flash 的输出 token 比 3.5 Flash 少约 17%（2026-07）。面试官越来越期望你能把*分词器压缩*与*推理（reasoning）长度*的影响区分开。
- **用于分词器训练的多语言语料规模扩大（2025-06）。** FineWeb2（arXiv:2506.20920）把一条按语言自适应的 FineWeb 流水线应用到 1,000 多种语言上（20 TB、50 亿文档），为训练繁殖率均衡的词表提供了远比以英语为中心的爬取数据更好的样本。
- **超越逐 token 的因子分解。** 多 token 预测头已是 DeepSeek-V3 风格模型的标配；基于自蒸馏（distillation）的 MTP（arXiv:2602.06019）把一个 NTP 模型转换为 MTP 解码器，在 GSM8K 上解码提速 3 倍以上，准确率损失不到 5%；扩散语言模型（Mercury，arXiv:2506.17298；LLaDA2.0，arXiv:2512.15745）每一步输出多个子词 token。分词器仍然定义了基本单元，但不再决定顺序步骤的数量。
- **嵌入表是一项部署成本。** Gemma 4 的 E2B/E4B 端侧模型（2026-04）使用逐层嵌入（per-layer embeddings）——这提醒我们，大词表的嵌入参数在 FLOPs 上很便宜，但在内存上并不便宜，这会影响端侧的词表决策。
- **原生多模态与语音 token。** Qwen3-Omni（arXiv:2509.17765）通过多码本编解码器流式输出语音（首包延迟约 234 ms），因此"分词"如今也包括与文本子词并存的离散音频码。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"比较 BPE、WordPiece 和 Unigram。"** 自底向上按频率合并 vs 按似然合并 vs 自顶向下剪枝；哪一种支持子词正则化。
- **"为什么使用字节级 BPE，代价是什么？"** 零 OOV/可表示任意 Unicode vs 非拉丁字符被拆成多个 token。
- **"为什么 LLM 不擅长算术/拼写？"** 分词掩盖了字符，且数字切分不一致。
- **"你会如何让分词器在各语言之间更公平？"** 平衡训练配比、测量繁殖率、扩展词表、考虑字节级方案。
- **"某个新模型版本的分词器对同样的文本多产生 30% 的 token，会改变什么？"** 每次请求的成本、以词计的有效上下文、每 token 损失/困惑度的可比性（改用 BPB）、每个文档的 KV 缓存大小，以及任何以 token 计价的评测（evaluation）或速率限制。

## 开放问题
端到端的**无分词器**模型能否在前沿规模上追平子词方法的效率仍未解决。最优词表大小作为模型/数据规模的函数，尚缺乏一条简洁的缩放定律（scaling law）。在不重新训练的情况下把冻结的分词器适配到新领域/新语言依然别扭，对故障 token 的鲁棒性也尚未被完全理解。

## 参考文献
- Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units (BPE).* arXiv:1508.07909.
- Kudo, T. (2018). *Subword Regularization (Unigram LM).* arXiv:1804.10959.
- Kudo, T., Richardson, J. (2018). *SentencePiece.* arXiv:1808.06226.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2).*
- Pagnoni, A. et al. (2024). *Byte Latent Transformer.* arXiv:2412.09871.
- Penedo, G. et al. (2025). *FineWeb2: One Pipeline to Scale Them All.* arXiv:2506.20920.
- Qwen Team (2025). *Qwen3-Omni Technical Report.* arXiv:2509.17765.
