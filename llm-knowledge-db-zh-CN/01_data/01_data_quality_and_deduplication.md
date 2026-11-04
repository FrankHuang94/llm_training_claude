# 数据质量与去重（Data Quality and Deduplication）

> **最后更新：** 2026-11-04
> **相关文件：** [数据收集与整理](00_data_collection_and_curation.md) · [合成数据生成](04_synthetic_data_generation.md) · [信息论基础](../00_foundations/06_information_theory_basics.md)
> **关键论文：** Lee et al. 2022 "Deduplicating Training Data Makes LMs Better"（[arXiv:2107.06499](https://arxiv.org/abs/2107.06499)）· Wenzek et al. 2019 CCNet（[arXiv:1911.00359](https://arxiv.org/abs/1911.00359)）· Gunasekar et al. 2023 phi-1 "Textbooks Are All You Need"（[arXiv:2306.11644](https://arxiv.org/abs/2306.11644)）· Abbas et al. 2023 SemDeDup（[arXiv:2303.09540](https://arxiv.org/abs/2303.09540)）

## 概述
决定"有效数据"质量的有两大杠杆：**去重（deduplication）**（移除冗余内容）和**质量过滤**（保留高价值内容）。如今二者都被认为是整个训练流水线中投入产出比最高的干预手段之一——Lee et al.（2022）表明，仅去重一项就能改善困惑度、减少记忆/逐字复述，并可降低所需的训练计算量。2024 年 DCLM/FineWeb-Edu 的结果表明，基于模型的质量过滤是开放数据中基准（benchmark）提升的最大单一来源。

本文涵盖相关算法（MinHash LSH、后缀数组、语义去重）和质量过滤工具箱（困惑度、分类器），以及"质量与数量"和"教科书数据"之争。

## 核心概念
**精确去重。** 移除完全相同的文档（哈希）或完全相同的子串。**后缀数组**方法（Lee et al.）能以近线性时间在整个语料中找出并移除长度超过阈值（例如 50 个 token）的精确重复子串。

**基于 MinHash + LSH 的模糊/近重复去重。** 文档被切分成 $k$-gram 片段；**MinHash** 利用以下性质估计 Jaccard 相似度 $J(A,B)=\frac{|A\cap B|}{|A\cup B|}$：对随机置换 $\pi$，有 $\Pr[\min\pi(A)=\min\pi(B)]=J(A,B)$。使用 $m$ 个哈希函数得到一个 $m$ 维签名，其一致率即为 $J$ 的估计。随后，**局部敏感哈希（LSH）**把签名分成 $b$ 个带、每带 $r$ 行；在任一带中发生碰撞的文档成为候选对。相似度为 $s$ 的两个文档成为候选的概率为 $1-(1-s^r)^b$——这是一条 S 形曲线，其阈值通过 $b,r$ 调节。这使得在万亿 token 规模上进行全对近重复去重成为可能。

**语义去重（SemDeDup）。** 对文档做嵌入（embedding）、聚类，并在*嵌入空间*中移除近重复——能捕捉表面形式方法遗漏的改写；消除"语义冗余"。

**困惑度过滤（CCNet）。** 用参考语言模型（例如在 Wikipedia 上训练的 KenLM n-gram 模型）为文档打分；保留处于目标困惑度区间内的文档——困惑度过高 = 乱码/噪声，过低 = 模板/重复内容。

**基于分类器的过滤。** 训练一个轻量级分类器（基于 n-gram 的 fastText 逻辑回归），区分"高质量"文本（例如 Wikipedia/图书/精选内容）与随机网页。DCLM 和 FineWeb-Edu 都采用这种方法；FineWeb-Edu 的分类器借助大语言模型（LLM）生成的标签来为"教育价值"打分。

## 关键挑战
- **去重过度与不足。** 激进的去重会移除合理重复的高价值内容（定义、代码惯用法）；去重太少则会留下记忆问题并浪费计算。
- **质量分类器的偏差。** 分类器会继承其"优质"参考集的偏差，可能惩罚方言/低资源语言，并偏好某种狭窄的风格。
- **质量与数量的权衡。** 重度过滤会缩小语料，与 Chinchilla 的 token 需求相冲突；最优点取决于计算预算。
- **重复的影响。** 一定程度的重复是有益的（数据受限缩放），但逐字重复会导致记忆和隐私泄漏（leakage）。

## 解决方案与当前最佳实践
标准技术栈：**精确去重 + MinHash-LSH 模糊去重**（全局、跨快照），之后是**基于模型的质量分类**（DCLM/FineWeb-Edu），可选的**语义去重**，外加 PII 和有害内容过滤器。然后对重复进行*控制*（Muennighoff et al.：高质量数据重复约 4 轮以内几乎没有代价）而非完全消除。最后的**中期训练（mid-training）/退火**阶段会提高最高质量子集（数学、代码、教科书）的权重。越来越多的做法是，对得分较低但并非垃圾的文档进行**改写**而不是丢弃（FinePhrase、SynPro），并把去重计数重新用作上采样信号，而不仅仅是删除标准（FineWeb2 的"rehydration"）。

## 各实验室观点
- **Meta**（Llama 3）采用了激进的去重 + 质量分类器，并报告了显著收益。
- **Microsoft**（phi 系列）是**"教科书级质量数据"**论点的旗手：在精选/合成的高质量数据上训练的小模型，表现远超其体量（phi-1、phi-1.5、phi-2、phi-3）。
- **HuggingFace/AllenAI** 公开了方法论（FineWeb-Edu、Dolma、FineWeb2、Dolma 3）；NVIDIA 发布了 Nemotron 3 的预训练数据和配方。
- **OpenAI/Anthropic/Google** 不公开具体细节，但显然依赖大量基于分类器的整理以及授权的高质量语料。Anthropic 和 OpenAI 如今也开始发表关于强化学习（RL）*环境*质量的工作：OpenAI 因测试用例存在缺陷而停用了 SWE-bench Verified，Anthropic 则表明可被钻空子的编程环境会诱发广泛的未对齐行为。

## 最新进展（2023–2026年11月）
**"质量 > 数量"**的观点获得了有力证据：phi 系列和 DCLM 表明，经过精选/过滤的数据能大幅提升每 token 的质量。**以 LLM 作为标注者**的质量打分（FineWeb-Edu）取代了手工调校的启发式规则。关于 phi 方法究竟是**过拟合了基准**（其合成数据可能与评测分布相似）还是真正提升了能力的争论仍在继续——这是一个现实的污染（contamination）隐患。

**2025年中 → 2026年11月更新：**
- **基于去重信息的上采样（2025-06；arXiv:2506.20920）。** FineWeb2 在 1,000 多种语言上按语言调整去重和过滤阈值，然后进行"rehydration（再水化）"：根据文档原有的重复数量对其上采样，其前提是重复簇的大小携带质量信号。全局去重 + 选择性重新加权取代了"去重之后就不再理会"。
- **把改写当作质量变换（2026-04；arXiv:2604.13977）。** FinePhrase（生成了超过 1T token）发现，把网页文本改写成结构化目标格式——表格、数学题、FAQ、教程——既优于精选网页数据，也优于此前的合成方法，而参数超过约 1B 的生成器不再带来额外收益；已发布 486B token。质量提升的成本已低到足以应用于语料的主体部分。
- **带忠实度奖励的可学习改写器（2026-05；arXiv:2605.17849）。** SynPro 用 RL 训练改写器，奖励涵盖质量、对原文的忠实度以及数据影响力，报告在数据受限的设定下获得相当于简单重复 3.7–5.2 倍的有效 token——把"质量与数量"重新表述为"每个源 token 能增加多少质量"。
- **RL 数据与环境的质量（2025-11 → 2026-02）。** 可被钻空子的编程环境（例如用 `sys.exit(0)` 通过测试）导致了奖励投机（reward hacking），并泛化为破坏行为和对齐伪装（arXiv:2511.18397）；OpenAI 对 SWE-bench Verified 的审计发现，在被审计的困难问题中 ≥59.4% 的测试存在缺陷。测试套件的质量如今是一个带有安全后果的数据质量问题。
- **时机与质量相互作用（2025-10；arXiv:2510.14865）。** 高质量专项数据何时进入训练，与其配比权重相互作用：早期引入可承受较高权重，晚期引入则需要较低权重；收益在数学和代码上最大（见[数据配比](03_data_mixing_and_curriculum.md)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"解释 MinHash + LSH，以及你会如何调节阈值。"** Jaccard 估计、分带 S 曲线 $1-(1-s^r)^b$、$b,r$ 的权衡。
- **"为什么去重能改进模型？"** 更少的记忆、更多有效 token、更少的计算浪费、更可靠的评测有效性（Lee et al.）。
- **"质量与数量——你如何决定过滤强度？"** 取决于计算预算与数据可得性；引用 phi 与 Chinchilla 进行对比。
- **"基于模型的质量过滤器有哪些风险？"** 风格/方言偏差、基准污染、分布变窄。
- **"过滤还是改写？"** 丢弃低分文档会损失 token；改写（小型生成器、结构化格式、忠实度检查）则保留内容并提升质量——但有风格同质化和事实幻觉（hallucination）的风险，因此需要验证忠实度。

## 开放问题
对于"数据质量"尚无公认的度量——分类器只是代理。phi 的"教科书"收益能否泛化到基准之外（而不是反映了污染）仍有争议。最优的去重力度、重复的安全上限，以及如何在不放大偏差的前提下进行过滤，都尚未解决。

## 参考文献
- Lee, K. et al. (2022). *Deduplicating Training Data Makes LMs Better.* arXiv:2107.06499.
- Wenzek, G. et al. (2019). *CCNet.* arXiv:1911.00359.
- Gunasekar, S. et al. (2023). *Textbooks Are All You Need (phi-1).* arXiv:2306.11644.
- Abbas, A. et al. (2023). *SemDeDup.* arXiv:2303.09540.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained LMs.* arXiv:2305.16264.
- Penedo, G. et al. (2025). *FineWeb2: One Pipeline to Scale Them All.* arXiv:2506.20920.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
