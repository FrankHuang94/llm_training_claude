# 数据收集与整理（Data Collection and Curation）

> **最后更新：** 2026-11-04
> **相关文件：** [数据质量与去重](01_data_quality_and_deduplication.md) · [数据配比与课程学习](03_data_mixing_and_curriculum.md) · [缩放定律](../00_foundations/04_scaling_laws.md)
> **关键论文：** Raffel et al. 2020 C4/T5（[arXiv:1910.10683](https://arxiv.org/abs/1910.10683)）· Gao et al. 2020 The Pile（[arXiv:2101.00027](https://arxiv.org/abs/2101.00027)）· Penedo et al. 2024 FineWeb（[arXiv:2406.17557](https://arxiv.org/abs/2406.17557)）· Li et al. 2024 DCLM（[arXiv:2406.11794](https://arxiv.org/abs/2406.11794)）

## 概述
预训练（pretraining）数据是决定基座模型质量的最大单一因素——在固定规模下，可以说比架构更重要。前沿模型消耗 10–30 万亿以上的 token（2025–26 年公开的数字：Kimi K2 为 15.5T，Nemotron 3 Super 约 25T，GLM-5 为 28.5T，Qwen3-Max 为 36T），其中绝大多数来自网页爬取。把 PB 级原始 HTML 变成干净、去重（deduplication）、经过质量过滤的 token 流的流水线是一项重大的工程成果，而其细节（用哪些过滤器、什么配比、去重到什么程度）是前沿实验室最严密保守的秘密之一。开源社区（The Pile → RedPajama → DCLM → FineWeb）已经逐步逆向还原出具有竞争力的流水线。

核心矛盾是**规模与质量**：Chinchilla 最优的训练需要海量 token，但未经处理的网页文本充满噪声、重复和模板内容。数据整理是一门尽可能多地榨取"有效 token"的艺术。

## 核心概念
**Common Crawl（CC）。** 原始底料：每月约 PB 级的网页，格式为 WARC（原始）/ WET（文本）。典型流水线：(1) 从 HTML 中**抽取文本**（trafilatura、resiliparse——优于 CC 默认的 WET），(2) **语言识别**（fastText/CLD3），(3) **质量过滤**（启发式规则 + 分类器），(4) **去重**（精确去重 + 模糊/MinHash 去重），(5) **PII/有害内容过滤**，(6) 针对评测集做**去污染**。

**里程碑式语料库。**
- **C4**（2020）：Colossal Clean Crawled Corpus；激进的启发式清洗（例如删除没有句末标点的页面、"脏词"列表）。约 750GB。
- **The Pile**（2020，EleutherAI）：825GB，由 22 个高质量来源（PubMed、ArXiv、GitHub、Books、StackExchange）精心组成的*混合*语料——开创了有意识的领域构成。
- **RedPajama**（2023）：对 LLaMA-1 数据配比的开放复现；RedPajama-v2 发布了附带质量信号的 30T token。
- **RefinedWeb**（2023，Falcon）：表明*仅用网页*、经过重度过滤和去重的数据可以媲美精选混合语料。
- **DCLM**（2024）：一个基准（benchmark）加数据集，表明**基于模型的过滤**（训练一个识别高质量文本的 fastText 分类器）带来了大部分收益；DCLM-Baseline 超越了此前的开放数据集。
- **FineWeb / FineWeb-Edu**（2024，HuggingFace）：15T token 的开放网页语料；FineWeb-Edu 用"教育价值"分类器进行过滤，显著提升了基准表现。

## 关键挑战
- **模板内容与噪声。** 导航栏、广告、SEO 垃圾充斥原始 HTML；抽取质量的影响会贯穿整条流水线不断累积。
- **数据墙。** 高质量公开文本是有限的（约数十万亿 token）；前沿模型正接近耗尽，被迫转向合成数据（synthetic data）与重复使用。
- **法律与伦理风险。** 版权、robots.txt/服务条款、GDPR/PII 以及授权许可仍处于诉讼之中。到 2026 年末，判例在*来源*问题上出现分野：使用合法获得的图书进行训练被认定为合理使用，但建立盗版图书的集中库则不是（*Bartz v. Anthropic*）；而截至 2026 年 11 月，NYT v. OpenAI/Microsoft 尚无实体裁决。
- **基准污染。** 爬取数据不可避免地包含测试集；未被发现的泄漏会抬高报告分数（见[数据污染](05_data_contamination_and_evaluation_leakage.md)）。

## 解决方案与当前最佳实践
已收敛的配方：更好的**文本抽取**（用 trafilatura/resiliparse 替代 WET）、**基于模型的质量分类器**（DCLM/FineWeb-Edu 风格）而不仅是启发式规则、**全局模糊去重**（MinHash LSH）、严格的**评测去污染**，以及最后在最高质量子集（教科书、精选代码/数学）上进行的**退火/中期训练（mid-training）**阶段。保存文档级的来源与质量信号，以便低成本地重新调整配比权重。

## 各实验室观点
- **OpenAI/Anthropic/Google** 的流水线均不公开；一般认为它们都大量使用基于分类器的过滤以及授权/精选数据合作。Anthropic 15 亿美元的 *Bartz* 和解（2026-07 获最终批准）要求销毁盗版文件；如今它还通过数据选择有意识地塑造模型能力（Opus 5 没有在网络安全任务上训练）。
- **Meta** 披露 Llama 3 在约 15T token 上训练，进行了大量去重和质量过滤，并包含较大比例的代码/数学/多语言数据；其 2026 年的 Muse 模型为闭源且无公开文档。
- **DeepSeek** 强调以代码和数学为主的配比以及高效的流水线。
- **HuggingFace/EleutherAI/AllenAI**（FineWeb、The Pile、Dolma）推动开放、可复现的数据整理，成为业界对标的基准——如今又加入了 FineWeb2（1,000 多种语言）、AI2 的 Dolma 3（约 6T token，支撑完全开放的 OLMo 3）以及 NVIDIA 发布的 Nemotron 3 数据与配方。

## 最新进展（2023–2026年11月）
从启发式过滤转向**基于模型的过滤**（FineWeb-Edu、DCLM）是 2024 年的头条成果。授权数据合作（新闻、Reddit 等论坛、Stack Overflow）大量涌现。**合成与改写的网页数据**（"Web Rephrase Augmented Pre-training"、phi 系列）应运而生，以突破数据墙。多模态（multimodal）和多语言爬取规模扩大。来源追溯/水印以及退出登记机制（例如 robots.txt 扩展）成为活跃的政策领域。

**2025年中 → 2026年11月更新：**
- **大规模多语言整理（2025-06；arXiv:2506.20920）。** FineWeb2 让 FineWeb 流水线在 1,000 多种语言上按语言自动适配（语言识别、去重、过滤阈值），并采用基于去重信息的上采样（"rehydration"，再水化）；产出约 20 TB / 50 亿文档。
- **"自然数据是被低估使用，而非已被耗尽"（2026-04/05）。** FinePhrase（arXiv:2604.13977；生成了超过 1T token）表明，把网页文本改写为结构化格式胜过精选的网页数据；SynPro（arXiv:2605.17849）报告在 Chinchilla 数据量 10% 的设定下，有效 token 是简单重复的 3.7–5.2 倍。数据整理正从*选择*转向*变换*（见[合成数据](04_synthetic_data_generation.md)）。
- **来源成为法律上的分界线。** *Bartz v. Anthropic*（2025-06）：合法获得的图书属于合理使用，盗版图书库则不是 → 15 亿美元集体和解（每部作品约 3,000 美元，约 46.5 万–50 万部作品；2026-07 获最终批准）。*Thomson Reuters v. ROSS*（第三巡回上诉法院，2026-09-29）：美国首个关于 AI 训练合理使用的上诉判决——不构成合理使用，但针对的是非生成式工具。*Getty v. Stability*（英国，2025-11）：模型权重不是"侵权复制品"。
- **监管层面的数据义务（2025-08 → 2026-08）。** 欧盟《AI 法案》的 GPAI 义务要求按委员会模板公开训练数据摘要，并制定尊重 TDM（文本与数据挖掘）退出声明的版权政策；执法权（罚款最高为营业额的 3% 或 1,500 万欧元）自 2026-08-02 起生效。数据流水线如今需要文档级的来源信息才能生成这些摘要。
- **数据选择成为安全杠杆（2026）。** Anthropic 表示在训练中降低了 Opus 4.7 的网络安全能力，且没有在网络安全任务上训练 Opus 5；据报道 Moonshot 从承诺开放权重（open weights）的 Kimi K3.1 Pro 版本中移除了网络安全相关的训练数据。语料中*排除*了什么，如今已成为一项有记录的发布决策。
- **前沿模型的输出更难采集。** 保留的、绑定账户的思考内容（Opus 5.5、Sonnet 5.5）、蒸馏（distillation）分类器（Fable 5）以及 Anthropic 2026 年 10 月关于蒸馏攻击的文章，都限制了第三方数据流水线可获取的推理轨迹。
- **完全开放的数据栈。** OLMo 3（2025-11）及其 Dolma 3（约 6T token）和 Nemotron 3（2025-12，数据与配方一并发布）在 7B–32B 和 30B-A3B 规模上提供了可复现的基线。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"讲讲从 Common Crawl 到预训练数据的流水线。"** 抽取 → 语言识别 → 质量过滤 → 去重 → 去污染 → 配比。
- **"从 The Pile 到 FineWeb-Edu 发生了什么变化？"** 从启发式整理 → 网页规模上基于模型的教育质量过滤。
- **"你如何应对数据墙？"** 重复（最多约 4 轮）、合成/改写数据、多模态、授权来源。
- **"在数据收集阶段如何防止基准污染（contamination）？"** 针对留出评测集做 n-gram/嵌入层面的去污染。
- **"在 *Bartz* 之后，你会如何改变数据获取流水线？"** 逐文档追踪获取来源、清除影子图书馆来源、尊重 TDM 退出声明，并保留足够的元数据以生成符合欧盟《AI 法案》要求的训练数据摘要。

## 开放问题
关于过滤*多少*、过滤*哪些*才是最优，目前没有原则性的理论——它是经验性的、依数据集而定的。量化"有效 token"与原始 token 的关系、数据墙带来的长期上限、合成数据的价值与风险（模型坍缩），以及网页规模训练的法律地位，都仍是开放且有争议的问题。

## 参考文献
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning (C4/T5).* arXiv:1910.10683.
- Gao, L. et al. (2020). *The Pile.* arXiv:2101.00027.
- Penedo, G. et al. (2023). *RefinedWeb.* arXiv:2306.01116.
- Penedo, G. et al. (2024). *FineWeb / FineWeb-Edu.* arXiv:2406.17557.
- Li, J. et al. (2024). *DataComp-LM (DCLM).* arXiv:2406.11794.
- Penedo, G. et al. (2025). *FineWeb2: One Pipeline to Scale Them All.* arXiv:2506.20920.
