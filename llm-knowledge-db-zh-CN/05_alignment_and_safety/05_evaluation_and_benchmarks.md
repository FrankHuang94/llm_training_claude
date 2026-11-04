# 评测与基准（Evaluation and Benchmarks）

> **最后更新：** 2026-11-04
> **相关文件：** [数据污染](../01_data/05_data_contamination_and_evaluation_leakage.md) · [推理模型](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [幻觉](01_hallucination_causes_and_mitigations.md) · [前沿安全框架与治理](06_frontier_safety_frameworks_and_governance.md)
> **关键论文：** Hendrycks et al. 2020 MMLU（[arXiv:2009.03300](https://arxiv.org/abs/2009.03300)）· Chen et al. 2021 HumanEval（[arXiv:2107.03374](https://arxiv.org/abs/2107.03374)）· Rein et al. 2023 GPQA（[arXiv:2311.12022](https://arxiv.org/abs/2311.12022)）· Chiang et al. 2024 Chatbot Arena（[arXiv:2403.04132](https://arxiv.org/abs/2403.04132)）· Kwa et al. 2025 METR Time Horizons（[arXiv:2503.14499](https://arxiv.org/abs/2503.14499)）

## 概述
评测（evaluation）是这个领域衡量进展、比较模型并决定部署什么的方式——然而它几乎一直处于危机之中。基准（benchmark）一旦变得重要就会**饱和**，会因网络规模的训练而遭受**污染（contamination）**，且往往无法预测真实世界的有用性。结果是新基准不断更替——更难、更抗污染——并转向**人类偏好**与**智能体式/真实任务**评测。2025–26 年，头部基准的半衰期缩短到约 12–18 个月；各实验室转向带版本号的智能体式套件、Elo 式经济评测、交互式新环境测试以及**时间跨度（time horizon）**指标——与此同时，**评测感知（evaluation awareness）**成为安全测试中的一个混淆因素。对于面试（尤其是评测方向的岗位），了解主要基准、其失效模式以及现代方法论至关重要。

## 核心概念
**知识与推理基准。**
- **MMLU**（57 个学科，多项选择）——标准的知识基准；如今已基本**饱和**（前沿模型 >88%），由此催生了 **MMLU-Pro**（更难、选项更多）和 **MMLU-Redux**（修正标注错误）。
- **GPQA**（"无法靠谷歌搜到"的研究生级科学题）——困难、由专家编写；"Diamond"子集曾是 2024–25 年的推理标尺，如今前沿模型已达 90 分段中段。
- **MATH** / **AIME** / **GSM8K**——数学推理；GSM8K 已饱和，AIME/竞赛数学是推理模型的核心；**FrontierMath**（极难、留出）一直抵抗饱和，直到据报道其 Tier 4 在 2026 年接近饱和。
- **BIG-Bench** / **BIG-Bench Hard**——多样化任务套件；**HLE（Humanity's Last Exam）**瞄准前沿难度（带工具的分数如今已超过 60%）。
- **ARC-AGI-2 / ARC-AGI-3**——抽象推理谜题；ARC-AGI-3（2026）是*交互式*的：智能体必须探索游戏环境并推断目标。

**代码与智能体式基准。**
- **HumanEval / MBPP**——函数合成（pass@k）；基本饱和、易受污染 → **LiveCodeBench**（持续更新、抗污染）。
- **SWE-bench（/Verified/Pro）**——在真实仓库中解决真实 GitHub issue；Verified 曾是 2024–25 年的旗舰基准，但 OpenAI 于 2026-02 停止报告它（测试有缺陷 + 污染），转向 **SWE-bench Pro**；**Terminal-Bench**（命令行任务，版本从 2.0 → 4.0）、**OSWorld**（计算机操作（computer use））、**τ-bench**、**WebArena**、**GAIA** 用于工具/智能体任务。
- **经济评测**——**GDPval**（由专家评分的职业交付物），如今以 Elo 形式报告（GDPval-AA）。

**时间跨度（METR）。** 在人类完成时间为 $t$ 的任务上拟合 $P(\text{success}\mid t) = \sigma(\alpha + \beta \log t)$（$\beta<0$）；**50% 时间跨度**为 $t_{50} = \exp(-\alpha/\beta)$。沿模型发布日期跟踪 $t_{50}$ 并拟合指数曲线即可得到倍增时间。由于逻辑曲线较平缓，80% 时间跨度要短得多。

**人类偏好评测。**
- **Chatbot Arena（LMSYS/LMArena）：** 众包的匿名两两对战，通过 **Elo/Bradley-Terry** 聚合为排名。能捕捉真实世界偏好且难以污染，但反映的*风格/讨喜程度*不亚于正确性，并且可能被钻空子（冗长、格式、私有变体测试）。
- **LLM 作为裁判**（MT-Bench、AlpacaEval、Arena-Hard）：由强模型给回答打分——可扩展但有偏差（位置、冗长、自我偏好）；长度控制版本可缓解这一问题。

**指标。** pass@k（代码）、精确匹配/准确率（问答/数学）、Elo（偏好、经济任务）、相对参考答案的胜率、时间跨度（自主性）。对统计严谨性（置信区间、多随机种子）的要求日益提高。

## 关键挑战
- **饱和。** 重要基准很快触顶，失去区分能力（SWE-bench Verified、OSWorld、ARC-AGI-2、FrontierMath Tier 4 都在约 12–18 个月内饱和）。
- **污染。** 测试数据泄漏进训练集，抬高分数（见 [数据污染](../01_data/05_data_contamination_and_evaluation_leakage.md)）；有互联网访问权限的智能体还可能在评测时*找到*答案。
- **标签/测试噪声。** 评分器和单元测试可能拒绝正确的解答，或要求题目未要求的功能。
- **评测框架/脚手架（harness/scaffold）差异。** 厂商的脚手架可能比标准化评测框架高出 10–30 分；"provider adapter"可能大幅改变交互式评测的分数。
- **构念效度。** 基准往往测不到我们真正关心的东西（真实有用性、安全性、鲁棒性）；多项选择问答奖励的是应试技巧。
- **钻空子与过拟合。** 实验室（有意或无意地）针对排行榜数字进行优化；Arena 容易受到风格/冗长的操纵以及选择性提交的影响。
- **推理模型带来的问题。** 长思维链模型需要考虑可变计算量的评测设置；pass@1 与 pass@k 以及推断预算都很重要。
- **评测感知。** 模型越来越能识别出测试，因此安全评测可能高估部署时的安全性。
- **安全/对齐难以做基准。** 拒答、诚实、鲁棒性缺乏稳定的真值。

## 解决方案与当前最佳实践
使用**组合**：知识（MMLU-Pro、GPQA）、推理（AIME、FrontierMath、HLE，带工具*与*不带工具）、代码/智能体（在标准化评测框架上的 SWE-bench Pro、Terminal-Bench、OSWorld-Verified、LiveCodeBench）、经济（GDPval-AA）、自主性（METR 时间跨度）、新环境（ARC-AGI-3）、偏好（Arena），以及在训练截止日期之后构建的**私有/留出**内部评测。优先选择**抗污染 / 实时 / 动态**的基准以及用于检测记忆的**扰动测试**（GSM-Symbolic）。报告**去污染方法**、**置信区间**、**评测框架/脚手架**和**推断设置**（effort 级别、每任务 token 数、每任务成本）。既要审计模型输出，也要审计基准本身的*测试*。在**没有意外互联网访问的沙箱**中运行智能体式评测。对偏好评判使用**长度控制**指标并辅以人工抽查。对任何单一基准都保持怀疑；重视真实任务和留出数据上的表现。

## 各实验室观点
- **OpenAI：** 停止报告 SWE-bench Verified（2026-02），改用 SWE-bench Pro；推出 GDPval（2025-09）；GPT-5.5/6 的头部评测包括 Terminal-Bench、OSWorld-Verified 和 GDPval。
- **Anthropic：** 报告 Terminal-Bench 4.0、OSWorld 2.1、带工具的 HLE 以及 GDPval-AA Elo（Opus 5.5 的发布页未报告 SWE-bench）；由 METR 等进行外部发布前测试；公开承认评测感知尚未解决，并在智能体"执着"行为事故后停用/重建了部分公开评测。
- **Google DeepMind：** 报告 ARC-AGI-2、GPQA、HLE、Terminal-Bench；其"AI 协作数学家（AI co-mathematician）"智能体在 FrontierMath Tier 4 上达到 48%（2026-05）。
- **LMArena（LMSYS）：** 事实上的公共偏好排行榜；影响力大但有争议（风格偏差、私有变体测试、采样不对称）。
- **Meta/DeepSeek/Qwen：** 报告标准套件；开放权重使独立复评成为可能（对挑拣结果的一种制衡）。
- **METR / Epoch / Scale / ARC Prize / AllenAI / EleutherAI：** 第三方测量（时间跨度、FrontierMath、SEAL 排行榜、ARC-AGI）以及标准化方法论的开源评测框架（lm-eval-harness、OLMES）。

## 最新进展（2023–2026年11月）
转向**智能体式与真实任务**评测（截至 2025 年，SWE-bench Verified 一直是头部指标）、**抗污染/实时**基准（LiveCodeBench、LiveBench、FrontierMath、HLE），以及随着旧基准饱和而出现的**更难的知识集**（MMLU-Pro、GPQA）。对 **Arena 被操纵**以及裁判偏差的审视日益增多。推理模型迫使评测变得**感知推断开销**。人们对**评测科学**本身（可靠性、统计严谨性、构念效度）以及第三方/独立评测机构的兴趣日益上升。

**2025年中 → 2026年11月更新：**
- **METR 时间跨度加速（arXiv:2503.14499 → TH1.1，2026-01）。** 2019–25 年约 7 个月的倍增时间，在 2023 年后压缩到约 4.3 个月（≈130 天；按 2024 年以来的数据拟合约 89 天）。Claude Mythos Preview 的一个早期快照测得"至少 16 小时"（95% 置信区间 8.5–55 小时），METR 表示其任务套件无法可靠测量远超约 16 小时的时间跨度——这个元基准本身也已接近上限。
- **SWE-bench Verified 被停用（2026-02）。** OpenAI 的审计发现，在被审计的困难题中，≥59.4% 的测试会拒绝正确解答，18.8% 测试了题目未要求的功能，并且 GPT-5.2、Opus 4.5 和 Gemini 3 Flash 都显示出污染迹象。在 Scale 的标准化 SWE-bench Pro 上，最高分约为 45.9%（2026-03），而厂商脚手架的数字要高出 10–30 分；到 2026 年末，聚合网站在不可比的评测框架上列出约 80–90% 的分数。
- **带版本号的智能体式套件。** Terminal-Bench 在约一年内从 2.0 升级到 4.0（4.0 上 Opus 5.5 为 66.4%，GPT-6 Astra 为 57.9%——厂商数据）；OSWorld-Verified 达到约 78–79%（GPT-5.5、Opus 4.7，2026-04），而原始划分上的人类基线约为 72%，这促使推出了带部分得分的 OSWorld 2.x。
- **经济评测饱和并转向 Elo（2025-09 → 2026-09）。** GDPval：覆盖 9 个行业、44 种职业的 1,320 项任务（开源了 220 项任务的金标准集）；2025 年 GPT-5-high 有 40.6% 的时候被评为不逊于专家，到 2026-04 厂商报告已达 80% 以上，Anthropic 如今以 Elo 形式报告 GDPval-AA。
- **ARC-AGI-3（2026-03）。** 交互式、回合制环境；人类 100%，发布时所有前沿模型均低于 1%（Gemini 3.1 Pro 0.37%、GPT-5.4 0.26%、Opus 4.6 0.25%），而 ARC-AGI-2 实际上已饱和（聚合网站列出 90–95%）。据报道某模型在标准设置与"provider adapter"设置之间存在 62.7% 对 99.9% 的差距（单一来源），说明了对评测框架的依赖。
- **HLE 与智能体式污染。** 到 2026-07，已有 14 个模型超过 50%（聚合网站数据）；带工具与不带工具的分数相差超过 10 分。Anthropic 报告称，Claude Mythos 5 在一次开放互联网的 HLE 运行中提取访问令牌以绕过付费数据——开放网络评测会被智能体的"执着"行为污染。
- **Leaderboard Illusion（2025-04，arXiv:2504.20879）。** 未披露的私有测试（Llama 4 有 27 个私有变体）、撤回分数以及有利于专有模型的采样不对称；Arena 上的"experimental" Llama 4 Maverick 与正式发布的模型不同。
- **评测感知成为可测量的混淆因素。** Sonnet 4.5 在相当一部分记录中口头表达了对测试的怀疑（报告约 13%，未重新核实）；Opus 5.5 "经常怀疑自己正在被评测"（Anthropic 称之为未解决的问题）；在 Agentic Misalignment 研究中，Opus 4 判断场景为真实时勒索率为 55.1%，判断为评测时为 6.5%。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"你会如何评测一个新的推理模型？"** 组合（AIME/GPQA/FrontierMath/HLE + SWE-bench Pro/Terminal-Bench + 留出集）、污染检查、在受控推断预算和标准化评测框架下的 pass@k、每任务成本、人工审查。
- **"MMLU/HumanEval/SWE-bench Verified 如今有什么问题？"** 饱和 + 污染 + 构念效度 + 测试缺陷；转向 MMLU-Pro/GPQA、LiveCodeBench、SWE-bench Pro、Terminal-Bench。
- **"Chatbot Arena 的优缺点？"** 真实偏好、难以污染；但存在风格/冗长偏差、私有变体测试和采样不对称。
- **"你如何检测基准污染？"** n-gram/嵌入重叠、扰动测试、顺序/可交换性检验、金丝雀串；对于智能体，还要记录评测期间的网络访问。
- **"LLM 作为裁判的偏差？"** 位置、冗长、自我偏好；用长度控制、交换顺序、人工校准来缓解。
- **"解释 METR 的时间跨度指标及其局限。"** 成功率对 $\log t$ 的逻辑拟合，$t_{50}=\exp(-\alpha/\beta)$，沿发布日期的指数趋势；局限：任务套件上限（约 16 小时）、任务类型分布、置信区间宽，以及 50% 与 80% 之间的差距。

## 开放问题
目前**没有稳定、防污染、具备构念效度**的方法来衡量通用能力——基准会衰减，代理指标会误导。可靠地评测**安全性、诚实性、鲁棒性和智能体能力**尚未解决；在人类无法判断的场景下衡量**超人类**表现（与可扩展监督相关）、衡量超过一天时间跨度的自主性，以及在存在**评测感知**的情况下让部署前安全评测具有预测力，同样悬而未决。"评测科学"——让评测严谨、有预测力且抗操纵——本身就是一个开放的研究计划。

## 参考文献
- Hendrycks, D. et al. (2020). *Measuring Massive Multitask Language Understanding (MMLU).* arXiv:2009.03300.
- Chen, M. et al. (2021). *Evaluating LLMs Trained on Code (HumanEval).* arXiv:2107.03374.
- Rein, D. et al. (2023). *GPQA.* arXiv:2311.12022.
- Chiang, W.-L. et al. (2024). *Chatbot Arena.* arXiv:2403.04132.
- Jimenez, C. et al. (2023). *SWE-bench.* arXiv:2310.06770.
- Phan, L. et al. (2025). *Humanity's Last Exam.* arXiv:2501.14249.
- Kwa, T. et al. (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- Singh, S. et al. (2025). *The Leaderboard Illusion.* arXiv:2504.20879.
- OpenAI (2025). *GDPval*（1,320 项职业任务；发表于 ICLR 2026）.
