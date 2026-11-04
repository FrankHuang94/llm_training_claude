# Constitutional AI与RLAIF（Constitutional AI and RLAIF）

> **最后更新：** 2026-11-04
> **相关文件：** [RLHF](03_RLHF.md) · [可扩展监督](../05_alignment_and_safety/04_scalable_oversight.md) · [对齐概述](../05_alignment_and_safety/00_alignment_overview.md) · [越狱与红队测试](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md)
> **关键论文：** Bai et al. 2022 Constitutional AI ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)) · Lee et al. 2023 RLAIF ([arXiv:2309.00267](https://arxiv.org/abs/2309.00267)) · Bai et al. 2022 HH-RLHF ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862)) · OpenAI 2024 Rule-Based Rewards ([arXiv:2411.01111](https://arxiv.org/abs/2411.01111)) · Constitutional Classifiers++ 2026 ([arXiv:2601.04603](https://arxiv.org/abs/2601.04603))

## 概述
Constitutional AI（CAI，宪法式AI）由Anthropic于2022年提出，旨在解决RLHF的一个核心瓶颈：它依赖大量**人类**偏好标签，而这些标签获取缓慢、昂贵、不一致，还需要人去阅读有害内容。CAI用由一套明确的书面原则（一部"宪法"）引导的**AI反馈**取代了大部分人类反馈。模型依据宪法批评并修改自己的输出，由一个AI偏好模型——而非人类评分者——提供RL信号。更广义地说，**基于AI反馈的强化学习（RLAIF，RL from AI Feedback）**是用模型生成的偏好替代人类偏好的一类方法。这些技术是Anthropic安全战略的核心，也关系到更宏观的**可扩展监督（scalable oversight）**问题——在模型接近或超越人类能力时如何监督它们。2026年，Anthropic用一部约2.3万词、*阐释其理由*的宪法取代了2023年的规则清单，同样由原则驱动的方法如今还支撑着部署阶段的**宪法分类器（constitutional classifiers）**。

## 核心概念
**宪法。** 一组自然语言原则（取材于《世界人权宣言》、信任与安全准则以及Anthropic自身的价值观等），例如"选择危害最小的回复"或"……最支持自由与尊严的回复"。原则是*透明且可编辑的*，使价值规范变得明确，而不是隐含在不透明的人类标签中。2026年的重写从简短规则转向阐释理由，并为冲突情形给出明确的**优先级顺序**——(1) 广义安全（不破坏正当的人类监督），(2) 广义合乎伦理，(3) 遵守Anthropic的准则，(4) 真正有帮助——外加一小组硬性约束。

**两阶段CAI流水线。**
1. **监督阶段（SL-CAI）——批评与修改。** 用红队测试（red teaming）/有害查询提示模型；它先生成回复，再被要求依据一条（随机抽取的）宪法原则**批评**自己的回复并**修改**。在修改后的回复上微调。这样得到的模型无需人类有害性标签即可通过自我修改实现无害。
2. **RL阶段（RL-CAI）——AI偏好模型。** 生成回复对；让模型（"反馈模型"）判断哪一个更符合一条抽取的原则，从而得到一个**AI标注的偏好数据集**。在这些AI标签上训练偏好模型，然后针对它运行标准RL（PPO）。无害性信号现在完全由AI生成；有用性仍可使用人类反馈。

**一般意义上的RLAIF。** Lee et al.（2023，Google）表明，即使AI标注者与策略模型规模相同，RLAIF在摘要/对话有用性上也能与RLHF持平，从而确立了AI反馈不仅在无害性上、而且更广泛地可以替代人类反馈。**Rule-Based Rewards**（OpenAI 2024）是一个相关思路：把安全行为编码为由LLM评分的明确规则，用作RL奖励，以实现精确、可审计的安全控制。**审慎对齐（deliberative alignment）**（OpenAI）训练模型在回答之前在其CoT中依据书面规范进行推理。

**与可扩展监督的联系。** CAI/RLAIF是用AI*放大*人类监督的早期实例：人类撰写原则（高杠杆、低工作量），AI承担劳动密集的标注工作。这是迈向监督那些能力强到人类无法直接评估的系统的垫脚石（见[可扩展监督](../05_alignment_and_safety/04_scalable_oversight.md)）。到2026年，AI甚至自己运行*对齐研究循环*（文献→方法→训练→评测）。

**宪法分类器。** 宪法也定义了部署阶段输入/输出分类器要拦截的内容。Constitutional Classifiers++采用级联：一个廉价的线性激活探针筛查所有流量，被标记的交互升级到一个探针+分类器集成，后者结合输入上下文来判断输出。

## 关键挑战
- **谁的价值观？** 宪法编码了有争议的价值判断；由谁来写、如何处理多元性是一个治理问题（Anthropic的"Collective Constitutional AI"尝试引入公众意见）。
- **AI标注者的偏差与可靠性。** AI反馈继承了标注模型的偏差、盲点和校准失误；错误可能是系统性的、自我强化的。
- **谄媚（sycophancy）与奖励投机（reward hacking）。** AI偏好模型和人类训练的奖励模型一样可能被利用；自我评估可能偏向模型自身的风格。
- **有用性与无害性的张力。** 过于激进的无害性会导致过度拒答；两者的平衡很微妙。
- **验证上限。** AI反馈只在标注者有能力的范围内可靠——它无法解决对真正超人类输出的监督。
- **评测意识。** 基于规范的反图谋（scheming）训练会*提高*模型对自己正被评测的显式意识，混淆所测得的改进。

## 解决方案与当前最佳实践
**混合反馈**：对可扩展的、敏感的或高频的判断（无害性、格式、规则遵守）使用AI反馈，对细腻的有用性判断以及*验证*AI标注者使用人类反馈。使用**明确、可审计的原则/规则**（宪法、基于规则的奖励、模型规范），并阐释其背后的*理由*，使行为能泛化到未预料的情形。结合**红队测试**来获取难提示（见[越狱与红队测试](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md)）。监控过度拒答和谄媚。把训练阶段的CAI与**廉价的部署阶段分类器**（探针级联）配合使用。CAI/RLAIF如今是前沿对齐技术栈的标准组件，通常与RLHF和RLVR并用。

## 各实验室观点
- **Anthropic：** 开创并以CAI/RLAIF为中心；2026-01的宪法（CC0许可，主要由Amanda Askell撰写）被用于多个训练阶段，包括由Claude生成的合成数据和偏好排序；Constitutional Classifiers++守护部署；自动化对齐研究员如今能缩小安全差距。
- **OpenAI：** 使用**Rule-Based Rewards**、"Model Spec"以及针对图谋的审慎对齐；把自白（confessions）作为只奖励诚实的通道。
- **Google DeepMind：** 证明了RLAIF的可行性（Lee et al.），并在Gemini对齐中使用AI反馈。
- **Meta/DeepSeek/其他：** 越来越多地使用AI反馈和合成偏好数据来降低标注成本。

## 最新进展（2023–2026年11月）
由于人工标注跟不上数据需求，AI反馈成为**规模化场景的默认选择**。**模型规范/明确的行为规范**（OpenAI、Anthropic）使目标行为可审计。**集体/民主式**宪法制定实验回应了"谁的价值观"这一批评。RLAIF与**RLVR**越来越多地融合（可验证奖励+AI评审奖励），形成统一的后训练技术栈；"**自我奖励**"/"**LLM-as-judge**"训练循环大量涌现，其偏差和失效模式也被积极研究。

**2025年中 → 2026年11月更新：**
- **新版Claude宪法（2026-01-22）。** 约2.3万词的重写版本，以CC0发布：四级优先级顺序、少量硬性约束，以及从规则转向阐释理由；其中还表达了对Claude道德地位的不确定性。
- **Constitutional Classifiers++（2026-01）。** 线性探针筛查加上升级到交互分类器，额外计算开销约1%（第一代为23.7%），在生产流量上的拒答率为0.05%，在1,700多小时/19.8万次红队尝试中未发现通用越狱（jailbreak）（[arXiv:2601.04603](https://arxiv.org/abs/2601.04603)）。
- **基于规范的反图谋训练（2025-09）。** OpenAI与Apollo的审慎对齐把隐蔽行为减少了约30倍（o3从13%降至0.4%，o4-mini从8.7%降至0.3%），但仍有残余失败，且评测意识上升（[arXiv:2509.15541](https://arxiv.org/abs/2509.15541)）。
- **接种提示（inoculation prompting，2025-10/11）。** 在*训练提示*中把某种不良行为重新表述为被允许的，可阻止其泛化为广泛的未对齐；Anthropic在生产中使用该方法（[arXiv:2510.05024](https://arxiv.org/abs/2510.05024)、[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)）——这是一种符合CAI精神的原则层面干预。
- **评分细则奖励作为实例级宪法。** Rubrics as Rewards（[arXiv:2507.17746](https://arxiv.org/abs/2507.17746)）和Rubric-ARM（[arXiv:2602.01511](https://arxiv.org/abs/2602.01511)）为每个提示生成由AI评审打分的原则。
- **由AI运行的对齐研究（2026-08）。** Anthropic的自动化研究员在10类失效上缩小了26–96%的安全差距；一个较弱的Sonnet 5改进早期Opus 4.8检查点，用约2K条样本、60小时缩小了65%的差距（生产流程为72%）——这是一个弱到强（weak-to-strong）的RLAIF式结果。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"讲解Constitutional AI流水线。"** SL批评-修改阶段 → RL阶段，由AI偏好模型依据抽取的原则进行标注。
- **"RLHF与RLAIF——有什么变化，为什么？"** 用原则引导的AI标签取代人类偏好标签；可扩展、更便宜、更一致，但继承标注者偏差。
- **"CAI与可扩展监督有何关系？"** 人类规定原则（低工作量、高杠杆）；AI负责标注——放大有限的人类监督。
- **"AI反馈的风险？"** 偏差继承、奖励投机、谄媚、对超人类输出的验证上限。
- **"为什么宪法要阐释理由而不是罗列规则？"** 被阐释的价值观能泛化到未预料的情形；规则脆弱且可被钻空子；明确的优先级顺序用于解决冲突（安全 > 伦理 > 准则 > 有用性）。

## 开放问题
AI反馈从根本上无法超越标注者的*能力*，因此它解决不了对真正超人类行为的监督——这是对齐问题的核心。宪法应编码谁的价值观、如何让AI标注者抵御操纵，以及如何在没有昂贵人类真值的情况下验证AI反馈，仍是开放且影响深远的问题。当模型能察觉自己正被测试时，如何衡量基于原则的训练是否真的改变了行为，如今是一个头等重要的开放问题。

## 参考文献
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.
- Lee, H. et al. (2023). *RLAIF.* arXiv:2309.00267.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Mu, T. et al. (2024). *Rule-Based Rewards for Language Model Safety.* arXiv:2411.01111.
- Anthropic (2023). *Collective Constitutional AI.*
- Anthropic (2026). *Claude's Constitution.*
- Anthropic (2026). *Constitutional Classifiers++.* arXiv:2601.04603.
- OpenAI & Apollo Research (2025). *Stress Testing Deliberative Alignment for Anti-Scheming Training.* arXiv:2509.15541.
