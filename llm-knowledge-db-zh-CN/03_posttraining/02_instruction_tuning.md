# 指令微调（Instruction Tuning）

> **最后更新：** 2026-11-04
> **相关文件：** [监督微调](01_supervised_finetuning_SFT.md) · [合成数据生成](../01_data/04_synthetic_data_generation.md) · [后训练概述](00_posttraining_overview.md)
> **关键论文：** Wei et al. 2021 FLAN ([arXiv:2109.01652](https://arxiv.org/abs/2109.01652)) · Sanh et al. 2021 T0 ([arXiv:2110.08207](https://arxiv.org/abs/2110.08207)) · Chung et al. 2022 Scaling Instruction-Finetuning ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416)) · Wang et al. 2022 Super-NaturalInstructions ([arXiv:2204.07705](https://arxiv.org/abs/2204.07705))

## 概述
指令微调（instruction tuning）是监督微调（SFT）的一种特定且影响深远的形式：在*大量、多样、被改写为自然语言指令的NLP任务集合*上微调，其明确目标是**对未见任务实现零样本泛化**。里程碑式的结果（FLAN，Wei et al. 2021）表明，在许多任务上做指令微调能大幅提升模型对*调优期间从未见过的任务*的指令遵循能力——指令遵循是一种可迁移的元技能。这与对话式SFT（以对话行为为目标）不同；指令微调的目标是*任务泛化*。两者在实践中早已融合，但这一概念区分以及FLAN系列工作仍是常见的面试素材。到2026年，指令格式的数据已经*向上游*迁移到中期训练（mid-training）乃至预训练（pretraining）（结构化改写），并*向下游*进入了RL（以实例级评分细则作为奖励）。

## 核心概念
**任务即指令的改写。** 取现有的监督数据集（情感分析、NLI、问答、摘要、翻译），把每一个都包装进自然语言的**指令模板**中（"请判断这条评论的情感：……"）。在数百到数千个此类任务上训练的模型会学到"读指令→执行任务"这一抽象。

**留出任务评测。** 关键方法论：把任务聚类，整*簇*留出，在其余任务上做指令微调，然后在留出的簇上做**零样本**评测。在那里的提升证明的是真正的泛化，而非记忆。

**缩放发现（Chung et al. 2022）。** 有三个维度能改善指令微调：**(1) 更多任务**（扩展到1.8k个任务），**(2) 更大的模型**，以及**(3) 加入思维链（chain of thought，CoT）**数据（CoT指令微调释放了推理泛化能力）。Flan-PaLM/Flan-T5在零样本和CoT上都取得了大幅提升；关键在于，指令微调+CoT数据保持乃至提升了推理能力，而非损害它。

**数据集。** FLAN集合、T0的P3（Public Pool of Prompts）、Super-NaturalInstructions（1.6k个任务，社区编写）、OpenAssistant（人类对话）、Dolly、ShareGPT（真实ChatGPT日志），以及合成指令集（Self-Instruct、Evol-Instruct）。现代数据配比（data mixture）混合了任务指令、对话、推理、智能体式（agentic）/工具使用（tool use）以及安全数据。

**指令层级与持久化指令。** 现代助手必须遵循来自多个主体、具有明确优先级的指令（system/developer > user > 工具输出），而智能体（agent）还要遵循长期存在、以文件形式提供的指令（AGENTS.md、SKILL.md文件夹）。因此，指令遵循是*带优先级地*被训练的，而非一种扁平的技能。

## 关键挑战
- **任务多样性与格式多样性。** 泛化既需要多样的*任务*，也需要多样的*措辞*；模板过窄会导致对模板表层形式的过拟合。
- **负迁移/能力退化。** 如果配比失衡，朴素的指令微调可能损害开放式生成或推理（早期FLAN损害了部分少样本能力；CoT数据修复了推理问题）。
- **标注成本与噪声。** 人工任务集合很昂贵；改写过程会引入模板痕迹。
- **评测泄漏（leakage）。** 任务多达数千个时，很难保证留出集的完整性（见[数据污染](../01_data/05_data_contamination_and_evaluation_leakage.md)）。
- **相互冲突的指令。** 工具输出或文档中被注入的指令必须被*忽略*；朴素的指令遵循本身就是攻击面（提示注入（prompt injection））。

## 解决方案与当前最佳实践
**混合大量任务+大量模板+CoT+对话+安全数据**，并平衡以避免偏斜。加入**推理（CoT）示例**来保持/增强推理能力。把**学术任务集合**（FLAN/SNI）与**真实及合成的对话**数据（ShareGPT/OpenAssistant/Evol-Instruct）结合——学术任务提供广度，对话数据提供自然的助手行为。去重并去污染。如今指令微调通常被并入偏好优化（preference optimization）之前更宽泛的SFT阶段，同时类指令的结构化数据会更早引入（中期训练），并在后期用**基于评分细则的奖励**强化指令遵循。

## 各实验室观点
- **Google**开创并扩展了指令微调（FLAN、与T0的合作、Flan-T5/PaLM）——这是奠基性的路线。
- **OpenAI**将其纳入InstructGPT的示范数据与对话对齐（alignment）之中；Model Spec和AGENTS.md把指令优先级和仓库级指令形式化。
- **Anthropic**在其2026年宪法中把有用性（包括指令遵循）放在明确优先级顺序的*最后*（广义安全 > 广义合乎伦理 > 遵守准则 > 真正有帮助）。
- **AllenAI/社区**推动了开放指令数据集（Tülu、SNI、Open-Instruct）和系统化的配比研究；OLMo 3（2025-11）发布了完全开放的Instruct和Think变体，包括数据和检查点。
- **Meta/Mistral/DeepSeek/Qwen**在各自的SFT阶段中混合了指令、对话和合成推理数据。

## 最新进展（2023–2026年11月）
前沿从人工任务集合转向了**合成的、演化的、富含推理的**指令数据（Evol-Instruct、Orca、长CoT蒸馏）。**Tülu**（AllenAI）对指令微调配比及其与DPO的相互作用提供了开放、严谨的研究。"指令微调"与"对话SFT"之间的界线变得模糊；而持久的洞见——**指令遵循能跨任务泛化，并受益于CoT**——支撑着所有现代助手。

**2025年中 → 2026年11月更新：**
- **结构化改写进入预训练（2026-04）。** FinePhrase（[arXiv:2604.13977](https://arxiv.org/abs/2604.13977)）研究了超过1T生成token：把网页数据改写为结构化目标形式（表格、数学题、FAQ、教程）优于精选网页数据和先前的合成方法；超过约1B参数的生成器没有额外收益；发布了486B token。
- **RL训练的改写器（2026-05）。** SynPro（[arXiv:2605.17849](https://arxiv.org/abs/2605.17849)）按质量、忠实度和数据影响力奖励改写器，在相当于Chinchilla 10%的数据规模下，报告的有效token数是简单重复的3.7–5.2倍——"数据墙"部分上是一个指令改写问题。
- **中期训练时机（2025-10）。** 早期引入专业/类指令数据可以容忍较高的配比权重，晚期引入则需要较低权重（[arXiv:2510.14865](https://arxiv.org/abs/2510.14865)）。
- **指令变成奖励（2025-07）。** Rubrics as Rewards（[arXiv:2507.17746](https://arxiv.org/abs/2507.17746)）把每条指令转化为由LLM评审打分的实例级检查清单：相比Likert式评审，在HealthBench上带来最高+31%的相对提升。
- **仓库级指令标准化（2025-12 → 2026-10）。** AGENTS.md（OpenAI）和Agent Skills（采用渐进式披露的SKILL.md文件夹）加入了Linux基金会的Agentic AI Foundation，使"遵循项目的指令文件"成为编程智能体被训练出的行为（见[智能体式编程与计算机操作](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)）。
- **专业化之后恢复指令遵循。** 同策略蒸馏（on-policy distillation）恢复了在领域中期训练中丢失的指令遵循能力（Thinking Machines，2025-10）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"FLAN证明了什么，为什么重要？"** 多任务指令微调 → 对未见任务的零样本泛化；指令遵循是一种可迁移的元技能。
- **"如何评测指令微调的泛化？"** 留出*任务簇*，零样本评测。
- **"哪三件事能扩展指令微调？"** 更多任务、更大模型、CoT数据（Chung et al.）。
- **"指令微调与RLHF各自提供什么？"** 指令微调通过模仿提供任务遵循的广度；RLHF则超越示范优化偏好。
- **"2026年指令微调还是一个独立阶段吗？"** 大体上不是：指令格式的数据出现在预训练（结构化改写）、中期训练、SFT和RL（评分细则奖励）中；仍然独立存在的问题是注入攻击下的指令*层级*。

## 开放问题
指令遵循为何能泛化得如此之好（跨任务迁移的机制）尚未完全理解。最优的任务/格式/CoT配比组成、避免对开放式能力的负迁移，以及在强基座模型和RL面前大规模指令微调是否仍有必要，都是开放问题。在长的智能体上下文中稳健地区分合法指令与被注入的指令，仍未解决。

## 参考文献
- Wei, J. et al. (2021). *Finetuned LMs Are Zero-Shot Learners (FLAN).* arXiv:2109.01652.
- Sanh, V. et al. (2021). *Multitask Prompted Training (T0).* arXiv:2110.08207.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned LMs.* arXiv:2210.11416.
- Wang, Y. et al. (2022). *Super-NaturalInstructions.* arXiv:2204.07705.
- Ivison, H. et al. (2023). *Tülu / Camels in a Changing Climate.* arXiv:2311.10702.
- Scale AI (2025). *Rubrics as Rewards.* arXiv:2507.17746.
- FinePhrase authors (2026). *FinePhrase.* arXiv:2604.13977.
