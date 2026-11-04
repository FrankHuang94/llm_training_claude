# 监督微调（Supervised Fine-Tuning, SFT）

> **最后更新：** 2026-11-04
> **相关文件：** [后训练概述](00_posttraining_overview.md) · [指令微调](02_instruction_tuning.md) · [合成数据生成](../01_data/04_synthetic_data_generation.md) · [LoRA及其变体](../06_efficiency/01_LoRA_and_variants.md)
> **关键论文：** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Zhou et al. 2023 LIMA ("Less Is More for Alignment") ([arXiv:2305.11206](https://arxiv.org/abs/2305.11206)) · Taori et al. 2023 Alpaca · Chung et al. 2022 Scaling Instruction-Finetuning (FLAN) ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416)) · Midtraining Bridges Pretraining and Posttraining 2025 ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865))

## 概述
监督微调（supervised fine-tuning，SFT）是第一个、也是成本最低的后训练（post-training）阶段：在精选的（提示，理想回复）对数据集上继续做下一token训练，让基座模型学会助手的*格式*与*行为*——遵循指令、使用对话结构、恰当地拒答，并给出有帮助、格式良好的回答。SFT本质上是"行为克隆"：模型模仿示范。它建立起后续RL/偏好阶段要进一步打磨的先验。一个核心且有些出人意料的发现（LIMA，Zhou et al. 2023）是：**少量（约1k条）质量极高、足够多样的样本**就能带来很强的指令遵循能力，这说明基座模型其实已经"知道"大部分能力，SFT主要是*激发风格与格式*——即"表层对齐假说（Superficial Alignment Hypothesis）"。在2026年的流水线中，SFT越来越多地被夹在**中期训练（mid-training）**（决定能力上限）与**RL/同策略蒸馏（on-policy distillation）**（纠正SFT的异策略弱点）之间。

## 核心概念
**目标函数与损失掩码。** SFT使用标准的交叉熵语言模型损失，但**只在助手/回复token上计算**——提示/用户token（以及系统提示）被从损失中掩去。这样可以避免模型学会*生成*用户轮次，把容量集中在产出好的回复上：
$$\mathcal{L}_{SFT}=-\sum_{i\in \text{response}}\log p_\theta(x_i\mid x_{<i})$$
忘记掩去提示是一个会降低质量的经典bug。

**对话模板。** 多轮对话通过特殊控制token来划分角色并序列化。常见格式：
- **ChatML**（OpenAI）：`<|im_start|>role ... <|im_end|>`。
- **Llama-2/3 chat**：`[INST] ... [/INST]`（Llama 2）/ Llama 3的header token `<|start_header_id|>role<|end_header_id|>`。
- **Mistral**：`[INST] ... [/INST]`。
模板在训练和推断（inference）时必须**完全一致**；不一致会导致严重退化。模板还编码系统提示、工具调用结构，并且——自混合推理模型出现以来（例如DeepSeek-V3.1的Think/Non-Think模式，2025-08）——还编码**推理模式开关和思考块**，使同一个检查点（checkpoint）可以同时针对两种模式做SFT。

**打包（packing）。** 为避免在padding上浪费计算，多个短样本被**打包**进一条长度达到上下文长度的序列，并通过注意力掩码（或文档分隔符/重置position id）确保样本之间不会跨边界相互注意。这能显著提升吞吐量（throughput）。

**多轮处理。** 每个助手轮次都贡献损失；更早的轮次作为上下文。需要注意让模型学会以完整的对话历史为条件。对于智能体（agent）轨迹，工具输出与用户轮次一样被掩去，只训练模型自己的动作/思考。

**异策略蒸馏与同策略蒸馏。** 经典的SFT式蒸馏（distillation）在*教师*样本上训练（异策略，off-policy），因此学生从未学会从自己的错误中恢复。**同策略蒸馏**（Thinking Machines，2025-10）则让学生自己采样，由教师用反向KL对每个token打分，即在$y\sim\pi_\theta$上最小化$\mathcal{L}=\sum_t \mathrm{KL}\big(\pi_\theta(\cdot\mid y_{<t})\,\|\,\pi_{T}(\cdot\mid y_{<t})\big)$——提供稠密监督，且没有曝光偏差。

## 关键挑战
- **数据质量起决定作用。** SFT对示范质量极其敏感；少量坏样本（格式错误、幻觉（hallucination）事实、谄媚（sycophancy））就会扩散。质量≫数量（LIMA）。
- **模仿上限。** SFT只能复现示范过的行为；它无法朝数据之外*更受偏好*的方向优化，也无法从负例中学习——这正是引入RL/DPO的动机。
- **分布偏移/曝光偏差。** 训练时以标准前缀为条件，推断时却要以自己的输出为条件。
- **灾难性遗忘（catastrophic forgetting）。** 大量SFT可能侵蚀基座模型的知识/能力（"对齐税"）。
- **模板/格式脆弱性。** 训练与服务之间模板不一致会破坏行为。
- **教师可得性。** 前沿实验室如今主动抵御蒸馏（绑定账户的"保留思考（preserved thinking）"、蒸馏分类器），因此从闭源模型获取长CoT教师轨迹变得更难。

## 解决方案与当前最佳实践
**激进地精选数据**（去重（deduplication）、过滤、验证；宁可少而精、多样——LIMA/phi的理念）。使用**合成数据（synthetic data）**（Self-Instruct/Evol-Instruct/Orca）加上人工撰写的黄金数据，并对代码和数学做**执行/验证**（见[合成数据](../01_data/04_synthetic_data_generation.md)）。**掩去提示**、**打包序列**、使用**一致的对话模板**，采用适度的轮数（1–3个epoch）和较低的学习率（learning rate）以限制遗忘。SFT通常是DPO/RLHF的**热启动**；许多流水线还会混入预训练数据以保持能力。对于参数高效的SFT，应**对所有层（尤其是MLP/MoE）施加LoRA**，并使用约为全量微调10倍的学习率。存在强教师时，在SFT之后进行**同策略蒸馏**；在已验证的智能体轨迹上做拒绝采样微调，作为智能体式RL的冷启动。

## 各实验室观点
- **OpenAI：** InstructGPT基于人类示范的SFT奠定了现代配方；ChatML统一了模板。
- **Google：** FLAN/T0把*指令*SFT扩展到数千个任务以实现零样本泛化（见[指令微调](02_instruction_tuning.md)）。
- **Meta：** Llama 2/3使用精选SFT数据（数万到数百万条）加上合成数据和拒绝采样得到的回复。
- **Microsoft：** phi/Orca强调合成的、高质量的、富含推理的SFT数据。
- **Anthropic：** SFT作为为Constitutional AI / RLAIF提供输入的监督阶段；2026年的宪法被用于生成由Claude撰写的合成训练数据。
- **Thinking Machines：** LoRA与全量微调的对比研究以及同策略蒸馏研究，重新界定了SFT应该做什么、不应该做什么。

## 最新进展（2023–2026年11月）
领域转向了**合成的、经过验证的、推理密集的SFT**数据以及**长CoT SFT**（在RL之前，从R1/o1式教师蒸馏推理轨迹作为冷启动——DeepSeek-R1使用了少量高质量长CoT SFT来引导RL）。"**先SFT后RL**"仍是标准做法，争论点在于如果后面接强RL，到底需要多少SFT（R1-Zero完全跳过了SFT）。质量优先于数量（LIMA）如今已是共识。

**2025年中 → 2026年11月更新：**
- **同策略蒸馏部分取代SFT式蒸馏（2025-10 → 2026）。** 学生采样、教师打分的反向KL以约RL十分之一的GPU小时在AIME24上达到约70%，并在领域中期训练后恢复了指令遵循能力（Thinking Machines博客）。针对反向KL模式寻求（mode-seeking）问题的修正随后出现：基于熵门控的前向/反向切换（EOPD，[arXiv:2603.07079](https://arxiv.org/abs/2603.07079)）和截断top-K反向KL（[arXiv:2603.25562](https://arxiv.org/abs/2603.25562)）。
- **LoRA Without Regret（2025-09）。** 当LoRA施加于所有层时（仅注意力层的LoRA表现较差），其效果可与全量微调相当，最优学习率约为全量微调的10倍——使低成本SFT在后训练数据规模下几乎无损（Thinking Machines博客）。
- **中期训练作为分布桥梁（2025-10）。** 专业数据进入的时机与其配比权重相互作用——早期引入可用高权重，晚期引入则需要低权重；数学/代码收益最大（[arXiv:2510.14865](https://arxiv.org/abs/2510.14865)）。PRISM（[arXiv:2603.17074](https://arxiv.org/abs/2603.17074)）发现中期训练重构了超过90%的权重，而RL只稀疏更新约5%。
- **SFT穿插在RL之中。** ComputerRL的"Entropulse"在长时程智能体式RL中交替进行RL与SFT阶段以恢复熵（[arXiv:2508.14040](https://arxiv.org/abs/2508.14040)）；Nebius的SWE智能体在多轮DAPO之前，先在真实代码仓库轨迹上做拒绝采样微调（[arXiv:2508.03501](https://arxiv.org/abs/2508.03501)）。
- **混合模式SFT。** 同一套权重如今同时服务思考与非思考模式（DeepSeek-V3.1，2025-08；V4的Thinking/Non-Thinking，2026-04），模板设计因此转向显式的模式token。
- **蒸馏防御（2026）。** Anthropic Opus 5.5的"保留思考"阻止API用户通过编辑先前上下文来提取推理过程，Fable 5的分类器会把疑似蒸馏的请求路由走——减少了可用于SFT的前沿长CoT轨迹供给。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么要掩去提示token上的损失？"** 让学习集中在回复上；避免学会生成用户轮次。
- **"什么是表层对齐假说（LIMA）？"** 知识存在于基座模型中；SFT主要激发风格/格式 → 少量高质量样本就足够。
- **"什么是序列打包，为什么要用？"** 拼接样本以填满上下文，并掩蔽跨样本注意力；吞吐量大幅提升。
- **"促使人们转向RLHF/DPO的SFT局限是什么？"** 模仿上限、没有负信号、无法超越示范进行优化。
- **"SFT式蒸馏与同策略蒸馏有何不同？"** 异策略的教师样本存在曝光偏差；OPD在学生自己的样本上用稠密的逐token教师KL训练——修复误差累积；需注意反向KL的模式寻求。

## 开放问题
相对于纯RL（R1-Zero），SFT到底有多必要仍无定论。表层对齐假说的确切机制、仅靠SFT蒸馏推理的极限，以及如何在不引起遗忘/对齐税的情况下做SFT，都是开放问题。最优数据选择（"选哪1k条？"）缺乏有原则的理论。中期训练、SFT与同策略蒸馏之间如何分工，如今也是一个开放的设计问题。

## 参考文献
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Zhou, C. et al. (2023). *LIMA.* arXiv:2305.11206.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned LMs (FLAN).* arXiv:2210.11416.
- Taori, R. et al. (2023). *Stanford Alpaca.*
- DeepSeek-AI (2025). *DeepSeek-R1 (cold-start SFT).* arXiv:2501.12948.
- CMU (2025). *Midtraining Bridges Pretraining and Posttraining Distributions.* arXiv:2510.14865.
- Thinking Machines Lab (2025). *LoRA Without Regret; On-Policy Distillation.* Blog posts.
