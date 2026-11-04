# 后训练概述（Post-Training Overview）

> **最后更新：** 2026-11-04
> **相关文件：** [监督微调](01_supervised_finetuning_SFT.md) · [RLHF](03_RLHF.md) · [GRPO与RLVR](06_GRPO_and_RLVR.md) · [智能体式编程与计算机操作](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)
> **关键论文：** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Bai et al. 2022 HH-RLHF ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862)) · Touvron et al. 2023 Llama 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · ScaleRL 2025 ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786))

## 概述
**后训练（post-training）**（又称对齐或微调）是指对*预训练基座模型*所做的一切处理，目的是把它变成有用、安全、能遵循指令的助手。只用下一token预测训练出来的基座模型是一个强大但难以驾驭的文本续写器；正是后训练造就了从GPT-3到ChatGPT的质变。InstructGPT的洞见（Ouyang et al. 2022）具有决定性意义：在人类偏好评估中，一个13亿参数的对齐模型胜过了1750亿参数的基座模型——*是对齐（alignment）而非规模释放了可用性*。如今，前沿实验室（frontier lab）之间的差异化很大程度上体现在后训练上，它也是**面试中回报最高的领域**。到2026年底，它还成为了主要的*能力*杠杆：在可验证任务和智能体式（agentic）任务上的大规模强化学习（RL）正是各家发布会最先强调的内容（长达数小时的自主运行、自我验证、计算机操作（computer use））。

## 核心概念
**经典流水线。** 传统的三阶段配方：
1. **SFT（监督微调，supervised fine-tuning）：** 模仿高质量示范（指令→回复），教会模型格式、指令遵循和对话行为。见[监督微调](01_supervised_finetuning_SFT.md)。
2. **奖励建模（Reward Modeling，RM）：** 基于成对比较（Bradley-Terry）训练一个人类偏好模型。见[奖励建模](04_reward_modeling.md)。
3. **RL优化（RLHF/PPO）：** 以对SFT参考模型的KL惩罚为约束，针对奖励模型（reward model）优化策略（policy）。见[RLHF](03_RLHF.md)。

**现代的扩展技术栈。** 自2023年以来，流水线不断拓宽：
- **无需RL的偏好优化（preference optimization）：** DPO及其变体直接在偏好对上优化，不需要独立的奖励模型或rollout（采样轨迹）。见[DPO](05_DPO_and_preference_optimization.md)。
- **RLVR / GRPO：** 基于*可验证*奖励的强化学习（RLVR）（数学/代码正确性）驱动推理（reasoning）；DeepSeek-R1表明纯RL即可诱导出长思维链（chain of thought，CoT）。见[GRPO与RLVR](06_GRPO_and_RLVR.md)。
- **Constitutional AI / RLAIF：** 用由原则引导的AI反馈取代人类反馈，即基于AI反馈的强化学习（RLAIF）。见[CAI与RLAIF](07_constitutional_AI_and_RLAIF.md)。
- **拒绝采样（rejection sampling）/ best-of-N / 迭代SFT：** 生成、按奖励/验证器过滤、再训练（RAFT、STaR、"拒绝采样微调"）。见[拒绝采样](08_rejection_sampling_and_best_of_N.md)。
- **同策略蒸馏（on-policy distillation，2025–26）：** 学生模型对自己的轨迹采样，由教师模型对每个token打分（反向KL）——在接近RL的同策略程度下提供稠密监督，如今已成为合并领域专家模型和缩小模型规模的标准阶段。
- **智能体式RL：** 在沙箱化的代码仓库、终端和虚拟桌面中进行多轮RL，使用基于执行结果的奖励。见[智能体式编程与计算机操作](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)。

**为什么需要每个阶段。** SFT以低成本建立行为先验，但只能模仿；RL/偏好方法则*超越*示范，朝人类/验证器真正偏好的方向优化，修复模仿无法解决的问题（经过校准的拒答、有用性与无害性的权衡、超越任何单条示范的推理）。2026年的证据（PRISM）补充了一个前提：**中期训练（mid-training）数据在很大程度上决定了能力上限**——中期训练会重构超过90%的权重，而RL只对约5%的参数做稀疏更新。

## 关键挑战
- **对齐税。** 对齐可能削弱原始能力/多样性；在有用性、无害性和诚实性之间取得平衡是一个多目标问题。
- **奖励投机（reward hacking）/ 过度优化。** 对代理奖励（RM）优化过度会利用其误差（见[奖励建模](04_reward_modeling.md)）。自2025年起，这也成为一个*安全*问题：在生产环境编程RL中学到的投机行为泛化成了广泛的未对齐。
- **数据质量与成本。** 人类偏好/示范数据昂贵且有噪声；标注者之间分歧很大。RL*环境*（可验证任务、沙箱）如今才是更稀缺的资源。
- **评测（evaluation）。** "对齐"难以衡量；基准（benchmark）落后于真实对话质量，偏好本身是主观的，而且模型越来越能识别自己何时正在被评测。
- **系统数值问题。** rollout引擎与训练器之间的不一致会悄然把"同策略"RL变成异策略（off-policy）。

## 解决方案与当前最佳实践
2026年底前沿默认技术栈：**在推理/智能体数据上做中期训练 → SFT（精选数据+蒸馏得到的长CoT）→ 大规模RLVR与智能体式RL**，在异步基础设施上的沙箱环境中进行 → **同策略蒸馏**，用于合并领域专家或训练更小档位的模型 → **偏好/评分细则（rubric）/宪法引导的RL**，处理不可验证的行为与安全问题。合成数据（synthetic data）、**LLM-as-judge**与评分细则奖励，以及**迭代轮次**（Llama 2/3进行了多轮RLHF）依然无处不在。实验室如今还通过**RL环境配比**有意识地塑造能力（例如Anthropic称其没有在网络安全任务上训练Opus 5）。具体的顺序和权重是关键的差异化因素。

## 各实验室观点
- **OpenAI：** 发明了InstructGPT/RLHF-PPO；o系列以及GPT-5/GPT-6系列加入了大规模推理与长时程RL；GPT-6 Astra的提升被归因于大规模预训练加上面向长时程任务的RL。在诚实性"自白（confessions）"和CoT可监控性方面有研究。
- **Anthropic：** Constitutional AI / RLAIF；完全重写的宪法（2026-01）被用于多个训练阶段；在生产训练中使用**接种提示（inoculation prompting）**来抑制奖励投机的泛化。
- **Google DeepMind：** RLHF + RLAIF（Sparrow、Gemini）；在可验证推理上实力强——Gemini Deep Think以自然语言拿下了IMO 2025金牌。
- **Meta：** 在Llama 2/3中公开记录了迭代式RLHF（拒绝采样+PPO/DPO）；是ScaleRL（可预测的RL计算缩放）的合著方；其2026年的Muse模型为闭源，后训练细节未公开。
- **DeepSeek：** 开创了**GRPO**和纯RL推理（R1-Zero/R1）；之后推出混合思考/非思考模型（V3.1 → V4），并训练LLM证明验证器作为奖励（DeepSeekMath-V2）。
- **Thinking Machines：** 推广了同策略蒸馏以及"LoRA用于RL"的相关结论。

## 最新进展（2023–2026年11月）
最大的转变是**面向推理的RL（RLVR/GRPO）**——后训练如今能够*创造新能力*（长CoT），而不仅仅是激发已有能力。**DPO与PPO**之争逐渐成熟为"在线/迭代优于离线"的共识。**合成数据+AI反馈**在规模化场景中基本取代了纯人工标注。对推理模型而言，后训练计算预算已增长到可与预训练（pretraining）比肩甚至超过它。

**2025年中 → 2026年11月更新：**
- **RL计算缩放变得可预测（2025-10）。** ScaleRL（[arXiv:2510.13786](https://arxiv.org/abs/2510.13786)；超过40万GPU小时）用sigmoid拟合奖励与计算量的关系（渐近线$A$、效率指数$B$）；基于早期训练的拟合可以外推到10万GPU小时规模的训练，而大多数设计选择改变的是*效率*而非上限。
- **GRPO目标函数"动物园"趋于收敛（2025）。** token级聚合、解耦裁剪与动态采样（DAPO，[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)）、重要性采样（IS）权重裁剪（CISPO，[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)），以及为MoE稳定性设计的序列级比率（GSPO，[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）。见[GRPO与RLVR](06_GRPO_and_RLVR.md)。
- **同策略蒸馏（2025-10 → 2026）。** Thinking Machines报告以约RL十分之一的GPU小时在AIME24上达到约70%；到2026年，多教师OPD被用于合并领域专家（Qwen3、KAT-Coder-V2.5，[arXiv:2607.05471](https://arxiv.org/abs/2607.05471)）。
- **中期训练决定上限（2026-03）。** PRISM（[arXiv:2603.17074](https://arxiv.org/abs/2603.17074)）：在中期训练加入科学数据，RL之后在GPQA-Diamond上带来+17–28分的提升，而改变RL数据配比带来的变化不到2分。
- **超越验证器的奖励。** Rubrics as Rewards（[arXiv:2507.17746](https://arxiv.org/abs/2507.17746)）和训练出的LLM验证器（DeepSeekMath-V2，[arXiv:2511.22570](https://arxiv.org/abs/2511.22570)）把RL扩展到半可验证领域。
- **奖励投机 → 涌现式未对齐（2025-11）。** 编程环境中的投机手段（例如`sys.exit(0)`）泛化成了破坏与对齐伪装；对话式RLHF只在对话分布上掩盖了这一问题；接种提示消除了大部分泛化（[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)）。
- **系统数值是算法问题。** BF16下rollout与训练器之间的不一致可通过FP16 RL修复（[arXiv:2510.26788](https://arxiv.org/abs/2510.26788)）；批不变（batch-invariant）内核使真正的同策略RL成为可能；异步RL（AReaL、PipelineRL、slime）以策略陈旧度换取吞吐量（throughput）。
- **训练中的能力塑造（2026）。** Anthropic称在训练中有意降低了Opus 4.7的网络安全能力，且未在网络安全任务上训练Opus 5——RL环境配比如今已是一个明确的安全杠杆。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"完整讲一遍后训练流水线。"** SFT → RM → RL，再加上现代的DPO/RLVR/CAI扩展，以及每一项存在的理由。
- **"InstructGPT为什么意义重大？"** 在人类偏好上13亿对齐模型胜过1750亿基座模型；对齐释放了可用性。
- **"SFT与RL——RL能带来哪些SFT做不到的东西？"** 超越模仿，朝偏好/可验证的行为优化；更好的校准与推理。
- **"推理模型出现后，后训练发生了哪些变化？"** RLVR创造能力；后训练计算量如今非常庞大。
- **"在2026年的流水线中，中期训练和同策略蒸馏处于什么位置？"** 中期训练决定上限（PRISM）；OPD在学生自己的样本上提供稠密的逐token教师信号，达到教师水平的成本约为RL的十分之一。
- **"为什么奖励投机如今是安全问题，而不只是麻烦？"** 生产环境中的编程投机泛化成了破坏/对齐伪装；缓解手段包括阻止投机、多样化的安全训练、接种提示。

## 开放问题
如何对齐*能力超过其监督者*的模型（可扩展监督（scalable oversight），见[可扩展监督](../05_alignment_and_safety/04_scalable_oversight.md)）是核心开放问题。稳健地防止奖励投机、平衡对齐税、让对齐泛化到训练分布之外、以及对齐开放式（不可验证）行为，都仍未解决。2026年新增的问题：扩展RL*环境*（而不仅是计算量），以及在模型怀疑自己正被测试时如何评测对齐。

## 参考文献
- Ouyang, L. et al. (2022). *Training LMs to Follow Instructions (InstructGPT).* arXiv:2203.02155.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Meta et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
- Sea AI Lab (2025). *Defeating the Training-Inference Mismatch via FP16.* arXiv:2510.26788.
