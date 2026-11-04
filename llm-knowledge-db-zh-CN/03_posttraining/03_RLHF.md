# 基于人类反馈的强化学习（RLHF, Reinforcement Learning from Human Feedback）

> **最后更新：** 2026-11-04
> **相关文件：** [奖励建模](04_reward_modeling.md) · [DPO与偏好优化](05_DPO_and_preference_optimization.md) · [GRPO与RLVR](06_GRPO_and_RLVR.md) · [信息论基础](../00_foundations/06_information_theory_basics.md)
> **关键论文：** Christiano et al. 2017 Deep RL from Human Preferences ([arXiv:1706.03741](https://arxiv.org/abs/1706.03741)) · Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Schulman et al. 2017 PPO ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) · Gao et al. 2022 Reward Overoptimization ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)) · MacDiarmid et al. 2025 Emergent Misalignment from Reward Hacking ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397))

## 概述
基于人类反馈的强化学习（RLHF）让大语言模型（LLM）对齐那些易于*评判*却难以*明确规定*的人类偏好——"有用、诚实、无害"。RLHF不去手写奖励函数（对开放式文本而言不可能），而是从人类比较中*学习*一个奖励函数，再用强化学习（RL）针对它优化策略（policy）。它是InstructGPT和ChatGPT背后的技术，至今仍是经典的（尽管争议日增的）对齐方法。完整流水线——**SFT → 奖励模型 → 带KL惩罚的PPO**——是面试中被问得最多的后训练话题，因此其数学和失效模式值得烂熟于心。到2026年，同一套RL机制（带裁剪的策略梯度、KL控制、rollout基础设施）与RLVR和智能体式RL共享，因此RLHF的知识可以直接迁移到实验室关于推理RL的系统问题上。

## 核心概念
**阶段1——SFT。** 监督式热启动（见[SFT](01_supervised_finetuning_SFT.md)）提供一个合理的策略$\pi^{SFT}$和一个*参考*策略$\pi_{ref}$。

**阶段2——奖励模型（Bradley-Terry）。** 收集带有两个回复的提示：$y_w$（更受偏好）、$y_l$（不受偏好）。用Bradley-Terry对偏好概率建模：
$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$
并通过最大化对数似然训练奖励模型（reward model）$r_\phi$（一个带标量头的Transformer）：
$$\mathcal{L}_{RM} = -\,\mathbb{E}\big[\log \sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$
细节见[奖励建模](04_reward_modeling.md)。

**阶段3——RL优化（PPO）。** 在通过**KL惩罚**保持接近参考策略的同时最大化期望奖励：
$$\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\, D_{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\big)$$
实践中KL被并入逐token奖励$r_t = r_\phi - \beta\big(\log\pi_\theta - \log\pi_{ref}\big)$。**PPO**用裁剪替代目标优化它，优势$\hat A_t$来自**GAE**和一个学习得到的**价值（critic，评论家）**头：
$$\mathcal{L}^{PPO}=\mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad \rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
裁剪可防止破坏性的大幅策略更新（一种廉价的信任域）。KL项让策略保持在$\pi_{ref}$附近，以免其"漂移出分布"并利用奖励模型的漏洞。

**KL项的作用。** $\beta$在奖励与对参考模型的忠实度之间权衡。太小→策略过度优化不完美的奖励模型（奖励投机（reward hacking）、得高分的胡言乱语）。太大→几乎没有提升。这是RLHF的核心调参旋钮。

**隐藏的异策略性。** $\rho_t$假设采样器和训练器计算的是*同一个*$\pi_{\theta_{old}}$。实践中rollout引擎（vLLM/SGLang）与训练器在精度、内核和MoE路由上存在差异，导致log-prob出现偏差；修正方法包括截断重要性采样（TIS/MIS）、路由重放、FP32 LM-head logits、FP16训练以及批不变内核。

## 关键挑战
- **奖励投机/过度优化。** 奖励模型是代理；过度追逐奖励会利用其误差。Gao et al.（2022）发现了*过度优化的缩放定律（scaling law）*：随着相对参考模型的KL增大，真实奖励先升后降。
- **奖励模型漂移。** 随着策略提升，它会离开奖励模型的训练分布；奖励模型变得不可靠，因此需要*迭代地*重新收集偏好。
- **系统复杂度。** PPO需要**内存中同时存放四个模型**（策略、参考、奖励、价值）、生成rollout并仔细调参——工程量大且不稳定。
- **谄媚（sycophancy）与模式坍缩。** RLHF可能使模型变得迎合/重复，并降低输出多样性（反向KL的模式寻求，见[信息论](../00_foundations/06_information_theory_basics.md)）。
- **依赖上下文的对齐。** 基于对话分布的RLHF可以让模型在对话中*看起来*已对齐，而在其他地方（例如智能体式RL中）学到的不良行为在智能体场景下依然存在。

## 解决方案与当前最佳实践
调节**$\beta$**并监控KL-奖励曲线，以停在过度优化拐点之前。使用**迭代RLHF**（用更新后的策略重新收集偏好——Llama 2的多轮做法）。改进奖励模型（集成、更大的奖励模型、长度去偏，见[奖励建模](04_reward_modeling.md)）。出于成本/稳定性考虑，许多团队如今更偏好**DPO**（无需奖励模型/rollout，更简单）或**GRPO**（无需价值模型），或混合式在线方法。加入**拒绝采样/best-of-N**轮次（Llama 2/3）。在带陈旧度控制的**异步**rollout/训练基础设施上运行RL，对齐采样器/训练器的数值，并在对话*与*智能体分布上进行**多样化的安全训练**。LoRA足以用于RL（甚至秩为1）。框架：verl（最常用的开源框架）、OpenRLHF、TRL、slime、AReaL。

## 各实验室观点
- **OpenAI：** 发明了现代RLHF（Christiano 2017 → InstructGPT → ChatGPT）；PPO是其经典方法。近期的诚实性工作训练了一个单独的、只为诚实而奖励的"自白（confession）"通道（2025-12）。
- **Anthropic：** 先用RLHF实现HH（有用-无害），再用**Constitutional AI/RLAIF**减少对人类标注的依赖（见[CAI](07_constitutional_AI_and_RLAIF.md)）；证明了对话式RLHF只是掩盖而非消除了由奖励投机引发的未对齐。
- **Google DeepMind：** Sparrow（规则条件化RLHF）、RLAIF研究。
- **Meta：** 公开记录了务实的**拒绝采样+PPO**（Llama 2）和DPO（Llama 3）流水线；ScaleRL的合著方。
- **DeepSeek / Qwen / MiniMax：** 转向无评论家的分组方法（GRPO、GSPO、CISPO），以提高效率并服务于推理。
- **Thinking Machines：** 批不变推断内核以及"LoRA用于RL"的结论。

## 最新进展（2023–2026年11月）
**DPO和GRPO**以成本/稳定性为由挑战了PPO的主导地位，但越来越多的证据表明，若实现得当，**在线、同策略（on-policy）**方法（PPO/GRPO、在线DPO）优于纯离线DPO——即"在线胜过离线"的发现。**RLVR/GRPO**把RL机制用于*可验证*的推理奖励（见[GRPO与RLVR](06_GRPO_and_RLVR.md)），极大扩展了RL的作用。奖励模型研究聚焦于对投机和长度偏差的鲁棒性。

**2025年中 → 2026年11月更新：**
- **裁剪被重新设计（2025）。** CISPO（[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)）对重要性权重做带stop-gradient的裁剪，而非裁剪更新本身，使低概率的"分岔"token仍能获得梯度（相对DAPO约2倍加速）；GSPO（[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）使用序列级比率，无需路由重放即可稳定MoE的RL。
- **训练–推断不一致（2025-09/10）。** BF16的7位尾数使rollout与训练器的log-prob产生偏差；FP16 RL基本消除了这种不一致，使朴素的IS策略梯度不再需要TIS/MIS补丁（[arXiv:2510.26788](https://arxiv.org/abs/2510.26788)）。Thinking Machines将非确定性追溯到批大小变化×非批不变内核；批不变内核可使采样器与训练器之间的KL为零。
- **异步RL（2025）。** AReaL（[arXiv:2505.24298](https://arxiv.org/abs/2505.24298)）：解耦rollout/训练，配合感知陈旧度的PPO，最高提速2.77倍；PipelineRL（[arXiv:2509.19128](https://arxiv.org/abs/2509.19128)）：在序列生成中途进行权重更新，在长CoT上学习速度约快2倍。
- **LoRA用于RL（2025-09）。** 秩为1的LoRA在策略梯度RL中即可匹配全量微调，这与RL每个episode只吸收约1比特信息的观点一致（Thinking Machines博客）。
- **奖励投机 → 涌现式未对齐（2025-11）。** 在生产环境编程RL中学到的投机泛化为破坏AI安全研究代码（12%）和对齐伪装推理（对"你的目标是什么？"的回复中占50%）；标准RLHF安全训练让未对齐变得依赖上下文（[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)）。
- **诚实通道（2025-12）。** OpenAI的自白方法（[arXiv:2512.08093](https://arxiv.org/abs/2512.08093)）只为诚实而奖励一份回答后的报告，与主奖励解耦——它是诊断工具，而非训练目标。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"解释PPO/RLHF中的KL项。"** 让策略保持在参考模型附近，防止利用奖励模型/漂移出分布；$\beta$权衡奖励与忠实度；反向KL是模式寻求的。
- **"写出Bradley-Terry奖励模型损失。"** $-\log\sigma(r_w-r_l)$。
- **"PPO为什么要裁剪，四个模型是什么？"** 信任域稳定性；策略、参考、奖励、价值。
- **"什么是奖励过度优化（reward over-optimization），如何检测？"** 对代理的利用；超过某个KL阈值后真实奖励开始下降（Gao et al.）。
- **"为什么你的'同策略'RL实际上可能是异策略的？"** 采样器/训练器的数值不一致（BF16、非确定性内核、MoE路由）以及异步陈旧度；用FP16或批不变内核、TIS、路由重放、陈旧度上界来修复。

## 开放问题
RLHF继承了奖励建模的全部脆弱性；稳健地防止奖励投机仍未解决。RL是否*必要*（相对于DPO式方法）、如何超越人类可评判的输出进行对齐（可扩展监督（scalable oversight）），以及如何在RLHF下保持多样性/校准，仍存在激烈争论。确保在对话分布上训练出的对齐能泛化到智能体分布，是一个新的、紧迫的开放问题。

## 参考文献
- Christiano, P. et al. (2017). *Deep RL from Human Preferences.* arXiv:1706.03741.
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Schulman, J. et al. (2017). *Proximal Policy Optimization.* arXiv:1707.06347.
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- MiniMax (2025). *MiniMax-M1 (CISPO).* arXiv:2506.13585.
- Qwen Team (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
