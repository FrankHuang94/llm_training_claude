# 面向大语言模型的信息论基础（Information Theory Basics for LLMs）

> **最后更新：** 2026-11-04
> **相关文件：** [缩放定律](04_scaling_laws.md) · [预训练目标](../02_pretraining/00_pretraining_objectives.md) · [DPO 与偏好优化](../03_posttraining/05_DPO_and_preference_optimization.md)
> **关键论文：** Shannon 1948 "A Mathematical Theory of Communication" · Cover & Thomas, *Elements of Information Theory* · Schulman 2015 TRPO（KL 信赖域，[arXiv:1502.05477](https://arxiv.org/abs/1502.05477)）· Rafailov et al. 2023 DPO（[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)）

## 概述
信息论是语言建模的数学基石。用最大似然训练语言模型，*恰好*等价于最小化数据分布与模型之间的交叉熵——也等价于为数据构建尽可能好的无损压缩器。许多面试常考的量（困惑度、每字节比特数、基于人类反馈的强化学习（RLHF）中的 KL 惩罚、PPO 中的熵奖励、标签平滑）都是少数几个信息论基本概念的直接应用。一个常被引用的说法——"**压缩即智能**"（以及强大的大语言模型（LLM）本身就是最先进压缩器这一实际观察）——完全建立在这一等价关系之上。

## 核心概念
**熵。** 分布 $p$ 的平均信息量：
$$H(p)=-\sum_x p(x)\log p(x)$$
它是对来自 $p$ 的样本进行编码时理论上最小的期望码长（若取 $\log_2$，单位为比特）。对语言而言，$H$ 就是缩放定律（scaling law）中不可约的"$E$"下限。

**交叉熵与语言模型目标。** 训练最小化
$$H(p,q)=-\sum_x p(x)\log q(x) = H(p)+D_{KL}(p\,\|\,q)$$
其中 $p$ 是数据，$q$ 是模型。由于 $H(p)$ 固定，最小化交叉熵就是最小化 $D_{KL}(p\|q)$——把模型推向数据分布。逐 token 的下一个 token 损失正是这个交叉熵。

**KL 散度。** $D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}\ge0$，当且仅当 $p=q$ 时为零。它是非对称的，不是度量。KL 在后训练（post-training）中无处不在：RLHF 的奖励包含 $-\beta\,D_{KL}(\pi_\theta\|\pi_{ref})$，用于让策略（policy）保持在参考模型附近（见 [RLHF](../03_posttraining/03_RLHF.md)）；DPO 的最优解是一个带 KL 正则的奖励最大化器；PPO/TRPO 使用 KL 信赖域。**前向与反向 KL** 的区别很重要：反向 KL（$D_{KL}(q\|p)$）是寻模的（mode-seeking，对应 RL/变分情形），前向 KL 是覆盖模式的（mode-covering，对应 MLE 情形）——这解释了为什么 RLHF 可能降低多样性。

**困惑度。** 标准的语言模型指标，即逐 token 交叉熵的指数：
$$\text{PPL}=\exp\!\Big(-\frac1N\sum_{i=1}^N\log q(x_i\mid x_{<i})\Big)$$
可以理解为模型的有效分支因子（等可能的下一个 token 的平均数量）。越低越好。它**依赖于分词器（tokenizer）**，因此跨模型比较困惑度需要相同的分词。

**每字节比特数（BPB）/ 每字符比特数。** 一种与分词器无关的替代指标：用*字节*数而不是 token 数对总负对数似然做归一化，从而可以在不同分词器之间公平比较，并直接与压缩率挂钩。

**互信息与 MDL。** 互信息 $I(X;Y)=H(X)-H(X|Y)$ 是对比学习目标（CLIP/InfoNCE）和表示学习的基础。**最小描述长度**原则把学习视为压缩：最好的模型使（模型描述长度 + 给定模型后的数据描述长度）最小，这为理解正则化以及"顿悟（grokking）"/泛化之争提供了一个视角。

## 关键挑战
- **困惑度对分词器的依赖。** 使简单比较产生误导；需要使用 BPB 或共享的分词器。
- **RL 中的 KL 估计。** KL 惩罚必须从样本中估计；有偏/高方差的估计量（$k_1,k_2,k_3$ 估计量）会影响稳定性。
- **非对称性的陷阱。** 选择前向还是反向 KL，决定了是覆盖模式还是坍缩到某个模式——这是常见的混淆来源。
- **损失 ≠ 能力。** 交叉熵的改进并不会线性地映射为下游技能（见[缩放定律](04_scaling_laws.md)）。

## 解决方案与当前最佳实践
在做跨模型论断时，与困惑度一起报告 **BPB**（当实验室在版本之间更换分词器时这一点必不可少，例如 Anthropic 在 Claude Opus 4.7 上就这样做了）。在 RLHF/GRPO 中，使用低方差的 KL 估计量（John Schulman 的 $k_3$ 估计量 $\frac{r-1-\log r}{}$，无偏且非负），并调节 $\beta$ 以平衡奖励与 KL。**标签平滑**可理解为加入一个与均匀分布之间的交叉熵项（惩罚过度自信，起正则作用）。RL 中的熵奖励通过最大化策略熵来鼓励探索；2025–26 年的推理（reasoning）RL 实践更偏好有针对性的熵控制（clip-higher、基于协方差的裁剪/惩罚），而不是全局的熵奖励。在蒸馏（distillation）方面，**同策略（on-policy）反向 KL**（学生采样，教师为每个 token 打分）如今已是标准的后训练阶段。

## 各实验室观点
各实验室在理论上大体一致，但表述角度不同。**OpenAI/DeepMind** 倾向于"压缩即智能"的观点（DeepMind 的 "Language Modeling Is Compression"（2023）表明 LLM 胜过 gzip/PNG）。**Anthropic** 强调在 RLHF 和 CAI 中把"与参考模型的 KL"作为安全/保守性的调节杠杆。**DeepSeek** 公开了 GRPO 中简洁的 KL 处理方式。**Thinking Machines** 把同策略蒸馏表述为逐 token 的反向 KL 奖励，并论证 RL 每个回合只传递约 1 比特信息。**OpenAI**（Kalai et al. 2025）把幻觉（hallucination）重新阐释为预训练加上二元评分评测共同导致的统计后果。各家内部都报告 BPB 一类的归一化指标。

## 最新进展（2023–2026年11月）
"Language Modeling Is Compression"（Delétang et al. 2023）在实证上把语言模型质量与任意模态的无损压缩联系起来。关于**更低损失 ⇒ 更通用的智能**的争论仍在继续。对上下文学习的信息论分析、熵在推理模型探索（GRPO）中的作用，以及对泛化/顿悟的 MDL 解释，都是活跃的研究方向。

**2025年中 → 2026年11月更新：**
- **RL 的熵定律（2025-05；arXiv:2505.22617）。** 不加干预时，推理 RL 会以可预测的方式用熵换取奖励，$R=-a\,e^{H}+b$，因此一旦熵坍缩，可达到的奖励就被封顶。每一步的熵变化由 token 对数概率与其优势之间的协方差驱动；Clip-Cov（对高协方差 token 断开梯度）和 KL-Cov（对其施加惩罚）能让熵保持活力。DAPO 的解耦 clip-higher（arXiv:2503.14476）是更简单的近亲方法。
- **以反向 KL 作为稠密奖励（2025-10，Thinking Machines；二手来源）。** 同策略蒸馏从学生模型采样，并以逐 token 的 $-\log\frac{\pi_\theta(x_t)}{\pi_T(x_t)}$ 作为奖励——即在学生自身分布上的寻模 $D_{KL}(\pi_\theta\|\pi_T)$——据报道只用约 RL 十分之一的 GPU 小时就在 AIME24 上达到约 70%。其寻模带来的失效模式催生了熵门控的前向/反向切换（arXiv:2603.07079）以及截断 top-K 反向 KL（arXiv:2603.25562）。
- **每回合的比特数（2025-09，"LoRA Without Regret"；二手来源）。** 以标量奖励进行的策略梯度 RL 每个回合只传递约 1 比特的信息量，这就是为什么即使秩为 1 的 LoRA 在 RL 中也能追平全量微调，而监督微调（SFT）需要大得多的容量——这是一个关于适配器（adapter）规模的信息容量论证。
- **数值噪声下的似然比（2025-07 → 2025-10）。** GSPO（arXiv:2507.18071）用长度归一化的序列级比率取代 token 级重要性比率，以降低 MoE RL 中的方差。BF16 只有 7 位尾数，会使 rollout 引擎与训练器的对数概率出现偏差，悄无声息地把"同策略"RL 变成异策略（off-policy）；把 RL 切换到 FP16 基本消除了这种失配（arXiv:2510.26788），而批不变内核能让采样器与训练器之间的 KL 精确为零（Thinking Machines；二手来源）。
- **作为统计下界的幻觉（2025-09；arXiv:2509.04664）。** 生成错误的下界由二元"是否有效"分类问题的错误率给出；恰好只出现过一次的事实（Good–Turing 单例率）必然以相当的比率产生幻觉。二元评分的基准（benchmark）进一步奖励猜测而非弃答；所提出的修复是采用带置信度阈值的评分，并为"我不知道"给分。
- **再次证明损失 ≠ 能力（2026-02；arXiv:2602.15014）。** 在扩散语言模型中，均匀状态扩散的似然比掩码扩散差，却在 GSM8K 上表现更好——这清楚地说明，跨模型家族的似然界并不能为下游技能排序。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么最小化交叉熵等同于最小化与数据之间的 KL？"** 分解 $H(p,q)=H(p)+D_{KL}(p\|q)$。
- **"定义困惑度并说明其注意事项。"** 平均 NLL 的指数；分支因子的直觉；依赖分词器 → 使用 BPB。
- **"前向与反向 KL——RLHF 用哪个，为什么重要？"** 反向 KL，寻模，多样性坍缩。
- **"PPO/GRPO 中的 KL 惩罚如何估计？"** 基于采样的 $k_3$ 估计量；偏差/方差的权衡。
- **"为什么同策略蒸馏使用反向 KL，会出什么问题？"** 从学生采样使期望可以计算，并解决暴露偏差；反向 KL 是寻模的，学生可能坍缩到教师的某一个模式上，而且在分布外的前缀上教师信号会变弱。
- **"为什么单例率构成幻觉的下界？"** Good–Turing：只出现过一次的事实所占比例，估计了那些模型无法可靠地与看似合理的编造区分开的、未见事实的概率质量。

## 开放问题
"压缩即智能"的等价关系能否延伸到*推理*与*能动性*（而不仅仅是预测）仍有争议。交叉熵下降与涌现（emergence）能力之间的精确关系，以及样本效率和持续学习（continual learning）的信息论极限，仍是开放问题。

## 参考文献
- Shannon, C. (1948). *A Mathematical Theory of Communication.* Bell System Tech. J.
- Cover, T., Thomas, J. *Elements of Information Theory* (2nd ed.).
- Delétang, G. et al. (2023). *Language Modeling Is Compression.* arXiv:2309.10668.
- Schulman, J. (2020). *Approximating KL Divergence*（博客，$k_1/k_2/k_3$ 估计量）.
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Cui, G. et al. (2025). *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* arXiv:2505.22617.
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- Kalai, A. T. et al. (2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
