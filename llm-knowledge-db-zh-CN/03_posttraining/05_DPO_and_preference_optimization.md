# DPO与偏好优化（DPO and Preference Optimization）

> **最后更新：** 2026-11-04
> **相关文件：** [RLHF](03_RLHF.md) · [奖励建模](04_reward_modeling.md) · [GRPO与RLVR](06_GRPO_and_RLVR.md)
> **关键论文：** Rafailov et al. 2023 DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) · Azar et al. 2023 IPO ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036)) · Ethayarajh et al. 2024 KTO ([arXiv:2402.01306](https://arxiv.org/abs/2402.01306)) · Meng et al. 2024 SimPO ([arXiv:2405.14734](https://arxiv.org/abs/2405.14734)) · Hong et al. 2024 ORPO ([arXiv:2403.07691](https://arxiv.org/abs/2403.07691))

## 概述
直接偏好优化（Direct Preference Optimization，DPO）是介于RLHF与推理时代之间最重要的后训练进展。它的洞见是：整个RLHF目标——奖励建模*加上*受KL约束的RL——可以压缩为**一个在偏好对上的、类似监督学习的损失**，从而省去显式的奖励模型（reward model）、rollout（采样轨迹）采样和PPO机制。DPO让偏好对齐变得便宜、稳定、可复现，引发了开放对齐模型的井喷（Zephyr、Tülu以及大量Llama微调模型）。理解DPO的推导——以及为什么在其假设下它与RLHF在数学上等价——是被问得最多的高阶后训练面试题之一。然而到2026年底，前沿能力的提升压倒性地来自*在线*RL（RLVR、智能体式RL）和同策略蒸馏（on-policy distillation）；DPO的角色是一个廉价、稳定的偏好阶段和一个强基线。

## 核心概念
**关键推导。** 受KL约束的RLHF目标有一个已知的**闭式最优策略**：
$$\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{ref}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big)$$
将其反解，可用最优策略表示奖励：
$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}+\beta\log Z(x)$$
把这个奖励代入Bradley-Terry偏好模型，难以计算的配分函数$Z(x)$会**相互抵消**（给定$x$时它对$y_w$和$y_l$相同），从而得到一个直接作用于策略的损失：
$$\mathcal{L}_{DPO}=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$
策略*本身*就是它的隐式奖励模型：$\hat r(x,y)=\beta\log\frac{\pi_\theta}{\pi_{ref}}$。DPO训练策略赋予$y_w$比$y_l$更高的隐式奖励，$\beta$控制对参考模型的KL约束强度。

**为什么有效。** DPO优化的是与RLHF*相同*的目标，只是离线、以闭式方式进行——没有奖励模型、没有采样、没有价值网络，稳定得多。梯度会提高偏好回复的似然、降低非偏好回复的似然，并按当前隐式奖励错得有多严重来加权。

**主要变体。**
- **IPO**（Azar et al.）：加入正则项，修正DPO对确定性偏好**过拟合**的倾向（当$y_w$总是被偏好时，DPO可能导致$\pi_\theta(y_l)\to 0$并引发退化）；用围绕目标间隔的平方损失替代逻辑损失。
- **KTO**（Ethayarajh et al.）：使用**非成对**的二元"好/坏"标签（Kahneman-Tversky前景理论效用），无需成对数据——在只有点赞/点踩时很实用。
- **SimPO**（Meng et al.）：**无参考模型**，用长度归一化的平均log-prob作为隐式奖励并加上目标间隔——更简单，内存中无需$\pi_{ref}$，效果强。
- **ORPO**（Hong et al.）：通过优势比（odds-ratio）惩罚把**SFT与偏好合并**为一个阶段，省去单独的SFT步骤。
- 其他：CPO、R-DPO（长度正则化）、在线/迭代DPO。

**DPO在2026年各类目标函数中的位置。** 离线DPO（固定偏好对，序列级对比）→ 迭代/在线DPO（由当前策略生成新的偏好对）→ 同策略蒸馏（学生采样，*逐token*教师信号）→ 分组相对RL（GRPO族，标量验证器奖励）。越往右，同策略程度和信号密度越高，但成本也越高。

## 关键挑战
- **离线/异策略（off-policy）局限。** DPO在*固定的*偏好数据集上训练；它从不从当前策略采样，因此无法纠正数据分布之外的错误——这是"DPO vs PPO"之争的核心弱点。
- **似然位移/退化。** DPO可能*降低*偏好回复的绝对概率（它只强制*间隔*），并可能过度压制，损害校准/多样性。
- **长度偏差与无奖励模型带来的偏差。** 若无显式控制，DPO会继承偏好数据中的长度和虚假偏差。
- **对$\beta$和参考模型敏感。** 性能对$\beta$以及$\pi_{ref}$的选择/质量很敏感。
- **偏好数据中的特质泄漏。** 偏好对可能悄悄地教会模型不想要的特质（例如谄媚（sycophancy））；没有模型内部信号就很难筛查数据。

## 解决方案与当前最佳实践
**迭代/在线DPO**——定期从当前策略生成新回复，进行标注（用奖励模型或AI评审），再继续DPO——找回了PPO同策略优势的大部分，如今是领先的配方之一（Llama 3、Tülu 3以及许多2024–2025年的模型）。使用**长度归一化/正则化**（SimPO、R-DPO）对抗长度偏差，用**IPO**防止过拟合，在只有非成对反馈时用**KTO**。调节$\beta$（约0.01–0.1）。**先做好SFT**很重要；ORPO则把它合并进来。对推理和智能体式任务，GRPO/RLVR（在线RL）通常优于离线DPO；存在强教师时，同策略蒸馏往往是更便宜的同策略选项。在条件允许时，用基于激活的特质方向（人格向量，persona vectors）筛查偏好数据。

## 各实验室观点
- **Stanford**（Rafailov et al.）提出了DPO；**HuggingFace**（Zephyr）和**AllenAI**（Tülu、OLMo）推广了开放的DPO配方。
- **Meta：** Llama 3在大部分对齐中使用**迭代DPO**（加上拒绝采样（rejection sampling））而非PPO——这是一个值得注意的生产级背书；其2026年的Muse模型为闭源，后训练细节未公开。
- **OpenAI/Anthropic：** 在前沿保留同策略RL（PPO/GRPO类），认为在线采样很重要；DPO在开放/社区模型中更常见。Anthropic的2026年宪法被用于生成由Claude给出的偏好排序。
- **DeepSeek：** 推理使用GRPO（在线RL）；部分偏好对齐使用DPO式方法。
- **Thinking Machines：** 把同策略蒸馏定位为介于SFT/DPO与RL之间的稠密奖励中间地带。

## 最新进展（2023–2026年11月）
**"DPO vs PPO"之争**基本收敛为"**在线/同策略胜过离线**，但离线DPO是一个出色的廉价基线"。从经验上看，调优良好的在线RL（PPO/GRPO）往往在难推理任务和前沿上胜出，而迭代/在线DPO以更低成本缩小了大部分差距。无参考模型（SimPO）和非成对（KTO）变体因实用性而普及。理论工作厘清了DPO的退化/似然位移失效模式。

**2025年中 → 2026年11月更新：**
- **同策略蒸馏成为新的中间地带（2025-10）。** 学生自采样轨迹、配合逐token反向KL教师奖励，以约RL十分之一的GPU小时达到教师水平（Thinking Machines博客）——与迭代DPO争夺"便宜但同策略"这一定位；2026年的工作加入了熵门控的前向/反向KL（EOPD，[arXiv:2603.07079](https://arxiv.org/abs/2603.07079)）。
- **在线RL的优势被量化为效率而非上限（2025-10）。** ScaleRL（[arXiv:2510.13786](https://arxiv.org/abs/2510.13786)）表明，大多数RL设计选择改变的是计算效率而非渐近值——这为"离线vs在线"提供了一个有用视角：更便宜的方法也许能更慢地达到同一上限；若数据偏离分布，则可能根本达不到。
- **偏好用于训练评审，而不只是策略（2026-02）。** Rubric-ARM（[arXiv:2602.01511](https://arxiv.org/abs/2602.01511)）用基于偏好反馈的交替RL训练评分细则生成器和评审，再由它们为策略提供奖励——偏好数据越来越多地经由一个学习型评分器流动。
- **用人格向量筛查数据（2025-08）。** 谄媚、幻觉等特质的激活方向可以标记出会诱导该特质的偏好/微调数据，而微调期间的"预防性引导"就像疫苗一样起作用（[arXiv:2507.21509](https://arxiv.org/abs/2507.21509)）。
- **由宪法生成的偏好（2026-01）。** Anthropic重写的宪法被用于多个训练阶段，包括由Claude生成偏好排序——AI标注的偏好对如今是前沿偏好优化的默认输入。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"从RLHF目标推导DPO。"** 闭式最优策略 → 反解出奖励 → 代入Bradley-Terry → $Z(x)$抵消 → 只含策略的损失。
- **"DPO与RLHF/PPO——如何权衡？"** DPO：更简单、稳定、离线，无需奖励模型/rollout；PPO：同策略，能纠正分布外错误，在难任务上更好，但成本高/不稳定。
- **"DPO中的$\beta$控制什么？"** 对参考模型的KL约束强度；隐式奖励的尺度。
- **"DPO为什么可能降低偏好回复的概率？"** 它强制的是*间隔*而非绝对似然 → 似然位移；这促成了IPO/正则化。
- **"何时用KTO或SimPO？"** 只有非成对好/坏标签时用KTO；需要无参考模型+长度归一化时用SimPO。
- **"DPO与同策略蒸馏——你会选哪个？"** 存在更强教师时选OPD（在学生自己的样本上提供稠密的逐token信号）；只有偏好对并且需要一个廉价稳定的阶段时选DPO。

## 开放问题
离线偏好优化能否在困难、分布外比重高的任务上追平在线RL，仍无定论。DPO的退化/校准失效模式尚未被完全驯服。如何把偏好优化与可验证奖励RL结合起来，并以迭代方式进行而不失稳定，是一个活跃的研究前沿。如何防止AI生成的偏好数据放大标注模型自身的偏差，如今变得更加紧迫。

## 参考文献
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Azar, M. G. et al. (2023). *A General Theoretical Paradigm (IPO).* arXiv:2310.12036.
- Ethayarajh, K. et al. (2024). *KTO.* arXiv:2402.01306.
- Meng, Y. et al. (2024). *SimPO.* arXiv:2405.14734.
- Hong, J. et al. (2024). *ORPO.* arXiv:2403.07691.
- Chen, R. et al. (2025). *Persona Vectors.* arXiv:2507.21509.
