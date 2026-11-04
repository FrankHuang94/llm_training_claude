# GRPO与RLVR：推理强化学习（GRPO and RLVR (Reasoning RL)）

> **最后更新：** 2026-11-04
> **相关文件：** [RLHF](03_RLHF.md) · [推理模型](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [智能体式编程与计算机操作](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md) · [DeepSeek路线图](../08_lab_roadmaps/04_deepseek_roadmap.md)
> **关键论文：** Shao et al. 2024 DeepSeekMath/GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · Lambert et al. 2024 Tülu 3 / RLVR ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) · Yu et al. 2025 DAPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) · ScaleRL 2025 ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786))

## 概述
GRPO（Group Relative Policy Optimization，分组相对策略优化）和可验证奖励强化学习（RLVR，Reinforcement Learning with Verifiable Rewards）是2024–2026年推理革命的算法核心。其中心思想是：RL的对象不再是脆弱的学习型奖励模型（reward model），而是一个**客观验证器**——数学判分器、代码单元测试、符号检查器——它返回可靠、难以投机的奖励。这让模型能够通过试错*学会推理*：生成长思维链（chain of thought，CoT），因最终答案正确而获得奖励，并强化那些奏效的推理。DeepSeek-R1震惊了整个领域，它表明**基于可验证奖励的纯RL**在没有任何推理专用监督数据的情况下，就能诱导出涌现的长CoT、自我验证和"顿悟时刻（aha moments）"。在2025–26年间，领域收敛到一族GRPO改进方法，学会了从小规模实验预测RL的缩放行为，并把RLVR扩展到长时程智能体式（agentic）环境——这是2026年底主要的能力缩放维度。

## 核心概念
**RLVR（可验证奖励）。** 对于答案可检查的领域，奖励由*验证器*计算：若最终答案与标准答案一致（数学）或通过全部单元测试（代码），则`reward = 1`，否则为`0`（有时加上格式奖励）。由于验证器在构造上是正确的，RLHF中*学习型奖励模型*这一失效模式基本被消除，从而能进行稳定的大规模RL——尽管环境漏洞利用（例如用`sys.exit(0)`伪造测试通过）依然存在。

**GRPO（算法）。** PPO需要一个独立的**价值/评论家（critic）**网络来估计优势的基线——代价高昂（模型内存翻倍），且在长序列上本身就难以训练。GRPO通过**分组相对基线**去掉了评论家：对每个提示，采样一*组*$G$个输出$\{o_1,\dots,o_G\}$，用验证器为每个输出打分得到奖励$\{r_i\}$，并通过**组内归一化**计算优势：
$$\hat A_i=\frac{r_i-\text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}$$
组均值*就是*基线（对该提示期望奖励的经验无偏估计）。随后用这些优势、以PPO式的裁剪目标更新策略（policy），并加上对参考模型的KL惩罚：
$$\mathcal{L}_{GRPO}=\mathbb{E}\Big[\tfrac{1}{G}\sum_i \min\big(\rho_i \hat A_i,\ \text{clip}(\rho_i,1\!-\!\epsilon,1\!+\!\epsilon)\hat A_i\big)\Big]-\beta\,D_{KL}(\pi_\theta\|\pi_{ref})$$
这更便宜（没有价值网络），天然适合结果奖励（优势在一个样本的所有token间共享），而且稳定。

**2025年的改进。** *Dr. GRPO*去掉了逐回复的长度归一化和标准差除法（两者分别偏向冗长的错误答案和过易/过难的提示）。*DAPO*解耦裁剪（$\epsilon_{low}<\epsilon_{high}$，"Clip-Higher"）以对抗熵坍缩，丢弃全对/全错的组（动态采样，因为此时$\hat A_i\equiv 0$），使用token级损失聚合，并对超长输出做奖励塑形。*CISPO*裁剪的是重要性权重本身并停止其梯度，$\mathcal{L}=\mathbb{E}\big[\text{sg}(\text{clip}(\rho_t,1-\epsilon_{low},1+\epsilon_{high}))\,\hat A_t\log\pi_\theta(a_t)\big]$，从而让罕见的"分岔"token保留梯度。*GSPO*用长度归一化的序列比率$s_i=\big(\pi_\theta(o_i|x)/\pi_{\theta_{old}}(o_i|x)\big)^{1/|o_i|}$取代token级比率，并在序列层面裁剪，消除了使MoE的RL失稳的token级重要性采样噪声。

**R1-Zero与R1。** **R1-Zero**在*没有SFT*的情况下直接对基座模型应用GRPO+RLVR——长CoT、反思和自我纠错纯粹从RL中*涌现*出来，证明推理可以在没有示范的情况下学会（尽管输出可读性/语言混杂受到影响）。**R1**在RL之前加入了少量**冷启动长CoT SFT**（并在最后加入面向通用对齐的RLHF阶段），以修复可读性并拓宽能力——这是生产配方。

**为什么RLVR能带来长推理。** 可验证奖励只关心*正确性*，因此模型可以自由使用所需数量的"思考token（thinking tokens）"；RL会发现*更审慎、更长*的推理带来更高奖励，于是**测试时计算（test-time compute）会自发地扩展**（见[推断时缩放](../04_reasoning_and_agents/02_inference_time_scaling.md)）。

**可预测的RL缩放。** ScaleRL用一个饱和sigmoid拟合通过率与RL计算量$C$的关系，$R(C)=R_0+\frac{A-R_0}{1+(C_{mid}/C)^{B}}$，其中$A$是渐近上限，$B$是效率指数；基于早期计算量的拟合可以外推到大得多的训练规模。

## 关键挑战
- **可验证领域局限。** RLVR需要可检查的奖励——对数学/代码/逻辑很自然，对开放式写作、有用性或安全则很难。
- **奖励稀疏与信用分配。** 长轨迹只有单一0/1结果奖励，信号粗糙；把功劳分配到具体推理步骤上仍未解决（PRM有帮助，但可能被投机利用）。
- **长度/"过度思考"。** 模型可能钻长度的空子，产生重复或不必要冗长的推理链，或混杂多种语言。
- **探索与熵坍缩。** 若不加干预，熵会以可预测的方式被换成奖励，$R=-a\,e^{H}+b$，其驱动因素是log-prob与优势之间的协方差；熵一旦耗尽，奖励上限就被锁定。
- **训练–推断不一致。** rollout引擎与训练器的数值差异（BF16、非批不变内核、MoE路由）让名义上同策略（on-policy）的RL变成异策略（off-policy）。
- **计算成本。** 每个提示生成大量长rollout代价高昂（推断密集型RL）；智能体rollout还增加了环境成本和长尾拖慢。
- **环境投机。** 基于测试和沙箱的奖励可能被利用，而投机行为可能泛化为更广泛的未对齐。

## 解决方案与当前最佳实践
**GRPO族目标函数+RLVR**，在数学/代码/逻辑/智能体任务上使用**结果验证器**，可选加入**冷启动长CoT SFT**（R1式）以及**格式/长度奖励**。收敛后的2026年配方（ScaleRL及同类工作）：token级或提示级损失平均、非对称裁剪或重要性权重裁剪（DAPO/CISPO）或针对MoE的序列级比率（GSPO）、零方差过滤、FP32 LM-head logits，以及PipelineRL式的异步训练。控制熵（Clip-Higher、Clip-Cov/KL-Cov、RL↔SFT交替）。对齐采样器与训练器的数值（FP16或批不变内核、TIS、路由重放）。LoRA足以用于RL。用**拒绝采样（rejection sampling）**收集正确轨迹来做SFT/蒸馏，并用**同策略蒸馏（on-policy distillation）**迁移到更小的模型。把可验证RL与评分细则/偏好RL结合以处理不可验证的行为；加固环境并应用**接种提示（inoculation prompting）**。

## 各实验室观点
- **DeepSeek：** 提出了GRPO（DeepSeekMath）和R1纯RL推理范式；在R1中刻意使用**基于规则的验证器而非神经PRM**，之后在DeepSeekMath-V2中训练了LLM证明验证器作为奖励；V3.2-Speciale声称在IMO/ICPC/IOI 2025达到金牌水平。
- **OpenAI：** o系列以及GPT-5/6思考模型都是基于推理的RL（方法未公开）；一个实验性推理模型宣布在IMO 2025获得35/42，一个OpenAI系统在ICPC 2025解出12/12（非官方）。
- **AllenAI（Tülu 3）：** 开放的RLVR配方，把"可验证奖励强化学习"正式确立为一个有名字的方法。
- **Google DeepMind：** Gemini Deep Think以自然语言、在时限内、无需Lean形式化，获得了官方评分的IMO 2025金牌（35/42）。
- **Qwen / MiniMax / 字节跳动Seed：** GSPO（Qwen3）、CISPO（MiniMax-M1）、DAPO——GRPO主要的开源后继者。
- **Meta等：** ScaleRL，迄今最大规模的公开RL计算缩放研究。

## 最新进展（2023–2026年11月）
2025年GRPO变体爆发（DAPO、Dr. GRPO、GSPO、VAPO），针对稳定性、长度控制和探索。围绕RL是**激发潜在**推理还是**教会新的**推理展开争论（"RL只是锐化了基座模型"的批评vs"RL发现了新行为"）。RLVR被扩展到数学/代码之外的**智能体/工具使用**任务（以执行结果为奖励），并通过LLM评审扩展到**半可验证**领域。推理模型的后训练计算量如今已可比肩预训练。

**2025年中 → 2026年11月更新：**
- **目标函数趋同（2025-03 → 07）。** DAPO从Qwen2.5-32B基座在AIME24上达到50分（[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)）；CISPO报告相对DAPO约2倍加速（[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)）；GSPO无需路由重放即可修复MoE的RL崩溃（[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）。
- **ScaleRL（2025-10）。** 超过40万GPU小时的消融实验；sigmoid计算–性能拟合可外推到10万GPU小时规模；大多数技巧移动的是效率$B$而非上限$A$（[arXiv:2510.13786](https://arxiv.org/abs/2510.13786)）。后续工作发现更大的模型在RL中计算和数据效率更高，并且RL所需计算量随推理深度呈幂律增长（[arXiv:2605.06638](https://arxiv.org/abs/2605.06638)）。
- **熵机制（2025-05）。** 熵的变化由Cov(log-prob, 优势)驱动；Clip-Cov/KL-Cov约束高协方差token（[arXiv:2505.22617](https://arxiv.org/abs/2505.22617)）；ComputerRL的Entropulse在长时程智能体式训练中交替RL与SFT（[arXiv:2508.14040](https://arxiv.org/abs/2508.14040)）。
- **数值与基础设施（2025）。** 用FP16替代BF16基本消除了rollout/训练器不一致（[arXiv:2510.26788](https://arxiv.org/abs/2510.26788)）；AReaL（最高2.77倍，[arXiv:2505.24298](https://arxiv.org/abs/2505.24298)）、PipelineRL（[arXiv:2509.19128](https://arxiv.org/abs/2509.19128)）和slime（GLM-5智能体式RL，[arXiv:2602.15763](https://arxiv.org/abs/2602.15763)）让RL异步化；verl仍是最常用的开源框架。
- **竞赛成绩（2025-07/09）。** IMO 2025端到端自然语言金牌（Gemini Deep Think，官方评分；OpenAI，由往届奖牌得主评分）；ICPC 2025：OpenAI 12/12，Gemini 2.5 Deep Think 10/12（新闻报道）。据报道IMO 2026有多个系统获得42/42（仅新闻来源）——奥赛数学作为基准实际上已经饱和。
- **超越精确匹配的RLVR。** Rubrics as Rewards（[arXiv:2507.17746](https://arxiv.org/abs/2507.17746)）以及自我验证的证明RL（DeepSeekMath-V2，[arXiv:2511.22570](https://arxiv.org/abs/2511.22570)）。
- **中期训练vs RL数据（2026-03）。** PRISM：RL对约5%的参数做稀疏更新；决定RL后上限的主要是中期训练（mid-training）数据，而不是RL数据配比（[arXiv:2603.17074](https://arxiv.org/abs/2603.17074)）——这为"激发vs创造"之争中"RL只是锐化"一方提供了证据。
- **智能体式RLVR（2025-08 → 2026）。** 在真实代码仓库和虚拟桌面上的多轮RL，以及并行智能体RL（Kimi K2.5 PARL，[arXiv:2602.02276](https://arxiv.org/abs/2602.02276)）；环境规模化如今是瓶颈（见[智能体式编程与计算机操作](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"GRPO与PPO有何不同？"** 没有评论家；用分组相对归一化优势作为基线；更便宜，对结果奖励稳定。
- **"写出GRPO的优势。"** 在采样组上$\hat A_i=(r_i-\text{mean})/\text{std}$。
- **"可验证奖励为什么意义重大？"** 低噪声、可靠的信号 → 稳定的大规模RL → 涌现的长CoT（但环境依然可能被投机利用）。
- **"R1-Zero证明了什么？"** 推理（长CoT、自我纠错）能从纯RL中涌现，无需SFT。
- **"RLVR的局限？"** 需要可检查的奖励；信用分配稀疏；难以用于开放式任务。
- **"朴素GRPO有哪些偏差，DAPO/Dr. GRPO/GSPO如何修正？"** 长度归一化偏向冗长的错误答案，标准差除法按难度重新加权（Dr. GRPO）；对称裁剪导致熵坍缩（Clip-Higher）；零优势组浪费计算（动态采样）；token级重要性采样噪声使MoE失稳（GSPO的序列比率）。
- **"你如何决定是否扩大一次RL训练？"** 在早期计算量上拟合ScaleRL的sigmoid；在投入之前比较不同配方的渐近值$A$与效率$B$。

## 开放问题
RL究竟是*创造*新能力，还是只*放大*基座模型中的潜在能力，仍有激烈争论。长推理的信用分配（不被投机的稳健过程奖励）、把可验证RL扩展到开放式/智能体领域、控制过度思考，以及RL、蒸馏与推断时搜索之间真正的计算最优（compute-optimal）平衡，都是开放问题。扩展并加固RL*环境*，以及防止RL侵蚀CoT可监控性，是新的前沿问题。

## 参考文献
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Lambert, N. et al. (2024). *Tülu 3 (RLVR).* arXiv:2411.15124.
- Yu, Q. et al. (2025). *DAPO.* arXiv:2503.14476.
- Liu, Z. et al. (2025). *Understanding R1-Zero-Like Training (Dr. GRPO).* arXiv:2503.20783.
- Qwen Team (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- MiniMax (2025). *MiniMax-M1 (CISPO).* arXiv:2506.13585.
- Meta et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
- Cui, G. et al. (2025). *The Entropy Mechanism of RL for Reasoning LMs.* arXiv:2505.22617.
