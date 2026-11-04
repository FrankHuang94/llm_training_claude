# 缩放定律（Scaling Laws）

> **最后更新：** 2026-11-04
> **相关文件：** [计算与内存效率](../02_pretraining/05_compute_and_memory_efficiency.md) · [数据配比与课程学习](../01_data/03_data_mixing_and_curriculum.md) · [模型架构变体](05_model_architecture_variants.md)
> **关键论文：** Kaplan et al. 2020（[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)）· Hoffmann et al. 2022 Chinchilla（[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)）· Schaeffer et al. 2023 "Emergent Abilities a Mirage?"（[arXiv:2304.15004](https://arxiv.org/abs/2304.15004)）· Wei et al. 2022 Emergent Abilities（[arXiv:2206.07682](https://arxiv.org/abs/2206.07682)）

## 概述
缩放定律（scaling law）是把模型损失与计算量 $C$、参数量 $N$ 和数据量 $D$ 联系起来的经验规律。它的发现——测试损失在许多个数量级上平滑地按**幂律**下降——把大语言模型（LLM）的开发从手艺变成了可预测的工程学科：你可以用小规模实验预测大模型的损失，并以最优方式分配固定的计算预算。缩放定律是为单次训练投入数亿美元的理论依据。

两个里程碑式的结果奠定了这一领域。**Kaplan et al.（2020，OpenAI）**确立了幂律形式，并认为模型规模应比数据增长得更快。**Hoffmann et al.（2022，DeepMind "Chinchilla"）**纠正了这一点，表明计算最优（compute-optimal）的训练需要让 $N$ 和 $D$ 大致*等比例*增长，并指出此前的模型（GPT-3、Gopher）在数据上严重训练不足。

## 核心概念
**幂律形式。** 损失可以分解为一个可约项加上一个不可约的熵下限 $E$：
$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
在 Chinchilla 的拟合中，经验上 $\alpha\approx0.34$，$\beta\approx0.28$。每一项都是幂律；不可约的 $E$ 反映了语言本身的熵。

**计算最优分配。** 训练计算量为 $C\approx 6ND$（前向+反向，每参数每 token 6 FLOPs）。在固定 $C$ 的约束下最小化 $L$，得到最优的 $N^\star\propto C^{a}$、$D^\star\propto C^{b}$，其中 $a\approx b\approx0.5$。著名的经验法则：计算最优点上**每个参数约 20 个 token**。Chinchilla（70B 参数、1.4T token）在*相同*计算量下胜过 Gopher（280B、300B token）——一个小 4 倍但训练更充分的模型。

**Kaplan 与 Chinchilla 的分歧。** Kaplan 的分析出现偏差，因为它 (a) 没有针对每个训练长度调整学习率（learning rate）调度，(b) 对嵌入参数的排除不一致，从而偏向"模型更大、数据更少"。Chinchilla 修正后的方法得出了等比例缩放的结论，并被业界采纳。

**过度训练（LLaMA 范式）。** 计算最优最小化的是*训练*成本，但对于一个要被*调用*数十亿次的模型，**推断（inference）成本占主导**。因此各实验室有意把较小的模型"过度训练"到远超每参数 20 token（Llama 3 8B 训练了约 15T token——约每参数 1800 token），以得到一个部署便宜、同尺寸下又很强的小模型。当你优化的是总成本（训练 + 推断）时，最优点会移动。

## 关键挑战
- **方法敏感性。** 结果严重依赖学习率调度、参数计数方式和拟合流程——小错误就会翻转结论（Kaplan 与 Chinchilla）。
- **数据枯竭。** 按 Chinchilla 比例，前沿模型正在耗尽高质量公开文本（"数据墙"），被迫转向合成数据（synthetic data）和多轮重复。
- **下游能力与损失。** 缩放定律预测的是*损失*，而不是*能力*；从损失到基准（benchmark）成绩的映射充满噪声。
- **定律的迁移。** 不同架构（MoE、SSM）、数据质量和模态下的定律各不相同；每种情形都需要重新拟合。

## 解决方案与当前最佳实践
各实验室在自己的数据/架构上拟合**自己的**缩放定律，执行 IsoFLOP 扫描（在固定 $C$ 下改变 $N$，找到损失最小的模型），然后外推。面向部署时，他们优化**考虑推断的**缩放（过度训练）。混合专家（MoE）有其自己的定律，关联*激活*参数与*总*参数；对 MoE 应使用 $C\approx6N_{\text{active}}D$（例如 Kimi K2：尽管总参数为 1T，计算量仍约为 $6\times32\text{B}\times15.5\text{T}\approx3\times10^{24}$ FLOPs）。2025–26 年的开放 MoE 大约处在每个*激活*参数 500–700 个 token 的水平（K2 为 15.5T/32B；GLM-5 为 28.5T/40B）——按 Chinchilla 标准属于严重过度训练。数据受限缩放定律（Muennighoff et al. 2023）量化了重复数据的价值：重复约 4 轮以内几乎与新数据一样好，之后收益递减。

## 各实验室观点
- **OpenAI** 开创了缩放定律（Kaplan）以及 o 系列的"推断缩放定律"——损失/准确率随测试时计算量提升（见[推理模型](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)）。它把 GPT-6 Astra（2026-09）的提升归因于大规模预训练（pretraining）加上长时程强化学习（RL），即预训练规模与 RL 规模一起重回舞台。
- **DeepMind** 提出了 Chinchilla，并持续以 IsoFLOP 驱动设计（Gemini、Gemma）。
- **Meta** 曾以 Llama 系列成为过度训练范式的代表；其闭源的 Muse Spark（2026-04）主打计算效率（"用少得多的计算"达到 Llama-4-Maverick 级别的推理，厂商说法）。
- **Moonshot** 报告了优化器（optimizer）层面的效率提升（Moonlight 中 Muon 相对 AdamW 约 2 倍计算效率；Kimi K3 声称相对 K2 约 2.5 倍缩放效率）——但审慎的第三方基准会让这些说法打折扣（见下文）。
- **DeepSeek** 发表了 MoE 缩放定律以及超参数缩放定律（批大小、学习率与计算量的关系）。

## 最新进展（2023–2026年11月）
三大转变占据主导。**(1) 推断时缩放**：o1/o3/R1 表明准确率会随测试时的推理 token 数增长，这是训练计算之外的新维度。**(2) 数据受限缩放**：面对公开文本之墙，工作聚焦于重复、合成数据和按质量调整的"有效数据"。**(3) 蒸馏缩放定律**（2024–2025）刻画了何时用大教师模型的输出训练小模型优于从头训练。精度缩放定律（以 fp8/int4 训练）也在兴起。

**2025年中 → 2026年11月更新：**
- **RL 计算存在可预测的（S 形）规律（2025-10；arXiv:2510.13786，ICLR 2026 口头报告）。** ScaleRL（超过 40 万 GPU 小时）用一条饱和的 sigmoid 曲线拟合性能与 RL 计算量的关系——渐近线 $A$ 和效率指数 $B$——早期实验的拟合可以外推到 10 万 GPU 小时级别的运行。大多数设计选择改变的是效率 $B$ 而非上限 $A$。与预训练的幂律不同，RL 曲线会饱和，因此应比较的量是渐近线。
- **优化器"加速"随规模缩小（2025-09；arXiv:2509.02046）。** 在对每个优化器单独调参的前提下对 11 个优化器做基准测试，相对调好的 AdamW 没有任何加速超过 1.4 倍，而且随模型规模增大而减小；配合 muP 和 $1/\text{width}$ 权重衰减，Muon 在 190M 到 1.4B 之间保持约 1.4 倍加速（arXiv:2512.05620），但若超参数缩放方式错误，收益便会消失。加速比的说法必须表述为规模的函数。
- **MoE 分配定律（2026-03；arXiv:2603.10379）。** 计算最优的专家:注意力计算比随总计算量按幂律增长，并依赖于稀疏度。相关的 ICLR 2026 工作（二手来源）发现，记忆与推理的最优稀疏度不同，推理有时更偏好更稠密的模型。
- **精度以"有效 token 税"的形式进入定律（2025-09；arXiv:2509.25149）。** 一个 12B 模型以 NVFP4 在 10T token 上预训练，损失曲线紧跟 FP8，而 MXFP4 需要多约 36% 的 token 才能追平——也就是说，数值格式会改变 $D_{\text{eff}}$。Nemotron 3 Super（120B-A12B）以 NVFP4 在约 25T token 上完成预训练。
- **超越重复的有效数据（2026-04/05）。** SynPro（arXiv:2605.17849）报告，在 Chinchilla 数据量 10% 的设定下，借助 RL 训练的改写器可获得相当于简单重复 3.7–5.2 倍的有效 token；FinePhrase（arXiv:2604.13977）发现参数超过约 1B 的改写生成器不再带来额外收益。
- **中期训练决定 RL 之后的上限（2026-03；arXiv:2603.17074）。** 在中期训练（mid-training）中加入科学数据，RL 之后 GPQA-Diamond 提升 17–28 分，而改变 RL 数据配比带来的变化不到 2 分；中期训练会重构超过 90% 的权重，而 RL 只稀疏地更新约 5%。
- **能力趋势线。** METR 的 50% 时间跨度最初约每 7 个月翻一番（arXiv:2503.14499）；2026 年 1 月的 TH1.1 任务集给出 2023 年后的翻倍时间约为 4–4.5 个月，到 2026 年年中前沿模型的时间跨度已 ≥16 小时，任务集在此开始饱和（二手来源）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"陈述并解释 Chinchilla 缩放定律。"** $L(N,D)$ 的形式、每参数约 20 token，以及"Gopher 训练不足"这一关键结论。
- **"为什么 Kaplan 与 Chinchilla 结论不同？"** 针对每个训练长度的学习率调度 + 参数计数方式。
- **"你有 $10^{24}$ FLOPs——模型该多大、数据该多少？"** 用 $C=6ND$ 和 $N\approx D/20$ 求解。
- **"为什么要把小模型过度训练到超过计算最优点？"** 推断成本主导了全生命周期成本。
- **"估算一个总参数 1T / 激活参数 32B 的 MoE 在 15.5T token 上的训练 FLOPs。"** $6\cdot N_{\text{active}}\cdot D\approx3\times10^{24}$；注意内存和通信随*总*参数量增长。
- **"你如何判断一个 RL 配方是否值得扩大规模？"** 用早期的计算量-奖励数据点拟合一条 sigmoid（ScaleRL）；比较渐近线 $A$，而不只是早期斜率。

## 开放问题
**涌现（emergence）之争**尚无定论：Wei et al.（2022）报告能力会在某个规模上突然出现，而 Schaeffer et al.（2023）认为这是不连续/非线性指标造成的假象——在平滑指标下，提升是渐进的。是否真的存在不连续的能力跃迁，对安全预测至关重要。同样悬而未决的还有：推理/智能体（agent）能力的缩放定律（RL 渐近线、环境数量的缩放）、合成数据的真实价值、关于总参数与激活参数以及精度的联合定律，以及数据墙是否会封顶当前范式。

## 参考文献
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal LLMs (Chinchilla).* arXiv:2203.15556.
- Wei, J. et al. (2022). *Emergent Abilities of LLMs.* arXiv:2206.07682.
- Schaeffer, R. et al. (2023). *Are Emergent Abilities a Mirage?* arXiv:2304.15004.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained LMs.* arXiv:2305.16264.
- Kwa, T. et al. (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- Wen, K. et al. (2025). *Fantastic Pretraining Optimizers and Where to Find Them.* arXiv:2509.02046.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
- Khatri, D. et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs.* arXiv:2510.13786.
