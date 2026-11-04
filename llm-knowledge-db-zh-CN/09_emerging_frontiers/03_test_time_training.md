# 测试时训练（Test-Time Training）

> **最后更新：** 2026-11-04
> **相关文件：** [非 Transformer 架构](04_neuromorphic_and_non_transformer.md) · [扩散与混合语言模型](06_diffusion_and_hybrid_language_models.md) · [推断时扩展](../04_reasoning_and_agents/02_inference_time_scaling.md) · [持续学习](01_continual_learning.md)
> **关键论文：** Sun et al. 2020 Test-Time Training ([arXiv:1909.13231](https://arxiv.org/abs/1909.13231)) · Sun et al. 2024 TTT layers / "Learning to (Learn at Test Time)" ([arXiv:2407.04620](https://arxiv.org/abs/2407.04620)) · Gu & Dao 2023 Mamba ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) · Akyürek et al. 2024 TTT for ARC ([arXiv:2411.07279](https://arxiv.org/abs/2411.07279)) · TTT-E2E 2025 ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675))

## 概述
测试时训练（test-time training，TTT）的思想是：模型可以**在推断（inference）过程中更新自身参数**，针对特定输入或上下文进行适应，而不是纯粹以冻结的权重运行。它模糊了训练与测试的边界：模型不是（或不仅是）生成更多 token（推断时*扩展*），而是从测试实例本身*学习*。TTT 涵盖两条相关的研究线：（1）**TTT 层**——一种新的序列建模原语，其隐藏状态本身就是一个通过在序列上做梯度下降来更新的小模型（是 RNN/状态空间模型（SSM）状态的推广）；（2）面向困难推理的**测试时适应**（在作答之前，在测试问题的增强版本上微调，正如 ARC-AGI 的创纪录结果所示）。两者都是连接架构、持续学习和推理的活跃前沿。

## 核心概念
**经典 TTT（Sun et al. 2020）。** 在测试时，先在每个输入上通过一个**自监督辅助任务**（例如旋转预测）来适应模型，再做主预测——以提升对分布偏移的鲁棒性。模型对每个测试样本执行几步梯度更新。

**TTT 层（Sun et al. 2024）——序列模型视角。** 把**序列模型的隐藏状态重新表述为一个（小）模型**，其权重在**每个时间步通过自监督梯度下降**来更新。RNN 把历史压缩成一个固定向量，注意力保留全部历史（二次复杂度），而 TTT 层把历史压缩进一个在线更新的*快速内部模型的权重*中——从而以**线性复杂度**获得一个*表达力强*、可学习的记忆。"更新规则"本身也是学出来的（元学习："学会在测试时学习"）。TTT-Linear/TTT-MLP 在长上下文上可与 Mamba 和 Transformer 竞争，其吸引力在于记忆可以随 token 增多而持续改进。

**面向推理的 TTT（测试时适应）。** 对于非常困难的分布外任务（例如 **ARC-AGI** 抽象推理），生成测试问题的增强变体（变换、留一示例），**在推断时用这些变体微调模型**，然后求解——带来大幅准确率提升（ARC 上的最先进结果）。这就是"从测试问题自身的结构中学习"。

**与 SSM 和快速权重的关系。** TTT 层推广了**线性注意力 / 状态空间模型**（循环状态更新是其特例），并复兴了经典的"**快速权重（fast weights）**"思想（Schmidhuber）——即随输入快速变化、独立于慢速基座权重的权重。最清晰的桥梁是**门控 delta 规则**：$S_t=\alpha_t S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top$ 恰好是在回归损失 $\tfrac12\|S k_t-v_t\|^2$ 上做一步在线梯度下降（步长 $\beta_t$，通过 $\alpha_t$ 实现权重衰减）——即一个线性 TTT 层。其*覆盖而非累加*的行为加上有界状态，正是 Gated DeltaNet/KDA 胜过加性线性注意力的原因，而这些层如今已在生产级混合模型中运行（Qwen3-Next、Kimi Linear/K3）。

**面向长上下文的端到端 TTT。** 不使用专门的循环层，而是拿一个标准（滑动窗口）Transformer，让它在推断时基于上下文用下一 token 梯度步更新自身的一部分 MLP 权重；并对初始化进行元学习（梯度的梯度），使这些更新步尽可能有用。长上下文由此变成*在提示上的持续学习*。

## 关键挑战
- **推断时的计算成本。** 每 token（TTT 层）或每问题（推理 TTT）的梯度更新会显著增加推断成本/延迟。
- **稳定性。** 在线更新有发散风险；内循环的学习率/更新规则必须精心设计或元学习得到。
- **硬件效率。** TTT 层需要硬件高效的内循环 kernel（mini-batch TTT），才能与优化过的注意力/Mamba kernel 竞争。
- **遗忘/作用范围。** 测试时更新通常是暂时的（每个输入后重置）；持久更新会重新引入持续学习问题（见[持续学习](01_continual_learning.md)）。
- **通用性。** 推理 TTT 的收益在狭窄、结构化的分布外任务上最明显；其普适性尚未得到证明——而交互式基准（ARC-AGI-3）需要探索，这是单实例微调本身无法提供的。
- **服务部署。** 每请求的权重更新会打破推断引擎在批处理/前缀缓存上的假设；元学习式 TTT 还需要二阶训练。

## 解决方案与当前最佳实践
**架构**方面：delta 规则层（Gated DeltaNet、KDA）——TTT 层的生产级后代——以**与全注意力约 3:1 的混合**形式部署；更丰富的 TTT 层（TTT-MLP、Titans/ATLAS 记忆）以及 **TTT-E2E** 仍是面向**具有可学习、线性成本记忆的长上下文**的研究阶段替代方案；应使用硬件感知的分块/mini-batch 内循环。**困难推理**方面：**在增强的测试实例上做测试时微调**是一种强大但昂贵的技术（ARC-AGI）。实践中，大多数已部署的"测试时计算"仍是**推断时*扩展***（更多 token/搜索，见[推断时扩展](../04_reasoning_and_agents/02_inference_time_scaling.md)）而非参数更新，但 TTT 是一个快速增长的互补方向。

## 各实验室观点
- **学术界（Stanford/CMU/MIT）：** TTT 层（Sun et al.）、用于 ARC 的 TTT（Akyürek et al.）——研究的推动者。
- **SSM/高效架构社区：** 把 TTT 视为线性注意力/SSM 连续谱的一部分（Mamba、DeltaNet、Gated DeltaNet、RWKV-7 的"表达性状态"）。
- **前沿实验室：** 主要追求推断时*扩展*（推理 token/搜索，如今已产品化为 effort 等级和自适应思考）；TTT 式参数适应更具探索性。**Google** 是最活跃的闭源实验室（Titans → ATLAS → Nested Learning/HOPE）。
- **中国开放实验室（Qwen、Moonshot）：** 大规模交付 delta 规则快速权重层（Qwen3-Next/3.5 的 Gated DeltaNet；Kimi Linear 以及具有 69 个 KDA 层 + 24 个 MLA 层的 Kimi K3）。
- **ARC Prize / 推理研究者：** 把测试时适应视为通往抽象、分布外泛化的道路。

## 最新进展（2023–2026年11月）
**TTT 层**作为一种具有*可学习*记忆的可信线性复杂度序列原语出现，在长上下文上可与 Mamba/Transformer 竞争。**面向 ARC-AGI 的测试时训练**取得了突出的抽象推理结果，使参数级测试时适应受到关注。更广泛的**"表达性循环状态"**趋势（Gated DeltaNet、RWKV-7、Titans）与 TTT 高度重叠，统一了 SSM、快速权重和在线学习。争论焦点：参数级测试时学习是否是 token 级推断扩展之后的下一个维度？

**2025 年中 → 2026 年 11 月更新：**
- **ATLAS（2025-05，Google；ICML 2026）。** Titans 的后继者，其记忆在过去 token 的*滑动窗口*上优化（Omega 规则），而不只针对最新 token，并具有更高容量的记忆和"DeepTransformers"；报告在 10M token 的 BABILong 上提升 80%（[arXiv:2505.23735](https://arxiv.org/abs/2505.23735)）。
- **delta 规则快速权重进入生产（2025-09 → 2026-07）。** Qwen3-Next（3 个 Gated DeltaNet : 1 个门控注意力）、Kimi Linear 的 KDA（通道级门控；声称在公平比较下胜过全注意力，1M 下解码最高提速 6×；[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）以及 Kimi K3（总参数 2.8T，69 个 KDA + 24 个 MLA 层）使 TTT 式在线回归成为前沿规模的原语。
- **Nested Learning / HOPE（2025-11，Google，NeurIPS 2025）。** 把架构与优化器视为以不同更新频率运行的嵌套学习器；HOPE 加入了由多种速率更新的 MLP 组成的连续记忆系统——把 TTT 推广为一个时间尺度层级。
- **TTT-E2E（2025-12）。** 滑动窗口 Transformer + 在部分 MLP 上执行测试时下一 token 梯度步 + 元学习初始化；在 3B 规模上像全注意力一样随上下文扩展，同时保持每 token 延迟恒定，在 128K 下比全注意力快 2.7×（[arXiv:2512.23675](https://arxiv.org/abs/2512.23675)）。
- **Mamba-3（2026-03，ICLR 2026 oral）。** 指数-梯形离散化、用于状态追踪的复数值状态转移，以及提高解码算术强度的 MIMO 更新；以一半的状态规模达到 Mamba-2 的困惑度（[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)）。
- **ARC 超越静态谜题（2026）。** ARC-AGI-2 已被前沿 API 模型接近饱和（90–95%，聚合站数据），受效率约束的 Kaggle 赛道据报道最高超过 88%（单一来源）；**ARC-AGI-3**（2026-03）是一个交互式游戏基准，发布时所有前沿模型得分都低于 1%——把考察重点从单实例适应转向探索与目标推断。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"什么是测试时训练，它与推断时扩展有何不同？"** 根据测试输入更新*参数* vs 用冻结权重生成更多 *token*/做搜索。
- **"把 TTT 层作为序列模型来解释。"** 隐藏状态 = 一个按 token 用自监督梯度下降更新的小模型 → 线性成本、表达力强的可学习记忆；推广了 RNN/SSM/线性注意力。
- **"为什么 TTT 对 ARC-AGI 有帮助？"** 在增强的测试实例上微调，利用问题自身结构实现分布外泛化。
- **"TTT 与快速权重/SSM 是什么关系？"** 它是快速权重的现代、元学习形式；SSM 递推是其特例。
- **"证明门控 delta 规则是一个 TTT 层。"** $\tfrac12\|Sk_t-v_t\|^2$ 对 $S$ 的梯度是 $(Sk_t-v_t)k_t^\top$；在衰减 $\alpha_t$ 之后以学习率 $\beta_t$ 走一步 SGD，得到 $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$（为了稳定性，键做 $L_2$ 归一化）。

## 开放问题
**参数级测试时学习**能否成为主流维度（相对于 token 级扩展）、让 TTT 在前沿规模上**硬件高效**、确保在线更新的**稳定性**，以及把推理 TTT 扩展到狭窄结构化任务之外，都是开放问题。更深层的问题——如何把慢速（预训练）学习与快速（测试时）学习统一为稳健的持续/终身学习系统——仍未解决。

## 参考文献
- Sun, Y. et al. (2020). *Test-Time Training.* arXiv:1909.13231.
- Sun, Y. et al. (2024). *Learning to (Learn at Test Time): TTT layers.* arXiv:2407.04620.
- Akyürek, E. et al. (2024). *The Surprising Effectiveness of TTT for Abstract Reasoning.* arXiv:2411.07279.
- Gu, A., Dao, T. (2023). *Mamba.* arXiv:2312.00752.
- Yang, S. et al. (2024). *Gated DeltaNet.* arXiv:2412.06464.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
- Stanford/NVIDIA et al. (2025). *End-to-End Test-Time Training for Long Context (TTT-E2E).* arXiv:2512.23675.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- CMU/Princeton/Together AI (Gu, Dao et al.) (2026). *Mamba-3.* arXiv:2603.15569.
