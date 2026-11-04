# 优化器选择（Optimizer Choices）

> **最后更新：** 2026-11-04
> **相关文件：** [训练稳定性](02_training_stability.md) · [缩放定律](../00_foundations/04_scaling_laws.md) · [数据配比与课程学习](../01_data/03_data_mixing_and_curriculum.md)
> **关键论文：** Kingma & Ba 2014 Adam（[arXiv:1412.6980](https://arxiv.org/abs/1412.6980)）· Loshchilov & Hutter 2017 AdamW（[arXiv:1711.05101](https://arxiv.org/abs/1711.05101)）· Chen et al. 2023 Lion（[arXiv:2302.06675](https://arxiv.org/abs/2302.06675)）· Liu et al. 2023 Sophia（[arXiv:2305.14342](https://arxiv.org/abs/2305.14342)）· Jordan et al. 2024 Muon · Liu et al. 2025 Muon is Scalable（[arXiv:2502.16982](https://arxiv.org/abs/2502.16982)）· Wen et al. 2025 Fantastic Pretraining Optimizers（[arXiv:2509.02046](https://arxiv.org/abs/2509.02046)）

## 概述
优化器（optimizer）把梯度映射为参数更新；在 LLM 规模下，这一选择影响收敛速度、稳定性和最终损失——直接折算为数百万美元的算力。在过去十年的大部分时间里，**AdamW 是 Transformer 预训练压倒性的默认选择**，至今仍是参照基线。然而自 2025 年起，**Muon**（针对矩阵参数的正交化更新）已在开放的前沿模型中达到万亿参数规模的生产级水准（Kimi K2、GLM-5，据报道还有 DeepSeek-V4），而严谨的基准测试表明，它相对于*调优充分的* AdamW 的真实优势并不大。理解 Adam *为什么*曾占主导、它在哪里失效、Muon 族方法究竟带来了什么，以及**学习率（learning rate）日程**（其重要性可以说不亚于优化器本身），是常见的面试主题。

## 核心概念
**Adam / AdamW。** Adam 维护梯度（一阶矩 $m_t$）及其平方（二阶矩 $v_t$）的指数移动平均：
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
偏差校正为 $\hat m_t=m_t/(1-\beta_1^t)$、$\hat v_t=v_t/(1-\beta_2^t)$，更新为 $\theta_t=\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$。逐坐标的自适应缩放能应对 Transformer 各参数之间差异悬殊的梯度幅值。**AdamW** 把权重衰减与梯度解耦（直接施加 $-\eta\lambda\theta$），而不是把它并入 $g$——这是自适应优化器的正确形式，如今已是通用做法。典型取值：$\beta_1=0.9$，$\beta_2=0.95$（LLM 中低于视觉领域的 0.999），$\epsilon=10^{-8}$，权重衰减 0.1。

**Adafactor。** 把二阶矩矩阵分解，使内存从 $O(nm)$ 降为 $O(n+m)$（次线性），以少量质量换取大幅内存节省；在优化器状态内存成为瓶颈时使用（T5、部分 TPU 训练）。

**Lion。** 基于符号：更新使用插值动量的*符号*，$\theta\mathrel{-}=\eta\,\text{sign}(\beta_1 m + (1-\beta_1)g)$，只保存动量（Adam 状态的一半）。有竞争力、内存占用小，但对学习率/权重衰减的调优更敏感。

**Sophia。** 一种轻量的**二阶**方法，用对角 Hessian 估计对更新进行预条件处理，并逐坐标裁剪；声称在某些设置下预训练提速约 2 倍，但其在前沿规模上的稳健性存在争议。

**Muon。**（2024）对二维权重矩阵进行优化，通过 Newton-Schulz 迭代把动量更新**正交化**（由动量的 SVD $M=U\Sigma V^\top$ 近似得到 $UV^\top$），使更新在各奇异方向上更为均匀。它只保存动量（一个状态缓冲区），对非矩阵参数（嵌入、输出头、归一化层）则配合使用 AdamW。要让它在大规模下工作，需要（Moonlight，2025）：**解耦权重衰减**，以及与 AdamW **匹配更新 RMS**（对 $A\times B$ 矩阵把正交化后的更新乘以 $\approx 0.2\sqrt{\max(A,B)}$），使为 AdamW 调好的学习率可以直接迁移；在超大规模下还需要 **QK-Clip** 来遏制注意力 logit 增长（见[训练稳定性](02_training_stability.md)）。

**Dion。** 一种通信高效的正交化优化器：对动量做摊销幂迭代，只对秩为 $r$ 的子空间做正交化，因此可直接作用于 FSDP/TP 分片后的矩阵，无需收集完整矩阵（见[分布式训练](01_distributed_training.md)）。

**学习率日程。**
- **Cosine 衰减**：预热后以余弦退火降到很小的最终学习率；长期以来的默认选择。需要事先确定总步数。
- **WSD（Warmup-Stable-Decay）**：预热 → 长时间*恒定*学习率（稳定阶段）→ 短暂的快速衰减。它把日程与总长度解耦（可以从任意检查点开始衰减），支持**继续/分支训练**，并且天然搭配**数据退火**（在衰减学习率的同时切换到高质量数据）。越来越受青睐（MiniCPM、DeepSeek、OLMo 2）。

## 关键挑战
- **优化器状态内存。** Adam 以 fp32 保存 $2\times$ 参数量的状态（$8\Psi$ 字节）——是内存墙的主要来源之一（见[分布式训练](01_distributed_training.md)）。
- **超参数敏感性。** 学习率、$\beta_2$、权重衰减、预热长度相互作用；选得不好会导致尖峰或收敛缓慢。
- **新优化器的规模泛化。** 许多替代方案在小规模上胜出，却无法在 100B+ 上稳健地击败 AdamW；门槛很高，而且**调优不足的基线**夸大了许多已报告的加速比。
- **日程僵化。** Cosine 需要固定的训练长度；中途改变 token 预算很别扭（这推动了 WSD 的出现）。
- **矩阵优化器的分片。** 正交化需要完整矩阵，这与 ZeRO/FSDP 分片相冲突，并增加了通信。

## 解决方案与当前最佳实践
**AdamW 配合 $\beta_2=0.95$、权重衰减 0.1、梯度裁剪 1.0、学习率预热 + cosine 或 WSD**，依然是稳妥的默认选择。**对二维隐藏层矩阵使用 Muon（或 MuonClip）+ 对嵌入/输出头/归一化层使用 AdamW**，现已是经过验证的前沿选项——前提是配合权重衰减、与 AdamW 匹配 RMS 的更新尺度、μP 风格的宽度缩放（包括约 $1/\text{width}$ 的权重衰减），使收益在规模扩大后依然保留，并用 QK-Clip 或 QK-norm 控制 logit。评估任何新优化器时，都要与**针对每个优化器分别调优**的 AdamW 对比。用 **μP** 在不同规模间迁移学习率（见[训练稳定性](02_training_stability.md)）。需要检查点灵活的衰减和干净的退火时采用 **WSD**。用 ZeRO/FSDP 分片优化器状态以放进内存。

## 各实验室观点
- **OpenAI/Anthropic/Meta** 历史上以 AdamW 变体作为生产默认（当前配方未公开）。
- **Google** 长期在 TPU 上使用 **Adafactor** 以节省内存（T5、PaLM 时代），同时也用 Adam。
- **DeepSeek/MiniCPM/OLMo** 推广了 **WSD** 日程，并公开了 μP/超参数缩放研究；据报道 DeepSeek-V4 以 **Muon** 作为主优化器，嵌入、输出头和 RMSNorm 权重则使用 AdamW。
- **月之暗面（Moonshot）**让 Muon 变得可扩展（Moonlight），并用 **MuonClip** 训练了 Kimi K2（1T）；据报道**智谱（Zhipu）**（GLM-5）也使用 Muon 训练。
- **Microsoft** 提出了原生支持分片的正交化更新方法 **Dion**；**Stanford/Marin** 发布了迄今最严谨的优化器基准测试。

## 最新进展（2023–2026年11月）
**Muon** 是讨论最多的新优化器（正交化更新；墙钟时间收益显著，已被一些 2025 年的大模型采用）。**WSD** 日程和"**小批量预热 → 恒定 → 退火**"的配方已成主流。关于**超参数缩放定律（scaling law）**（最优学习率/批大小与计算量的关系，DeepSeek/Cerebras）的研究降低了调参成本。fp8 训练对优化器状态的精度提出了新问题。

**2025年中 → 2026年11月更新：**
- **Muon is Scalable / Moonlight（2025-02）。** 加入权重衰减并让逐参数更新的 RMS 与 AdamW 匹配后，Muon 就能扩展；报告计算效率约为 AdamW 的 2 倍（约需 AdamW 52% 的 FLOPs），并在一个用 5.7T token 训练的 16B-A3B MoE 上得到验证（[arXiv:2502.16982](https://arxiv.org/abs/2502.16982)）。
- **1T 规模的 MuonClip（2025-07）。** Kimi K2 用 Muon + 按头 QK-Clip 在 15.5T token 上训练了一个 1T 参数的 MoE，零损失尖峰（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）。
- **Dion（2025-04，2025-09 修订）。** 基于幂迭代的秩 $r$ 正交化，兼容 FSDP/TP 分片——是对 Newton–Schulz 需要完整矩阵这一问题的通信感知型回应（[arXiv:2504.05295](https://arxiv.org/abs/2504.05295)）。
- **现实检验：Fantastic Pretraining Optimizers（2025-09，ICLR 2026）。** 对 11 种优化器分别进行严格调优后做基准测试，相对调优充分的 AdamW，没有任何加速比超过 **1.4 倍**，且加速比**随模型规模增大而缩小**；矩阵预条件方法（Muon、SOAP）在小规模上胜出（[arXiv:2509.02046](https://arxiv.org/abs/2509.02046)）。
- **超参数迁移决定收益能否保留（2025-12）。** 在 μP 和 $1/\text{width}$ 权重衰减下，Muon 从 190M 到 1.4B 都保持了相对 AdamW 约 1.4 倍的加速；缩放方式错了，收益就会消失（[arXiv:2512.05620](https://arxiv.org/abs/2512.05620)）。
- **生产采用（2026）。** 据报道使用 Muon 训练的前沿开放模型包括 Kimi K2、GLM-5 和 DeepSeek-V4（依据 "Spectral Scaling Laws of Muon"，[arXiv:2606.04058](https://arxiv.org/abs/2606.04058)）；也有人提出针对 MLA + Muon MoE 的 μP 式学习率迁移（2026-08，单一来源，[arXiv:2608.20061](https://arxiv.org/abs/2608.20061)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么在 Transformer 上 Adam 胜过 SGD？"** 逐坐标的自适应性能应对异质的梯度尺度；SGD 需要大量调参，在这里表现更差。
- **"AdamW 与 Adam 有什么区别，为什么重要？"** 解耦的权重衰减；对自适应方法而言是正确的正则化方式。
- **"Cosine 与 WSD 的取舍？"** 固定训练长度 vs 检查点灵活的衰减 + 与退火的协同。
- **"Adam 增加了多少内存，如何管理？"** 约 $8\Psi$ 字节的 fp32 状态；用 ZeRO/FSDP 分片，或改用 Adafactor/Lion/Muon（单个缓冲区）。
- **"Muon 做了什么，要让它在 1T 规模上工作需要什么？"** 用 Newton–Schulz 把动量正交化（$\approx UV^\top$）；加入权重衰减，让更新的 RMS 与 AdamW 匹配，嵌入/输出头/归一化层保留 AdamW，并加入 QK-Clip 防止 logit 爆炸。
- **"一篇论文声称比 AdamW 快 2 倍——你怎么评估？"** 检查基线是否针对该优化器单独调优、随规模的变化趋势（收益随规模缩小）、超参迁移规则（μP、权重衰减缩放）、墙钟时间与通信成本，而不仅仅是步数。

## 开放问题
Adam 为什么在 Transformer 上（相对 SGD）如此有效，仍缺乏完整的理论（近期研究把它与重尾/异质梯度以及损失曲面的条件数联系起来）。正交化方法（Muon/Dion）或二阶方法（SOAP、Sophia）能否在最大规模上保持实质性优势仍有争议：一方面生产采用在增加，另一方面公平的基准测试显示收益随规模缩小。Muon 的谱缩放定律、学习率/批大小/权重衰减/日程的最优联合缩放，以及通信高效的矩阵优化器，仍是活跃的前沿方向。

## 参考文献
- Kingma, D., Ba, J. (2014). *Adam.* arXiv:1412.6980.
- Loshchilov, I., Hutter, F. (2017). *Decoupled Weight Decay (AdamW).* arXiv:1711.05101.
- Chen, X. et al. (2023). *Symbolic Discovery of Optimization Algorithms (Lion).* arXiv:2302.06675.
- Liu, H. et al. (2023). *Sophia.* arXiv:2305.14342.
- Hu, S. et al. (2024). *MiniCPM (WSD schedule).* arXiv:2404.06395.
- Liu, J. et al. (2025). *Muon is Scalable for LLM Training.* arXiv:2502.16982.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence (MuonClip).* arXiv:2507.20534.
- Ahn, K. et al. (2025). *Dion: Distributed Orthonormalized Updates.* arXiv:2504.05295.
- Wen, K. et al. (2025). *Fantastic Pretraining Optimizers and Where to Find Them.* arXiv:2509.02046.
