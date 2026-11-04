# 训练稳定性（Training Stability）

> **最后更新：** 2026-11-04
> **相关文件：** [优化器选择](03_optimizer_choices.md) · [分布式训练](01_distributed_training.md) · [混合精度与量化](04_mixed_precision_and_quantization.md)
> **关键论文：** Yang et al. 2022 muP/μTransfer（[arXiv:2203.03466](https://arxiv.org/abs/2203.03466)）· Chowdhery et al. 2022 PaLM（损失尖峰）（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)）· Zhang et al. 2022 OPT logbook（[arXiv:2205.01068](https://arxiv.org/abs/2205.01068)）· Wortsman et al. 2023 Small-scale proxies for stability（[arXiv:2309.14322](https://arxiv.org/abs/2309.14322)）· Kimi Team 2025 Kimi K2（MuonClip/QK-Clip）（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）

## 概述
在前沿规模下，一次训练可能耗资数千万美元、持续数月，因此**稳定性**——避免发散、损失尖峰（loss spike）和隐性退化——至关重要。在小规模下看不见的不稳定性（1B 模型训练得好好的）可能让一个 100B+ 的训练脱轨。已公开的"训练日志"（OPT、BLOOM、GLM-130B）读起来像事故报告：自发的损失尖峰、NaN、硬件引起的比特翻转，以及为恢复而采取的人工干预（回滚、跳过批次、降低学习率）。理解不稳定性的*成因*以及*预防性*的架构/优化选择，对任何预训练岗位都必不可少——而自 2025 年起，同样的方法论也延伸到了长时间的 **RL 后训练**。

## 核心概念
**损失尖峰。** 训练损失的突然跃升，有时可恢复，有时是致命的。成因包括：大梯度范数撞上损失曲面的尖锐区域、注意力 logit 增长（softmax 饱和）、fp16 数值溢出，以及与特定数据批次相关。**注意力 logit**（$QK^\top$ 幅值爆炸）和**输出 logit** 的增长是常见元凶。

**梯度裁剪（gradient clipping）。** 限制全局梯度范数：若 $\|g\| > c$，则缩放 $g \leftarrow g\cdot c/\|g\|$。这是对抗尖峰的标准第一道防线；典型取 $c=1.0$。也存在逐层或自适应的裁剪变体。

**学习率日程。** **预热（warmup）**（在最初约数千步内线性提升学习率）可避免训练早期梯度大、Adam 二阶矩估计尚不成熟时的不稳定。**冷却/衰减**（cosine 或 WSD）在后期降低学习率，使模型落入极小值（见[优化器](03_optimizer_choices.md)）。

**初始化与 μP。** 缩放初始化（例如 $1/\sqrt{d}$，把残差分支按 $1/\sqrt{2L}$ 缩放）控制激活/梯度方差随深度的变化。**最大更新参数化（Maximal Update Parametrization，μP）**（Yang et al.）对初始化和逐层学习率进行参数化，使*最优超参数与宽度无关*——从而实现 **μTransfer**：在小模型上调超参，直接迁移到大模型而无需重新调优。这降低了昂贵训练的风险。

**数值稳定手段。**
- **z-loss**：加入 $\lambda(\log Z)^2$ 惩罚 softmax 的对数配分函数 $Z$，使 logit 保持有界（PaLM，被广泛采用）。
- **QK-norm**：在注意力之前对查询和键施加 LayerNorm/RMSNorm，抑制 logit 增长（对超大/长上下文训练有帮助）。
- **QK-Clip**（Kimi K2）：一种权重空间的替代方案。每次优化器更新后，对于最大 softmax 前 logit $S^h_{\max}$ 超过阈值 $\tau$ 的每个头 $h$，把 $W_q^h, W_k^h$ 乘以 $\sqrt{\gamma_h}$（$\gamma_h=\tau/S^h_{\max}$），在不改动前向计算的情况下恢复 logit 上界。它适用于 MLA——在 MLA 中推断时不会物化每个头的键，QK-norm 用起来很别扭。
- **注意力之后的输出（sigmoid）门控**：消除注意力汇（attention sink）和巨量激活，并能承受更大的学习率（Qwen 门控注意力）。
- **bf16 优于 fp16**：更宽的动态范围可避免溢出（见[混合精度](04_mixed_precision_and_quantization.md)）。
- **嵌入/logit 软截断**（Gemma 2）：用 tanh 把 logit 截断到固定范围。

## 关键挑战
- **依赖规模的不稳定性。** 在约 10B 参数以下不会出现的病态现象，使小规模调试不可靠。
- **取证困难。** 在训练中途区分坏批次、硬件故障（静默数据损坏/比特翻转）与真正的优化不稳定性并不容易。
- **大规模下的硬件故障。** 在 1 万块以上 GPU 时，SDC（静默数据损坏）、ECC 错误和 NCCL 挂起是家常便饭，并可能伪装成模型不稳定。
- **干预代价高。** 回滚到检查点浪费算力；跳过批次或降低学习率未必能解决根本原因。
- **优化器引起的 logit 增长。** 更快的优化器（Muon）和更宽的残差流（Hyper-Connections）打开了通往注意力 logit 或信号爆炸的新途径，而 AdamW 时代的配方无需应对这些问题。
- **RL 特有的不稳定性。** 熵坍缩以及训练器与 rollout 之间的数值不匹配会让长时间的 RL 训练漂移或崩溃，MoE 策略尤甚。

## 解决方案与当前最佳实践
**预防**：bf16、μP/缩放初始化、残差缩放、学习率预热、梯度裁剪、z-loss，以及（对大规模/长时间训练）QK-norm 或 QK-Clip、输出门控和 logit 软截断。**检测**：监控梯度范数、损失、每个头的最大注意力 logit 和逐层激活统计；使用能低成本复现不稳定性的**小规模代理**（Wortsman et al. 2023）来挑选稳定的配置。**恢复**：频繁保存检查点，出现尖峰时**回滚并跳过出问题的数据批次**（PaLM 公开的配方），或短暂降低学习率。确定性、可恢复的数据加载必不可少。**针对 RL**：匹配训练器与 rollout 的数值（FP16 或批次不变内核、FP32 的 LM-head logit），监控策略熵，并对 MoE 策略使用序列级比率。

## 各实验室观点
- **Google** 记录了 PaLM 中的损失尖峰和跳过批次的恢复方法；推广了 z-loss 和 μP 风格的缩放。
- **Meta（OPT）**公开了一份坦诚记录不稳定性与人工干预的训练日志——一份奠基性参考。
- **DeepMind/Microsoft** 推进了用于超参数迁移的 μP/μTransfer。
- **月之暗面（Moonshot）**提出了 QK-Clip（MuonClip），并报告一次 15.5T token、1T 参数的训练零损失尖峰；据报道 Kimi K3 的报告把 2.8T 规模下的稳定性归功于"SiTU-GLU"和"Quantile Balancing"。
- **Qwen** 落地了输出门控注意力（Qwen3-Next）；**DeepSeek** 提出流形约束超连接（mHC）来稳定加宽的残差流。
- **OpenAI/Anthropic** 不公开配方，但显然在稳定性工具和监控上投入巨大；Anthropic 曾撰文强调可复现、确定性基础设施的重要性。

## 最新进展（2023–2026年11月）
**μTransfer** 越来越成为降低超参选择风险的标准做法。**QK-norm 和 logit 软截断**（Gemma 2、OLMo 2）成为常见的稳定手段。**fp8 训练**（H100）重新引出了动态范围方面的稳定性问题，通过逐张量缩放（Transformer Engine）加以解决。更好的**静默数据损坏检测**和弹性恢复在 10 万 GPU 级别的训练中趋于成熟。"无尖峰"配方和基于小代理的稳定性预测是活跃的工程方向。

**2025年中 → 2026年11月更新：**
- **QK-Clip 让 Muon 在 1T 规模上变得安全（2025-07）。** Kimi K2 的 MuonClip 在最大注意力 logit 超过阈值时按头重新缩放 $W_q, W_k$，在一个 1T 参数 MoE 的 15.5T token 训练中防止了 logit 爆炸，且没有出现损失尖峰（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）。
- **门控注意力消除注意力汇（2025-05，NeurIPS 2025）。** 在 SDPA 之后加一个按头的 sigmoid 门，为低秩的 V–O 路径增加了非线性并带来依赖输入的稀疏性；首 token 注意力占比从 46.7% 降至 4.8%，更大的学习率变得稳定，RULER 提升超过 10 分（[arXiv:2505.06708](https://arxiv.org/abs/2505.06708)）。
- **用于加宽残差流的 mHC（2025-12）。** Hyper-Connections 破坏了恒等映射并使训练不稳定；把流之间的混合矩阵投影到双随机矩阵上（Birkhoff 多面体，通过 Sinkhorn–Knopp），可在 27B 规模恢复稳定的信号传播（[arXiv:2512.24880](https://arxiv.org/abs/2512.24880)）。据报道 DeepSeek-V4 使用了它，但存在争议。
- **4 比特稳定性配方（2025-09 → 2026-04）。** NVFP4 预训练只有在对 Wgrad 输入施加随机 Hadamard 变换、使用随机舍入、并把少数敏感层保留为 BF16 的情况下，才能维持在约 FP8 的质量（[arXiv:2509.25149](https://arxiv.org/abs/2509.25149)）；在 Nemotron 3 中，由于下溢清零（flush-to-zero），Mamba 输出投影不得不保留为 MXFP8。
- **RL 数值问题即稳定性问题（2025-09/10）。** BF16 的 7 位尾数使训练器与 rollout 引擎的对数概率产生偏离；把 RL 切换到 FP16 基本消除了这种不匹配（[arXiv:2510.26788](https://arxiv.org/abs/2510.26788)）。ScaleRL 的配方把 LM-head logit 保持为 FP32（[arXiv:2510.13786](https://arxiv.org/abs/2510.13786)）；GSPO 的序列级重要性比率修复了曾导致大型 MoE RL 崩溃的 token 级重要性采样噪声（[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）。
- **熵坍缩是可预测的（2025-05）。** 在不加干预时，RL 以 $R=-a\,e^{H}+b$ 的形式用熵换取奖励，其驱动因素是对数概率与优势之间的协方差；Clip-Cov / KL-Cov 针对的就是高协方差 token（[arXiv:2505.22617](https://arxiv.org/abs/2505.22617)）。
- **混合架构中的巨量激活（2026-08，单一来源）。** 一项最大 1.3B、受控的 Gated-DeltaNet 混合架构预训练研究报告，激活尖峰聚集在全注意力层之前（[arXiv:2608.12149](https://arxiv.org/abs/2608.12149)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"什么导致训练损失尖峰，如何处理？"** logit/梯度增长、坏批次、fp16 溢出；裁剪、z-loss、QK-norm、回滚 + 跳过批次。
- **"什么是 μP / μTransfer，为什么重要？"** 最优超参与宽度无关 → 小模型上调、大模型上用；降低大规模训练的风险。
- **"预训练为什么用 bf16 而不是 fp16？"** 更宽的指数范围避免溢出，无需损失缩放。
- **"如何区分硬件故障与优化不稳定性？"** 可复现性检查、梯度/激活监控、ECC/SDC 日志。
- **"QK-norm 与 QK-Clip——Kimi 为什么需要后者？"** QK-norm 在前向中归一化激活（与 MLA 不物化键的方式不太兼容）；QK-Clip 在更新之后通过重新缩放权重来限制最大 logit，直接针对 Muon 引起的 logit 增长。
- **"既然预训练用 BF16，为什么 RL 要用 FP16？"** RL 的重要性比率依赖于训练器与采样器在对数概率上的一致；FP16 更多的尾数位能缩小不匹配，而预训练之后梯度尺度已经良好，动态范围就没那么关键了。

## 开放问题
目前没有关于损失尖峰的预测性*理论*——缓解手段大多是经验性的。用小代理可靠地预测大规模不稳定性正在进步但尚未解决。随着训练扩展到 10 万+ 加速器和更低精度（fp8/fp4），数值精度、优化器动态（Muon 族）、架构（混合架构、加宽残差、极端 MoE 稀疏度）和硬件故障之间的相互作用带来了新的、尚未被充分理解的失效模式；RL 阶段的稳定性（熵、不匹配、MoE 路由漂移）仍缺乏统一的处理框架。

## 参考文献
- Yang, G. et al. (2022). *Tensor Programs V: μTransfer.* arXiv:2203.03466.
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Zhang, S. et al. (2022). *OPT (logbook).* arXiv:2205.01068.
- Wortsman, M. et al. (2023). *Small-scale Proxies for Large-scale Transformer Training Instabilities.* arXiv:2309.14322.
- Dehghani, M. et al. (2023). *ViT-22B (QK-norm).* arXiv:2302.05442.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence (MuonClip).* arXiv:2507.20534.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- DeepSeek-AI (2025). *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880.
- Cui, G. et al. (2025). *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* arXiv:2505.22617.
