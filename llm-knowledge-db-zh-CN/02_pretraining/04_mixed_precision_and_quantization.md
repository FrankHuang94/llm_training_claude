# 混合精度与低精度训练（Mixed Precision and Low-Precision Training）

> **最后更新：** 2026-11-04
> **相关文件：** [训练稳定性](02_training_stability.md) · [计算与内存效率](05_compute_and_memory_efficiency.md) · [量化技术](../06_efficiency/02_quantization_techniques.md)
> **关键论文：** Micikevicius et al. 2017 Mixed Precision Training（[arXiv:1710.03740](https://arxiv.org/abs/1710.03740)）· Kalamkar et al. 2019 bfloat16（[arXiv:1905.12322](https://arxiv.org/abs/1905.12322)）· Micikevicius et al. 2022 FP8 Formats（[arXiv:2209.05433](https://arxiv.org/abs/2209.05433)）· DeepSeek-AI 2024 DeepSeek-V3（FP8 训练）（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）· NVIDIA 2025 Pretraining LLMs with NVFP4（[arXiv:2509.25149](https://arxiv.org/abs/2509.25149)）

## 概述
以降低的数值精度进行训练是杠杆最高的效率技术之一：与 fp32 相比，它大致将内存减半（fp16/bf16）或减为四分之一（fp8），并使张量核心吞吐量翻倍/翻两番，只要处理得当，质量几乎没有损失。fp32 → fp16/bf16 → fp8 → **fp4** 的演进与 GPU 硬件代际（Volta → Ampere → Hopper → Blackwell → Rubin）相对应；到 2026 年，4 比特（NVFP4）预训练已从研究走向生产级规模的训练。本文讨论*训练时*的数值问题；**推断量化**（GPTQ/AWQ/int4）见[量化技术](../06_efficiency/02_quantization_techniques.md)。

## 核心概念
**浮点格式。** 浮点数 = 符号位 + 指数（范围）+ 尾数（精度）。
- **fp32**：1/8/23——基线。
- **fp16**：1/5/10——精度高但*指数窄*（最大约 65504）；容易上溢/下溢 → 需要损失缩放。
- **bf16**：1/8/7——*与 fp32 相同的指数范围*（巨大的动态范围），但只有 7 位尾数。以精度换范围；是 **LLM 预训练的默认格式**，因为对稳定性而言范围比精度更重要，而且无需损失缩放。
- **fp8**：E4M3（1/4/3，精度更高）和 E5M2（1/5/2，范围更大）。Hopper/Blackwell 张量核心支持 fp8 矩阵乘法。
- **fp4（E2M1）**：只有 16 个可表示的值，只有配合细粒度的**块缩放**（微缩放）才可用。

**微缩放（块缩放）格式。** 每个小块中的元素共享一个缩放因子。**MXFP4**（OCP MX）：32 个元素为一块，使用 2 的幂（E8M0）缩放因子。**NVFP4**（Blackwell）：16 个元素为一块，使用 **E4M3** 缩放因子，*外加*一个逐张量的 FP32 缩放因子——更小的块和非 2 的幂的缩放因子能更紧密地跟踪局部动态范围，这就是 NVFP4 在训练中胜过 MXFP4 的原因。DeepSeek 的 **UE8M0** FP8 缩放因子同样是只含指数（2 的幂）的块缩放因子。

**混合精度训练（经典配方）。** 保留一份 **fp32 主权重副本**；前向/反向用 fp16/bf16 计算；矩阵乘法在 fp32 中累加。对 **fp16** 使用**损失缩放**：在反向传播前把损失乘以一个大因子 $S$，使小梯度不会下溢为零，再在优化器步骤前除回去（动态损失缩放在溢出时调整 $S$）。**bf16** 凭借其范围通常可以省去损失缩放。

**FP8 训练。** 把矩阵乘法的输入转换为 fp8，并使用跟踪每个张量动态范围的**逐张量（或更细粒度）缩放因子**（延迟缩放使用 amax 的历史记录）。NVIDIA 的 **Transformer Engine** 将其自动化。敏感部件（优化器状态、主权重、某些归约，有时还有注意力）保持更高精度。在 H100 上 fp8 的吞吐量约为 bf16 的 2 倍。

**FP4 训练（NVFP4 配方）。** 将 GEMM 输入量化为 NVFP4：对权重使用 **2D 块缩放**（使前向和反向看到一致的量化结果），对权重梯度（Wgrad）GEMM 的输入施加**随机 Hadamard 变换**以分散离群值，对梯度使用**随机舍入**以保持无偏，并把少数敏感层（主要是最后几个块）保留为 BF16。

## 关键挑战
- **上溢/下溢（fp16）。** 范围窄会导致 NaN/inf 或静默的梯度下溢；必须使用损失缩放，而且很繁琐。
- **精度损失（bf16/fp8）。** 尾数位少会造成可能累积的舍入误差（例如在大规模归约、LayerNorm、softmax 中）。
- **FP8 稳定性。** 离群值和不断变化的动态范围使逐张量缩放很微妙；某些层/操作必须保持更高精度。
- **FP4 偏差与下溢清零。** 每块只有 16 个量级时，确定性舍入会使梯度产生偏差，而极小的值会被清零（Nemotron 3 正因如此把 Mamba 输出投影保留为 MXFP8）。
- **确定性与可复现性。** 低精度加上非确定性的归约使不稳定性的调试变得复杂——在 RL 中，BF16 舍入还会让训练器与采样器产生分歧。

## 解决方案与当前最佳实践
**bf16 仍是稳妥的预训练默认选择**（fp32 主权重 + bf16 计算、fp32 累加、无需损失缩放）。对于 **fp16**（较旧的硬件），使用**动态损失缩放**。对于 **fp8**（Hopper 及以后），使用 **Transformer Engine** 配合逐张量/分块缩放，把优化器状态和敏感操作保持在更高精度，并仔细验证稳定性。对于 **fp4**（Blackwell 及以后），使用 **NVFP4** 配合 2D 权重缩放、对 Wgrad 的 Hadamard 变换、随机舍入，并让最敏感的层使用 BF16。始终**在 fp32 中累加梯度/归约**，并把 **LayerNorm/softmax/路由器（router）**的计算保持在更高精度。对于 **RL 后训练**，可考虑使用 **FP16**（尾数更多）以保持 rollout 与训练器的对数概率一致。

## 各实验室观点
- **Google** 为 TPU 联合开发了 **bfloat16** 并广泛使用；Ironwood（TPU v7）的单芯片指标约为 4.6 PF FP8。
- **NVIDIA** 通过 Transformer Engine 推动 fp8，如今又推动 **NVFP4**：Nemotron 3 Super（120B-A12B）和 Ultra 均以 NVFP4 预训练；Rubin 以 50 PF NVFP4 作为宣传指标。
- **DeepSeek** 在 V3 中展示了**生产级 fp8 预训练**，采用细粒度（tile/块）量化；随后在 V3.1 中推出面向国产芯片的 UE8M0 FP8 缩放因子，并在 V4 中对路由专家使用 **FP4 QAT**。
- **月之暗面（Moonshot）/ OpenAI** 把 4 比特推进到后训练和发布环节：Kimi K2 Thinking 使用 INT4 仅权重 QAT；gpt-oss 以 MXFP4 发布 MoE 权重。
- **Meta/OpenAI/Anthropic** 未公开当前的预训练精度；以 bf16 为主、最大的矩阵乘法用 fp8，是 2024 年前后的常规做法。

## 最新进展（2023–2026年11月）
**FP8 训练在前沿模型中成为主流**（DeepSeek-V3 详尽的 fp8 配方是关键参考）。**Blackwell** 硬件增加了 fp4 和微缩放（MX）格式；研究探索了结合块缩放和随机舍入的 **fp4 / 低于 8 比特训练**。**精度缩放定律**（Kumar et al. 2024）把低精度训练/推断的损失代价刻画为比特数、模型规模和 token 数的函数——为量化的激进程度提供依据。

**2025年中 → 2026年11月更新：**
- **最长的公开 FP4 训练（2025-09，v2 2026-03）。** NVIDIA 用 **NVFP4 在 10T token 上**训练了一个 12B 的 Mamba-Transformer 混合模型，损失和准确率接近 FP8；MXFP4 需要多约 36% 的 token 才能追平（[arXiv:2509.25149](https://arxiv.org/abs/2509.25149)）。
- **生产规模的 NVFP4（2025-12 → 2026-04）。** Nemotron 3 Super（120B / 12B 激活）在约 25T token 上以 NVFP4 预训练，报告与 BF16 的损失差距小于 1%；由于下溢清零，Mamba 输出投影保留为 MXFP8（[arXiv:2512.20856](https://arxiv.org/abs/2512.20856)，[arXiv:2604.12374](https://arxiv.org/abs/2604.12374)）。检查点以 NVFP4、FP8 和 BF16 三种精度发布。
- **面向特定硬件的 FP8 缩放因子（2025-08）。** DeepSeek-V3.1 的 **UE8M0 FP8**（只含指数、2 的幂的块缩放因子）被描述为"为下一代国产芯片量身定制"——出口管制（export controls）如今也在塑造数值格式。
- **针对专家的 FP4 QAT（2026-04）。** DeepSeek-V4 对路由专家权重进行 FP4 量化感知训练，据报道其稀疏注意力索引器也以 FP4 计算；专家占据了参数内存的大头，因此 4 比特在这里收益最大。
- **后训练与发布中的 4 比特（2025-08 → 2026-07）。** gpt-oss 的 MoE 权重被后训练为 MXFP4（约 4.25 比特/参数）；Kimi K2 Thinking 在后训练中对 MoE 组件使用 INT4 仅权重 QAT（生成速度约 2 倍，所有基准均在 INT4 下报告）；Kimi K3 的权重以 MXFP4 发布。
- **FP16 在 RL 中回归（2025-10）。** BF16 的 7 位尾数使 rollout 引擎与训练器的对数概率产生偏离；以 FP16 进行 RL 基本消除了训练–推断不匹配，无需重要性采样补丁（[arXiv:2510.26788](https://arxiv.org/abs/2510.26788)）。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"训练用 bf16 还是 fp16——选哪个，为什么？"** bf16：与 fp32 同范围的指数 → 稳定，无需损失缩放；fp16：精度更高但需要损失缩放。
- **"为什么要保留 fp32 的主权重副本？"** 小的更新在 16 位下会被舍入吞掉；主权重保留了它们。
- **"什么是损失缩放，什么时候需要？"** 放大损失，使小的 fp16 梯度保持在下溢阈值之上；动态缩放在溢出时回退。
- **"fp8 训练难在哪里？"** 逐张量的动态范围缩放、离群值，以及要把敏感操作保持在更高精度。
- **"为什么 NVFP4 的训练效果优于 MXFP4，FP4 配方还需要什么？"** 16 元素 vs 32 元素的块、E4M3 vs 2 的幂的缩放因子（+ 逐张量 FP32 缩放）带来更精细的范围跟踪；再加上对 Wgrad 的 Hadamard 变换、对梯度的随机舍入、2D 权重缩放，以及让敏感层使用 BF16。
- **"为什么 RL 可能要用 FP16 而不是 BF16？"** 策略梯度的比率要求采样器和训练器在对数概率上一致；更多的尾数位能减少不匹配。

## 开放问题
*训练*精度能降到多低（低于 fp4，或包括注意力和优化器状态在内的所有张量都用 fp4）而不损失质量，仍未解决。有原则的逐张量 vs 逐块缩放、离群值处理、哪些层必须保持高精度以及原因，以及前沿规模下精确的精度缩放定律权衡（例如 FP4 的代价是否随 token 数增长），都是活跃的研究课题；格式（MX/NVFP4/UE8M0）与优化器数值、非 NVIDIA 硬件以及 RL 训练–推断一致性的协同设计也是如此。

## 参考文献
- Micikevicius, P. et al. (2017). *Mixed Precision Training.* arXiv:1710.03740.
- Kalamkar, D. et al. (2019). *A Study of BFLOAT16.* arXiv:1905.12322.
- Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning.* arXiv:2209.05433.
- DeepSeek-AI (2024). *DeepSeek-V3 (FP8 training).* arXiv:2412.19437.
- Kumar, T. et al. (2024). *Scaling Laws for Precision.* arXiv:2411.04330.
- Rouhani, B. et al. (2023). *Microscaling Data Formats for Deep Learning.* arXiv:2310.10537.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
- NVIDIA (2025). *Nemotron 3 (hybrid Mamba-Transformer MoE).* arXiv:2512.20856.
- Sea AI Lab (2025). *Defeating the Training-Inference Mismatch via FP16.* arXiv:2510.26788.
