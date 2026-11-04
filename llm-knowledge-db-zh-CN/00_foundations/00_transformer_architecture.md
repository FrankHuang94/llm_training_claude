# Transformer 架构（Transformer Architecture）

> **最后更新：** 2026-11-04
> **相关文件：** [注意力机制](01_attention_mechanisms.md) · [位置编码](03_positional_encoding.md) · [模型架构变体](05_model_architecture_variants.md)
> **关键论文：** Vaswani et al. 2017 "Attention Is All You Need"（[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)）· Radford et al. 2018/2019（GPT-1/2）· Devlin et al. 2018 BERT（[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)）· Touvron et al. 2023 LLaMA（[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)）· Shoeybi et al. 2019 Megatron-LM（[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)）

## 概述
Transformer（Vaswani et al.，NeurIPS 2017）以自注意力（self-attention）取代循环与卷积，使之成为建模序列依赖的唯一机制。它的决定性特性在于：任意两个位置之间只需 $O(1)$ 次顺序操作即可交互——不同于路径长度随序列线性增长的 RNN——因此可以在序列维度上大规模并行，天然契合 GPU/TPU 硬件。这种并行性再加上良好的缩放特性，是该架构成为几乎所有现代大语言模型（LLM）底座的最重要原因。

原始的编码器-解码器设计衍生出三种范式。**仅编码器**模型（BERT）使用双向注意力和掩码语言建模目标，擅长表示/理解类任务。**仅解码器**模型（GPT 系列）使用因果（自回归）注意力和下一个 token 预测，并已被证明是生成式 LLM 的主导范式，因为下一个 token 预测是一个通用的、可干净扩展的自监督目标。**编码器-解码器**模型（T5、原始 Transformer）在条件生成（翻译、摘要）上依然强劲，但在前沿规模上已基本让位于仅解码器模型。

## 核心概念
**缩放点积注意力。** 给定查询 $Q\in\mathbb{R}^{n\times d_k}$、键 $K\in\mathbb{R}^{n\times d_k}$、值 $V\in\mathbb{R}^{n\times d_v}$：
$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
$\sqrt{d_k}$ 缩放用于抵消点积幅值随维度增长（方差 $\propto d_k$）的问题，否则 softmax 会被推入饱和、梯度很小的区域。$Q,K,V$ 是输入的线性投影：$Q=XW_Q$，依此类推。

**多头注意力（MHA）。** 不是在 $d_{model}$ 上做单个注意力函数，而是运行 $h$ 个头，每个头维度为 $d_k=d_{model}/h$：
$$\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O,\quad \text{head}_i=\text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$
不同的头可以各自专精（句法、位置、归纳（induction））。由于每个头的维度缩小，总计算量与单头相当。

**块结构。** 一个 Transformer 层由两个子层组成——注意力和逐位置前馈网络（FFN）——每个子层都包裹在残差连接与归一化之中：$x \leftarrow x + \text{Sublayer}(\text{Norm}(x))$。FFN 通常为 $\text{FFN}(x)=W_2\,\sigma(W_1 x)$，隐藏维度为 $4d_{model}$；现代模型将 $\sigma$ 替换为 **SwiGLU**（Shazeer 2020），$\text{SwiGLU}(x)=(W_1 x\odot\text{Swish}(W_3 x))$，并把隐藏维度设为约 $\frac{8}{3}d_{model}$ 以保持参数量不变。

**因果掩码。** 仅解码器模型在 softmax 内部加入下三角掩码 $M$（当 $j>i$ 时 $M_{ij}=-\infty$），使位置 $i$ 只能关注 $\le i$ 的位置，从而保持自回归分解 $p(x)=\prod_i p(x_i\mid x_{<i})$。

## 关键挑战
- **注意力的二次方开销。** 注意力的计算量为 $O(n^2 d)$，分数矩阵的内存为 $O(n^2)$——这是长上下文（long context）的核心瓶颈（见[注意力机制](01_attention_mechanisms.md)）。
- **深度/规模下的训练不稳定。** 深层堆叠会出现梯度/激活爆炸；归一化的放置位置至关重要。
- **位置信息。** 自注意力具有置换等变性，因此必须显式注入位置信号（见[位置编码](03_positional_encoding.md)）。
- **推断（inference）时的 KV 缓存内存。** 自回归解码需要缓存键/值，其大小随上下文长度和批大小线性增长。

## 解决方案与当前最佳实践
稠密模型（dense model）的"默认配方"（LLaMA 2/3、Mistral 以及大多数 2023–2025 年的模型）：**仅解码器**、**前置归一化**（pre-norm，在子层之前做 Norm——远比原始的后置归一化稳定）、以 **RMSNorm** 取代 LayerNorm（去掉均值中心化，更便宜且同样有效；Zhang & Sennrich 2019）、**RoPE** 位置编码、**SwiGLU** FFN、用于削减 KV 缓存的 **GQA**，以及线性层不带偏置项。输入嵌入（embedding）与输出投影之间的权重共享在小规模下很常见。Megatron 风格的张量并行把注意力和 FFN 矩阵分片到多个设备上。

到 2026 年末，*开放前沿*的模块又往前走了一步：FFN 是细粒度的**混合专家（MoE）**（数百个专家、top-8 到 top-16 路由、共享专家）；token 混合器常常是**混合式**的（每个全注意力层配约 3 个线性注意力或稀疏注意力层，或者 Mamba-2 层配少量注意力层）；注意力输出可能带有 **sigmoid 输出门控**；稳定性部分交由优化器（optimizer）处理（与 Muon 配合的 **QK-Clip**），而不再只靠归一化；**MTP 头**与主 LM 头一起训练。残差流本身如今也成了设计对象（Hyper-Connections / mHC）。见[模型架构变体](05_model_architecture_variants.md)。

## 各实验室观点
- **OpenAI** 开创了仅解码器的 GPT 路线；GPT-3/4/5/6 的细节均未公开，但 **gpt-oss**（2025 年 8 月）是 GPT-2 之后 OpenAI 首次公开其模块设计：MoE FFN（top-4 路由）、GQA、带状（滑动窗口）与稠密注意力层交替，以及每个头可学习的注意力汇（attention sink）logit。
- **Google DeepMind** 坚持编码器-解码器的时间最长（T5/UL2），并在部分 Gemini 组件中使用；PaLM 推广了注意力与 FFN 并行的层结构以及多查询注意力（MQA）。
- **Meta** 通过 LLaMA 确立了开放的仅解码器配方（RMSNorm+RoPE+SwiGLU），使之成为社区默认；在 Llama 4（MoE）之后，其前沿产品线转向闭源的 Muse 模型（2026 年 4 月），因此 Meta 不再定义开放基线。
- **DeepSeek** 直接在注意力模块本身上创新：多头潜在注意力（MLA）、可学习的稀疏注意力（sparse attention）（V3.2 中的 DSA、V4 中的压缩稀疏注意力）、细粒度 MoE FFN，以及残差流研究（mHC）（见 [DeepSeek 路线图](../08_lab_roadmaps/04_deepseek_roadmap.md)）。
- **Qwen / Moonshot** 推动混合模块：以 3:1 组合 Gated DeltaNet 与门控 softmax 注意力（Qwen3-Next、Qwen3.5），以及以 3:1 组合 Kimi Delta Attention 与 MLA（Kimi Linear、Kimi K3）。
- **Mistral** 为了效率强调滑动窗口注意力和 MoE（Mixtral）。

## 最新进展（2023–2026年11月）
稠密的仅解码器模块正日益混合化：MoE FFN（Mixtral、DeepSeek-V3、Llama 4）把参数量与激活计算量解耦；注意力变体（MLA、GQA）削减 KV 内存；非注意力的 token 混合器（Mamba/状态空间模型（SSM）混合体，见[非 Transformer 架构](../09_emerging_frontiers/04_neuromorphic_and_non_transformer.md)）与注意力层交错排列（如 Jamba、Zamba）。归一化研究（QK-norm、三明治归一化）仍在继续，以稳定超大规模训练。

**2025年中 → 2026年11月更新：**
- **输出门控注意力（2025-05，NeurIPS 2025；arXiv:2505.06708）。** 在 SDPA 输出上施加按头的 sigmoid 门，为原本低秩的 $W_V W_O$ 路径增加了非线性，并带来依赖输入的稀疏性。它消除了注意力汇（首 token 注意力占比从 46.7% 降至 4.8%），能承受更大的学习率，在 RULER 上提升超过 10 分；已在 Qwen3-Next 中落地。与之对照的是 gpt-oss（2025-08），后者转而在 softmax 分母中加入一个显式的可学习汇 logit。
- **3:1 比例的混合 token 混合器（2025-09/10）。** Qwen3-Next-80B-A3B 每个门控注意力层配 3 个 Gated DeltaNet 层（48 层 = 12 × [3 GDN + 1 attn]）；Kimi Linear（arXiv:2510.26692）采用 3 KDA : 1 MLA，并声称这是线性混合架构首次在短上下文、长上下文和 RL 三种设定下的公平比较中胜过全注意力。Kimi K3（2026-07）将其扩展到 69 个 KDA 层 + 24 个 MLA 层。MiniMax M2（2025-10）是反例：在混合架构于多跳推理（reasoning）上落败后，它回到了全注意力。
- **残差流成为设计维度（2025-12；arXiv:2512.24880）。** Hyper-Connections 把残差拓宽为多条相互混合的流，但破坏了恒等映射；DeepSeek 的 **mHC** 将混合矩阵投影到双随机矩阵上（通过 Sinkhorn–Knopp 投影到 Birkhoff 多面体），在 27B 规模恢复了稳定的信号传播。据报道 DeepSeek-V4 使用了它，但存在争议。另据报道，Kimi K3 加入了块级的"Attention Residuals"。
- **借助优化器实现稳定（2025-07；arXiv:2507.20534）。** MuonClip/QK-Clip 在最大注意力 logit 超过阈值时按头重新缩放 $W_Q,W_K$，使一个 1T 参数的 MoE（Kimi K2）在 15.5T token 上训练且零损失尖峰（loss spike）——这是应对 logit 增长的 QK-norm 之外的另一条路。
- **MTP 头已成标配。** DeepSeek-V3 风格的多 token 预测模块（既提供训练信号，又充当投机解码（speculative decoding）的草稿）如今已出现在 Qwen3-Next 和 GLM-4.5/5 中。
- **自回归假设正在受到检验。** LLaDA2.0（2025-12；arXiv:2512.15745）把预训练好的自回归 MoE Transformer 转换为最大 100B 参数的块扩散语言模型——同样的 Transformer 堆叠，只是用分块双向去噪取代了因果掩码。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"推导缩放点积注意力并解释 $\sqrt{d_k}$。"** 需要给出 softmax 饱和/方差论证，以及张量形状和因果掩码。
- **"为什么 LLM 选择仅解码器而不是编码器-解码器？"** 下一个 token 预测的通用性、更简单的缩放、上下文学习、训练/推断之间不存在编码器不对称。
- **"前置与后置归一化、RMSNorm 与 LayerNorm——分别是什么，为什么？"** 梯度流、训练稳定性、计算节省。
- **"逐项讲解一个 Transformer 层的参数量。"** $4d^2$（注意力 QKVO）$+ 2\cdot d\cdot d_{ff}$（FFN）；嵌入为 $Vd$。
- **"为什么在注意力之后加 sigmoid 门能消除注意力汇？"** softmax 必须把概率质量放在某处；没有门控时，想要"什么都不做"的头会把质量倾倒在首个 token 上。输出门可以直接把该头的贡献置零，于是注意力汇（以及相关的巨量激活）就不再需要了。
- **"如果用可学习的混合拓宽残差流，会出什么问题，如何修复？"** 恒等映射丧失 → 信号随深度爆炸/衰减；把混合约束为双随机矩阵（mHC），从而保持范数。

## 开放问题
对于上下文学习、长度泛化等能力而言，注意力是否*必不可少*（相对于 SSM/线性注意力（linear attention））仍存在激烈争论；2025–26 年的证据表明，少数全注意力层就足以应对大多数基准（benchmark），但 MiniMax 关于多跳推理的发现说明，剩余的差距尚无定论。注意力与 FFN 之间的最优参数分配、极端规模下的深度与宽度权衡，以及有原则的归一化放置方式，目前都只是在经验上而非理论上得到确定。

## 参考文献
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. arXiv:1706.03762.
- Devlin, J. et al. (2018). *BERT.* arXiv:1810.04805.
- Touvron, H. et al. (2023). *LLaMA.* arXiv:2302.13971.
- Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
- Xiong, R. et al. (2020). *On Layer Normalization in the Transformer Architecture*（前置归一化）. arXiv:2002.04745.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.* arXiv:2505.06708.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880.
