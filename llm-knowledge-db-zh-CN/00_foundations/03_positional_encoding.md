# 位置编码（Positional Encoding）

> **最后更新：** 2026-11-04
> **相关文件：** [Transformer 架构](00_transformer_architecture.md) · [注意力机制](01_attention_mechanisms.md) · [上下文窗口扩展](../06_efficiency/06_context_window_extension.md)
> **关键论文：** Vaswani et al. 2017（正弦编码）· Su et al. 2021 RoFormer/RoPE（[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)）· Press et al. 2021 ALiBi（[arXiv:2108.12409](https://arxiv.org/abs/2108.12409)）· Peng et al. 2023 YaRN（[arXiv:2309.00071](https://arxiv.org/abs/2309.00071)）· Ding et al. 2024 LongRoPE（[arXiv:2402.13753](https://arxiv.org/abs/2402.13753)）

## 概述
自注意力（self-attention）具有置换等变性：打乱输入 token 的顺序，输出也会以相同方式被打乱，因为注意力运算本身不包含任何顺序概念。位置编码（positional encoding）注入序列顺序信息，使模型能够区分"狗咬人"和"人咬狗"。其设计空间已从加在嵌入（embedding）上的*绝对*编码，演进为直接在注意力计算中编码 token 间距离的*相对*和*旋转*方案——这一转变主要由**长度外推**（使用比训练时更长的上下文）的需求推动。

当今主流选择是**旋转位置嵌入（RoPE）**，被 LLaMA、Mistral、Qwen、DeepSeek、Gemma 以及大多数 2023–2026 年的模型采用——不过，2025–26 年的混合架构越来越多地让循环层（线性注意力/状态空间模型（SSM））承载顺序信息，而让部分注意力层不带位置编码（NoPE）。理解 RoPE 为何胜出，以及它如何被拉伸到长上下文（long context）（YaRN、LongRoPE），是核心的基础知识。

## 核心概念
**正弦（绝对）编码。** 原始 Transformer 在 token 嵌入上加上固定的正弦信号：
$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d}\big),\quad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d}\big)$$
按几何级数递增的频率使模型能表示一系列波长。**可学习的绝对**嵌入（GPT、BERT）只是为每个位置学习一个向量，直到某个最大长度——但无法外推到更长。

**RoPE（旋转）。** RoPE 不是*加上*位置，而是把查询和键向量*旋转*一个与位置成正比的角度。维度两两成对，被视为二维坐标并旋转 $m\theta_i$，其中 $\theta_i=10000^{-2i/d}$，$m$ 为位置。关键性质：位置 $m$ 处旋转后的查询与位置 $n$ 处旋转后的键，其点积只依赖于**相对**偏移 $m-n$：
$$\langle R_m q,\; R_n k\rangle = g(q,k,\,m-n)$$
这在注意力内部"免费"提供了相对位置感知，保持向量范数，并使注意力随距离衰减——而且不需要额外参数。RoPE 在每一层只作用于 Q 和 K（不作用于 V）。

**ALiBi。** 在注意力分数上加一个线性的、按头区分的偏置：$\text{score}_{ij} += -m\cdot|i-j|$，每个头有自己的斜率 $m$。完全不使用位置嵌入；零样本长度外推能力强，在部分开放模型（BLOOM、MPT）中很流行。

**RoPE 的上下文扩展。** 直接把 RoPE 用到超出训练长度会失败（注意力崩溃）。**位置插值**（Chen et al. 2023）把位置线性缩放 $m\to m\cdot\frac{L_{train}}{L_{new}}$，使其留在训练范围内，然后做简短微调（fine-tuning）。**NTK-aware** 缩放调整基数 $10000$，使插值在各频率上非均匀分布。**YaRN** 把分段 NTK 插值与注意力温度修正结合起来，只需极少量微调就能扩展上下文（例如 4K→128K）。**LongRoPE** 搜索非均匀的逐维缩放因子，以达到 2M+ token。

## 关键挑战
- **长度外推。** 大多数编码在超出训练长度后急剧退化；罪魁祸首是 RoPE 的高频维度。
- **高频混叠。** 短波长的旋转维度在长距离上变得模糊，需要依频率区别处理。
- **长上下文微调的成本。** 扩展方法仍需要在长序列上做一定的继续训练，代价高昂。
- **与注意力汇的相互作用。** 起始 token 会充当注意力汇（attention sink）；位置方案与之存在复杂的相互作用（StreamingLLM）。

## 解决方案与当前最佳实践
**RoPE 是默认选择**，用 **YaRN 或 NTK-aware 缩放**做上下文扩展，并配合一个简短的长上下文微调阶段。长上下文配方（Llama 3.1 128K、Qwen、Gemini）会提高 RoPE 基频（"theta 缩放"），并在长文档上退火。在 2026 年的开放旗舰模型中，扩展是分阶段进行的：先有约 256K 的原生窗口（Qwen3-Next、Qwen3.8 为 262,144），再向约 1M 扩展；或采用多阶段长度课程学习（curriculum）（据报道 DeepSeek-V4 先在 4K→16K 上做稠密训练，在 64K 时开启稀疏注意力，最终达到 1M）。在简洁性和外推比峰值质量更重要的场合，ALiBi 仍是一个受尊重的替代方案。

## 各实验室观点
- **Meta**（LLaMA）和 **Mistral** 把 RoPE 确立为标准；Llama 3.1 通过 RoPE 基数/theta 缩放达到 128K。
- **Google DeepMind** 率先在生产中提供 1–2M 窗口（Gemini 1.5），采用专有的位置/架构方法外加检索；到 2026 年，约 1M 已是基本配置而非差异化卖点（Claude Opus 4.6 的 1M beta、GPT-5.4/GPT-6 Astra 约 1.05M、DeepSeek-V4、Kimi K3、MiniMax M3、Nemotron 3，据报道 Meta Muse Wonder 也是 1M）。
- **Microsoft** 提出了用于极端扩展的 LongRoPE。
- **DeepSeek** 通过一个解耦的"携带 RoPE"的键分量把 RoPE 与 MLA 结合起来。
- **Moonshot** 据报道在 Kimi K3 的 24 个 MLA 层中使用 NoPE，依靠交错其间的 69 个 KDA（delta 规则）层来提供顺序信息。

## 最新进展（2023–2026年11月）
RoPE 基数/theta 缩放如今已是长上下文的常规操作。**LongRoPE2** 和逐维搜索进一步打磨 1M+ 的上下文。**NoPE**（不使用位置编码）在仅解码器模型中重新受到关注——仅靠因果掩码就能编码位置，而 NoPE 在某些设定下表现出出人意料的强长度泛化。SSM/注意力混合模型的位置处理沿用其注意力层的做法。

**2025年中 → 2026年11月更新：**
- **以循环作为位置信号（2025-09 → 2026-07）。** 在 3:1 混合模型（Qwen3-Next、Kimi Linear arXiv:2510.26692、Kimi K3）中，Gated DeltaNet/KDA 层是带有可学习衰减 $\alpha_t$ 的因果循环，它以类似 ALiBi 线性偏置的方式编码"新近性"——但依赖于数据。这使得稀疏的全注意力层使用 NoPE 成为可行（K3 的 MLA 层据报道即如此），从而让这些层彻底绕开 RoPE 外推问题。
- **滑动窗口限定了 RoPE 问题的范围。** gpt-oss（2025-08）在 128K 上下文下让 128 token 的带状层与稠密层交替；据报道 MiMo-V2.5 在 1M 上下文下以 5:1 混合 SWA 层与全局层，窗口为 128 token。只有全局层会看到长程位置，因此只有它们需要扩展。
- **注意力汇、门控与长上下文（2025-05；arXiv:2505.06708）。** StreamingLLM 中的注意力汇部分是一种位置现象（首 token 吸收了概率质量）；sigmoid 输出门消除了它（首 token 质量从 46.7% 降至 4.8%），并使 RULER 提升超过 10 分——位置鲁棒性是通过改变注意力而非编码本身获得的。
- **有效长度与名义长度。** Opus 4.6（2026-02）报告在 1M 长度的 MRCR v2 8-needle 上达到 76%，而 Sonnet 4.5 为 18.5%：窗口大小已经商品化，真正有区分度的指标是满长度下的多针检索。
- **以测试时训练替代外推（2025-12；arXiv:2512.23675）。** TTT-E2E 使用一个滑动窗口 Transformer，在测试时对上下文执行下一个 token 的梯度步（初始化经过元学习，只更新部分 MLP）；在 3B 规模下，它随上下文的扩展表现与全注意力相当，每 token 延迟恒定，在 128K 时快 2.7 倍。ATLAS（arXiv:2505.23735）报告在 10M token 的 BABILong 上提升 80%。窗口之外的位置信息存在于权重中，而非编码中。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么 RoPE 取代了正弦/可学习编码？"** 通过旋转获得相对位置、保持范数、无参数、更好的外推、干净的长上下文扩展。
- **"推导 RoPE 的相对位置性质。"** 证明 $\langle R_m q, R_n k\rangle$ 只依赖于 $m-n$。
- **"如何把一个 4K 模型扩展到 128K？"** 位置插值 / NTK / YaRN + theta 缩放 + 长上下文微调。
- **"ALiBi 是什么，什么时候用？"** 线性距离偏置、强外推、无位置编码。
- **"为什么 3:1 线性/全注意力混合模型可以在全注意力层中去掉 RoPE？"** 循环层的衰减已经编码了顺序和新近性；不带位置的全局层则负责基于内容的检索，并避免在 1M 长度下出现分布外的旋转角度。

## 开放问题
长度泛化的机制尚未被完全理解；为什么有些模型能外推而另一些不能，缺乏完整的理论。对于仅解码器语言模型，显式位置编码是否必要（NoPE）仍有争论。在不做任何微调的情况下真正实现"短训练、长测试"并泛化到数百万 token 仍未解决；混合 NoPE 设计或测试时训练（TTT）的记忆是否比 RoPE 缩放泛化得更好，也尚未在前沿规模上做过正面对比。

## 参考文献
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Press, O. et al. (2021). *Train Short, Test Long: ALiBi.* arXiv:2108.12409.
- Chen, S. et al. (2023). *Extending Context Window via Position Interpolation.* arXiv:2306.15595.
- Peng, B. et al. (2023). *YaRN.* arXiv:2309.00071.
- Ding, Y. et al. (2024). *LongRoPE.* arXiv:2402.13753.
- Kazemnejad, A. et al. (2023). *The Impact of Positional Encoding on Length Generalization in Transformers*（NoPE）. arXiv:2305.19466.
- Xiao, G. et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
