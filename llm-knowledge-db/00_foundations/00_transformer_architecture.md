# Transformer Architecture

> **Last Updated:** 2026-06-18
> **Related Files:** [Attention Mechanisms](01_attention_mechanisms.md) · [Positional Encoding](03_positional_encoding.md) · [Model Architecture Variants](05_model_architecture_variants.md)
> **Key Papers:** Vaswani et al. 2017 "Attention Is All You Need" ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) · Radford et al. 2018/2019 (GPT-1/2) · Devlin et al. 2018 BERT ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805)) · Touvron et al. 2023 LLaMA ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) · Shoeybi et al. 2019 Megatron-LM ([arXiv:1909.08053](https://arxiv.org/abs/1909.08053))

## Overview
The Transformer (Vaswani et al., NeurIPS 2017) replaced recurrence and convolution with self-attention as the sole mechanism for modeling sequence dependencies. Its decisive property is that any two positions interact in $O(1)$ sequential operations — unlike RNNs whose path length grows linearly — making it massively parallelizable across the sequence dimension and therefore ideally suited to GPU/TPU hardware. This parallelism, combined with favorable scaling behavior, is the single most important reason the architecture became the substrate of essentially all modern LLMs.

Three paradigms emerged from the original encoder-decoder design. **Encoder-only** models (BERT) use bidirectional attention and masked-language-modeling objectives, excelling at representation/understanding tasks. **Decoder-only** models (GPT family) use causal (autoregressive) attention and next-token prediction, and have proven to be the dominant paradigm for generative LLMs because next-token prediction is a universal, self-supervised objective that scales cleanly. **Encoder-decoder** models (T5, original Transformer) remain strong for conditional generation (translation, summarization) but have largely ceded ground to decoder-only models at frontier scale.

## Core Concepts
**Scaled dot-product attention.** Given queries $Q\in\mathbb{R}^{n\times d_k}$, keys $K\in\mathbb{R}^{n\times d_k}$, values $V\in\mathbb{R}^{n\times d_v}$:
$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
The $\sqrt{d_k}$ scaling counteracts the growth of dot-product magnitude with dimension (variance $\propto d_k$), which would otherwise push softmax into saturated, low-gradient regions. $Q,K,V$ are linear projections of the input: $Q=XW_Q$, etc.

**Multi-head attention (MHA).** Rather than a single attention function over $d_{model}$, we run $h$ heads each of dimension $d_k=d_{model}/h$:
$$\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O,\quad \text{head}_i=\text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$
Different heads can specialize (syntactic, positional, induction). Total compute is comparable to single-head because per-head dimension shrinks.

**Block structure.** A Transformer layer is two sublayers — attention and a position-wise feed-forward network (FFN) — each wrapped in a residual connection and normalization: $x \leftarrow x + \text{Sublayer}(\text{Norm}(x))$. The FFN is typically $\text{FFN}(x)=W_2\,\sigma(W_1 x)$ with hidden dim $4d_{model}$; modern models replace $\sigma$ with **SwiGLU** (Shazeer 2020), $\text{SwiGLU}(x)=(W_1 x\odot\text{Swish}(W_3 x))$, using ~$\frac{8}{3}d_{model}$ hidden to match parameter count.

**Causal masking.** Decoder-only models add a lower-triangular mask $M$ ($M_{ij}=-\infty$ for $j>i$) inside the softmax so position $i$ attends only to $\le i$, preserving the autoregressive factorization $p(x)=\prod_i p(x_i\mid x_{<i})$.

## Key Challenges
- **Quadratic attention cost.** Attention is $O(n^2 d)$ in compute and $O(n^2)$ in memory for the score matrix — the central bottleneck for long context (see [Attention Mechanisms](01_attention_mechanisms.md)).
- **Training instability at depth/scale.** Deep stacks suffer gradient/activation blow-ups; placement of normalization is critical.
- **Position information.** Self-attention is permutation-equivariant, so positional signal must be injected explicitly (see [Positional Encoding](03_positional_encoding.md)).
- **KV-cache memory at inference.** Autoregressive decoding caches keys/values, growing linearly with context and batch.

## Solutions & Current Best Practices
The frontier "default recipe" (LLaMA 2/3, Mistral, most 2023–2026 models): **decoder-only**, **pre-norm** (Norm before sublayer — far more stable than the original post-norm), **RMSNorm** instead of LayerNorm (drops the mean-centering, cheaper, equally effective; Zhang & Sennrich 2019), **RoPE** positions, **SwiGLU** FFN, **GQA** for KV-cache reduction, and no bias terms in linear layers. Weight tying between input embedding and output projection is common at small scale. Megatron-style tensor parallelism shards the attention and FFN matrices across devices.

## Lab Perspectives
- **OpenAI** pioneered the decoder-only GPT line; GPT-3/4 details are closed but follow the dense decoder-only paradigm (with MoE strongly suspected in GPT-4).
- **Google DeepMind** retained encoder-decoder longest (T5/UL2) and uses it in some Gemini components; PaLM popularized parallel attention+FFN layers and multi-query attention.
- **Meta** standardized the open decoder-only recipe (RMSNorm+RoPE+SwiGLU) with LLaMA, making it the community default.
- **DeepSeek** innovates on the attention block itself with Multi-head Latent Attention (MLA) and fine-grained MoE FFNs (see [DeepSeek roadmap](../08_lab_roadmaps/04_deepseek_roadmap.md)).
- **Mistral** emphasizes sliding-window attention and MoE (Mixtral) for efficiency.

## Latest Developments (2023–2026)
The dense decoder-only block is increasingly hybridized: MoE FFNs (Mixtral, DeepSeek-V3, Llama 4) decouple parameters from active compute; attention variants (MLA, GQA) cut KV memory; and non-attention token mixers (Mamba/SSM hybrids, see [Non-Transformer](../09_emerging_frontiers/04_neuromorphic_and_non_transformer.md)) interleave with attention layers (e.g., Jamba, Zamba). normalization research continues (QK-norm, sandwich norm) to stabilize very large runs.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Derive scaled dot-product attention and explain the $\sqrt{d_k}$."** Expect the softmax-saturation/variance argument, plus shapes and the causal mask.
- **"Why decoder-only over encoder-decoder for LLMs?"** Universality of next-token prediction, simpler scaling, in-context learning, no train/inference encoder asymmetry.
- **"Pre-norm vs post-norm, RMSNorm vs LayerNorm — what and why?"** Gradient flow, training stability, compute savings.
- **"Walk through the parameter count of a Transformer layer."** $4d^2$ (attention QKVO) $+ 2\cdot d\cdot d_{ff}$ (FFN); embeddings $Vd$.

## Open Problems
Whether attention is *necessary* (vs SSMs/linear-attention) for capabilities like in-context learning and length generalization is actively contested. The optimal allocation of parameters between attention and FFN, depth vs width tradeoffs at extreme scale, and principled normalization placement all remain empirically rather than theoretically settled.

## References
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. arXiv:1706.03762.
- Devlin, J. et al. (2018). *BERT.* arXiv:1810.04805.
- Touvron, H. et al. (2023). *LLaMA.* arXiv:2302.13971.
- Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
- Xiong, R. et al. (2020). *On Layer Normalization in the Transformer Architecture* (pre-norm). arXiv:2002.04745.
