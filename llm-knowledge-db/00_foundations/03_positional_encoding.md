# Positional Encoding

> **Last Updated:** 2026-06-18
> **Related Files:** [Transformer Architecture](00_transformer_architecture.md) · [Attention Mechanisms](01_attention_mechanisms.md) · [Context Window Extension](../06_efficiency/06_context_window_extension.md)
> **Key Papers:** Vaswani et al. 2017 (sinusoidal) · Su et al. 2021 RoFormer/RoPE ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) · Press et al. 2021 ALiBi ([arXiv:2108.12409](https://arxiv.org/abs/2108.12409)) · Peng et al. 2023 YaRN ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071)) · Ding et al. 2024 LongRoPE ([arXiv:2402.13753](https://arxiv.org/abs/2402.13753))

## Overview
Self-attention is permutation-equivariant: shuffle the input tokens and the output shuffles identically, because the attention operation contains no notion of order. Positional encoding injects sequence-order information so the model can distinguish "dog bites man" from "man bites dog." The design space has evolved from *absolute* encodings added to embeddings toward *relative* and *rotary* schemes that encode the distance between tokens directly in the attention computation — a shift driven largely by the need for **length extrapolation** (using a context longer than seen in training).

The dominant modern choice is **Rotary Position Embedding (RoPE)**, used by LLaMA, Mistral, Qwen, DeepSeek, Gemma, and most 2023–2026 models. Understanding why RoPE won, and how it is stretched for long context (YaRN, LongRoPE), is core foundational knowledge.

## Core Concepts
**Sinusoidal (absolute).** The original Transformer adds fixed sinusoids to token embeddings:
$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d}\big),\quad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d}\big)$$
Geometrically progressing frequencies let the model represent a range of wavelengths. **Learned absolute** embeddings (GPT, BERT) simply learn a vector per position up to a max length — but cannot extrapolate beyond it.

**RoPE (rotary).** Instead of *adding* position, RoPE *rotates* the query and key vectors by an angle proportional to position. Pairs of dimensions are treated as 2D coordinates and rotated by $m\theta_i$ where $\theta_i=10000^{-2i/d}$ and $m$ is the position. The key property: the dot product of a rotated query at position $m$ and rotated key at position $n$ depends only on the **relative** offset $m-n$:
$$\langle R_m q,\; R_n k\rangle = g(q,k,\,m-n)$$
This gives relative-position awareness "for free" inside attention, preserves vector norms, and decays attention with distance — all without extra parameters. RoPE is applied per-layer to Q and K only (not V).

**ALiBi.** Adds a linear, head-specific bias to attention scores: $\text{score}_{ij} += -m\cdot|i-j|$ with per-head slope $m$. No positional embeddings at all; strong zero-shot length extrapolation, popular in some open models (BLOOM, MPT).

**Context extension of RoPE.** Naively running RoPE beyond training length fails (attention collapses). **Position Interpolation** (Chen et al. 2023) linearly rescales positions $m\to m\cdot\frac{L_{train}}{L_{new}}$ so they stay in-range, then fine-tunes briefly. **NTK-aware** scaling adjusts the base $10000$ to spread the interpolation unevenly across frequencies. **YaRN** combines NTK-by-parts interpolation with an attention-temperature correction, extending context (e.g., 4K→128K) with minimal fine-tuning. **LongRoPE** searches non-uniform per-dimension rescaling factors to reach 2M+ tokens.

## Key Challenges
- **Length extrapolation.** Most encodings degrade sharply beyond training length; high-frequency RoPE dimensions are the culprit.
- **High-frequency aliasing.** Short-wavelength rotary dimensions become ambiguous at long range, demanding frequency-dependent treatment.
- **Cost of long-context fine-tuning.** Extension methods still need some continued training at long length, which is expensive.
- **Interaction with attention sinks.** Initial tokens act as attention sinks; positional schemes interact with this in nontrivial ways (StreamingLLM).

## Solutions & Current Best Practices
**RoPE is the default**, with **YaRN or NTK-aware scaling** for context extension and a short long-context fine-tuning phase. Long-context recipes (Llama 3.1 128K, Qwen, Gemini) increase the RoPE base frequency ("theta scaling") and anneal on long documents. ALiBi remains a respected alternative where simplicity and extrapolation matter more than peak quality.

## Lab Perspectives
- **Meta** (LLaMA) and **Mistral** standardized RoPE; Llama 3.1 used RoPE base/theta scaling for 128K.
- **Google DeepMind** pushes the longest contexts (Gemini 1.5: 1–2M tokens), with proprietary positional/architectural methods plus retrieval.
- **Microsoft** produced LongRoPE for extreme extension.
- **DeepSeek** integrates RoPE with MLA via a decoupled "RoPE-carrying" key component.

## Latest Developments (2023–2026)
RoPE base/theta scaling is now routine for long context. **LongRoPE2** and per-dimension search refine 1M+ contexts. There is renewed interest in **NoPE** (no positional encoding) for decoder-only models — causal masking alone can encode position, and NoPE shows surprisingly strong length generalization in some settings. Hybrid SSM/attention models inherit position handling from their attention layers.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why did RoPE displace sinusoidal/learned encodings?"** Relative-position via rotation, norm preservation, no params, better extrapolation, clean long-context extension.
- **"Derive the relative-position property of RoPE."** Show $\langle R_m q, R_n k\rangle$ depends only on $m-n$.
- **"How do you extend a 4K model to 128K?"** Position interpolation / NTK / YaRN + theta scaling + long-context fine-tune.
- **"What is ALiBi and when would you use it?"** Linear distance bias, strong extrapolation, no PE.

## Open Problems
The mechanism of length generalization is not fully understood; why some models extrapolate and others don't lacks a complete theory. Whether explicit positional encoding is even necessary (NoPE) for decoder-only LMs is debated. Achieving true train-short/test-long generalization to millions of tokens without any fine-tuning remains unsolved.

## References
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Press, O. et al. (2021). *Train Short, Test Long: ALiBi.* arXiv:2108.12409.
- Chen, S. et al. (2023). *Extending Context Window via Position Interpolation.* arXiv:2306.15595.
- Peng, B. et al. (2023). *YaRN.* arXiv:2309.00071.
- Ding, Y. et al. (2024). *LongRoPE.* arXiv:2402.13753.
