# Attention Mechanisms

> **Last Updated:** 2026-06-18
> **Related Files:** [Transformer Architecture](00_transformer_architecture.md) · [Context Window Extension](../06_efficiency/06_context_window_extension.md) · [Compute & Memory Efficiency](../02_pretraining/05_compute_and_memory_efficiency.md)
> **Key Papers:** Dao et al. 2022 FlashAttention ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) · Dao 2023 FlashAttention-2 ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) · Shazeer 2019 MQA ([arXiv:1911.02150](https://arxiv.org/abs/1911.02150)) · Ainslie et al. 2023 GQA ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) · Liu et al. 2023 Ring Attention ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889))

## Overview
Self-attention's expressivity comes at a cost: time and memory scale as $O(n^2)$ in sequence length $n$. As context windows grew from 512 (BERT) to 128K–10M tokens (2024–2026), attention became the dominant systems bottleneck — both for the score matrix during training and for the key/value (KV) cache during autoregressive inference. The field's response splits into two complementary lines: **exact, IO-aware kernels** (FlashAttention) that keep the math identical but avoid materializing the $n\times n$ matrix in slow memory, and **architectural approximations** (MQA/GQA, sliding window, linear/sparse attention) that change what is computed to reduce memory or FLOPs.

Understanding these mechanisms is essential because the memory wall — not raw FLOPs — typically determines whether a long-context model is trainable and servable. The KV cache in particular dominates inference memory and latency at scale.

## Core Concepts
**FlashAttention (IO-awareness).** The insight is that standard attention is *memory-bound*, not compute-bound: reading/writing the $n\times n$ scores to GPU HBM dominates wall-clock time. FlashAttention tiles $Q,K,V$ into blocks that fit in fast SRAM and computes attention with **online softmax** — maintaining running max $m$ and normalizer $\ell$ so the output is computed in a single streaming pass without ever storing the full score matrix:
$$\ell^{(j)} = e^{m^{(j-1)}-m^{(j)}}\ell^{(j-1)} + \sum e^{S_{ij}-m^{(j)}}$$
This cuts memory from $O(n^2)$ to $O(n)$ and gives 2–4× speedups. **FlashAttention-2** improves parallelism and work partitioning (better GPU occupancy, fewer non-matmul FLOPs); **FlashAttention-3** exploits Hopper (H100) async/FP8 features for further gains.

**Multi-Query Attention (MQA).** All heads share a single key and value head ($h$ query heads, 1 KV head), shrinking the KV cache by a factor of $h$. This dramatically improves decoding throughput but can degrade quality and destabilize training.

**Grouped-Query Attention (GQA).** Interpolates: $g$ KV heads shared among groups of query heads ($1 < g < h$). GQA recovers nearly all MHA quality at a fraction of the KV cache (LLaMA-2 70B, LLaMA-3, Mistral use it). The KV cache size is $2\cdot g\cdot d_k\cdot n\cdot \text{layers}$, so reducing $g$ from $h$ to e.g. 8 is a large saving.

**Sliding-window attention (SWA).** Each token attends only to the previous $w$ tokens (Mistral uses $w=4096$). Stacking $L$ layers gives an effective receptive field of $L\cdot w$. Reduces per-layer cost to $O(n\cdot w)$.

**Ring Attention.** Distributes the sequence across devices in a ring; each device holds a block of Q and streams K/V blocks around the ring, overlapping communication with computation, enabling near-infinite context limited only by total device memory.

## Key Challenges
- **KV-cache memory at inference.** Grows linearly with context × batch × layers; often the binding constraint for serving long-context models.
- **Quality–efficiency tradeoff.** MQA/SWA save memory but can hurt recall over long ranges; GQA is the pragmatic compromise.
- **Hardware-specific kernels.** FlashAttention must be re-tuned per GPU generation; portability to TPUs/AMD is nontrivial.
- **Long-range information loss.** Sliding window and sparse patterns risk dropping genuinely long-range dependencies.

## Solutions & Current Best Practices
The 2026 default: **FlashAttention-2/3 kernels + GQA** for dense models, with **sliding-window or hybrid global/local** patterns for very long context. Inference stacks (vLLM, TensorRT-LLM) add **PagedAttention** — virtual-memory-style paging of the KV cache to eliminate fragmentation and enable high-throughput batching. Quantizing the KV cache (FP8/INT8) is increasingly standard. DeepSeek's **MLA** compresses K/V into a low-rank latent, achieving GQA-like memory with MHA-like quality.

## Lab Perspectives
- **Google** introduced MQA (Shazeer 2019) and uses aggressive KV reduction for TPU serving; PaLM used MQA.
- **Meta/Mistral** standardized GQA (and Mistral, SWA) in open models.
- **DeepSeek** bet on MLA as a distinct point in the design space — latent KV compression rather than head sharing.
- **Anthropic/OpenAI** kernels are proprietary but long-context Claude/GPT clearly rely on FlashAttention-class kernels plus KV optimizations.

## Latest Developments (2023–2026)
FlashAttention-3 with FP8; **MLA** popularization via DeepSeek-V2/V3; **PagedAttention/prefix caching** as serving standards; hybrid attention-SSM models (Jamba, Zamba) that use full attention only in a few layers; and research into **native sparse attention** (e.g., DeepSeek's NSA, 2025) that is trainable end-to-end rather than applied post-hoc.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does GQA reduce memory, and what does it trade off?"** KV-cache factor of $h/g$; minor quality loss vs MQA's larger loss.
- **"Explain how FlashAttention avoids the $O(n^2)$ memory."** Online softmax + tiling in SRAM; it's IO-aware, exact, memory-bound reasoning.
- **"How big is the KV cache for a given model/context?"** Compute $2\cdot \text{layers}\cdot g\cdot d_k\cdot n\cdot \text{bytes}$ — show you can size it.
- **"How would you serve a 1M-token context model?"** Ring/sequence parallelism, paged + quantized KV, prefix caching.

## Open Problems
Whether sub-quadratic attention (linear attention, SSMs) can fully match softmax attention on retrieval/in-context tasks remains open. Optimal hybrid ratios of global vs local attention, and trainable sparsity that matches dense quality, are active research. KV-cache compression without recall degradation is not solved.

## References
- Dao, T. et al. (2022). *FlashAttention.* arXiv:2205.14135.
- Dao, T. (2023). *FlashAttention-2.* arXiv:2307.08691.
- Shah, J. et al. (2024). *FlashAttention-3.* arXiv:2407.08608.
- Ainslie, J. et al. (2023). *GQA.* arXiv:2305.13245.
- Liu, H. et al. (2023). *Ring Attention.* arXiv:2310.01889.
- Kwon, W. et al. (2023). *PagedAttention / vLLM.* arXiv:2309.06180.
