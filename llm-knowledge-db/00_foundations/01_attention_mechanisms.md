# Attention Mechanisms

> **Last Updated:** 2026-11-04
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

**Linear attention and the delta rule.** Dropping the softmax lets attention be written as a recurrence over a matrix-valued state $S_t\in\mathbb{R}^{d_v\times d_k}$ with $O(1)$ memory per step. Additive linear attention ($S_t=S_{t-1}+v_tk_t^\top$) only accumulates and its recall degrades. **Gated DeltaNet** instead overwrites:
$$S_t=\alpha_t\,S_{t-1}\big(I-\beta_t k_tk_t^\top\big)+\beta_t\,v_tk_t^\top,\qquad o_t=S_tq_t$$
where $\alpha_t$ is a decay gate and $\beta_t$ a write strength — an online regression step that replaces the value stored under key $k_t$, keeping the state bounded. Kimi Delta Attention (KDA) makes the decay channel-wise rather than head-wise.

**Learned top-$k$ sparse attention.** A cheap scorer selects which past tokens (or KV blocks) each query attends to; the main attention runs only over the selected set. DeepSeek Sparse Attention (DSA) uses a few-head FP8 ReLU "lightning indexer" — still $O(L^2)$ but with a tiny constant — and runs MLA over the top-$k$ tokens.

## Key Challenges
- **KV-cache memory at inference.** Grows linearly with context × batch × layers; often the binding constraint for serving long-context models.
- **Quality–efficiency tradeoff.** MQA/SWA save memory but can hurt recall over long ranges; GQA is the pragmatic compromise.
- **Hardware-specific kernels.** FlashAttention must be re-tuned per GPU generation; portability to TPUs/AMD is nontrivial.
- **Long-range information loss.** Sliding window and sparse patterns risk dropping genuinely long-range dependencies.

## Solutions & Current Best Practices
The dense-model default remains **FlashAttention-2/3 kernels + GQA**, with **sliding-window or hybrid global/local** patterns for long context. For 1M-token open flagships in 2026 the choice is between (a) **~3:1 linear:full hybrids** (Gated DeltaNet/KDA + gated attention or MLA) and (b) **learned top-$k$ sparse attention over (compressed) KV** (DSA → CSA/HCA, MiniMax MSA), typically combined with MLA. Inference stacks (vLLM, TensorRT-LLM) add **PagedAttention** — virtual-memory-style paging of the KV cache to eliminate fragmentation and enable high-throughput batching. Quantizing the KV cache (FP8/INT8) is increasingly standard. DeepSeek's **MLA** compresses K/V into a low-rank latent, achieving GQA-like memory with MHA-like quality.

## Lab Perspectives
- **Google** introduced MQA (Shazeer 2019) and uses aggressive KV reduction for TPU serving; PaLM used MQA.
- **Meta/Mistral** standardized GQA (and Mistral, SWA) in open models.
- **DeepSeek** bet on MLA as a distinct point in the design space — latent KV compression rather than head sharing — then layered learned sparsity on top: NSA (research) → DSA (V3.2) → CSA + HCA (V4, 1M default context).
- **Moonshot/Qwen** bet on delta-rule hybrids (KDA + MLA in Kimi Linear/K3; Gated DeltaNet + gated attention in Qwen3-Next/3.5); Moonshot also serves long context with MoBA block sparsity.
- **MiniMax** went lightning-attention hybrid (M1) → full attention (M2, citing multi-hop losses and low-precision state sensitivity) → block-sparse MSA (M3).
- **Anthropic/OpenAI** kernels are proprietary but long-context Claude/GPT clearly rely on FlashAttention-class kernels plus KV optimizations; both now ship ~1M-token windows (Opus 4.6 1M beta, 76% on MRCR v2 8-needle at 1M; GPT-5.4/GPT-6 Astra ~1.05M). OpenAI's open gpt-oss alternates 128-token banded and dense layers with GQA (64 Q / 8 KV heads) and learned sink logits.

## Latest Developments (2023–Nov 2026)
FlashAttention-3 with FP8; **MLA** popularization via DeepSeek-V2/V3; **PagedAttention/prefix caching** as serving standards; hybrid attention-SSM models (Jamba, Zamba) that use full attention only in a few layers; and research into **native sparse attention** (e.g., DeepSeek's NSA, 2025) that is trainable end-to-end rather than applied post-hoc.

**Mid-2025 → Nov 2026 update:**
- **Trainable sparsity reached production (2025-02 → 2025-12).** NSA (arXiv:2502.11089, ACL 2025 best paper) combines compressed, selected and sliding branches; MoBA (arXiv:2502.13189) applies MoE-style top-$k$ gating over KV blocks. **DSA** in DeepSeek-V3.2-Exp/V3.2 (arXiv:2512.02556) is trained in two stages — a dense warm-up aligning the indexer to the main attention distribution with a KL loss, then sparse continued training — and matched V3.1-Terminus quality while API prices fell >50%. GLM-5 (2026-02; arXiv:2602.15763) adopted DSA at 744B parameters.
- **Compressed sparse attention (2026-04).** DeepSeek-V4 compresses each $m$-token KV block into one entry (CSA), lets a lightning indexer pick the top-$k$ compressed entries, and adds dense attention over a far more heavily compressed KV (HCA). At 1M context V4-Pro reportedly needs ~27% of V3.2's per-token FLOPs and ~10% of its KV cache; sparse attention is reportedly switched on only at a 64K stage after dense training (DeepSeek-V4 tech report: arXiv:2606.19348, as listed on its Hugging Face card).
- **Delta-rule hybrids (2025-09/10).** Qwen3-Next (3 Gated DeltaNet : 1 gated attention) reports >10× prefill throughput over Qwen3-32B beyond 32K; Kimi Linear (arXiv:2510.26692) reports up to 75% less KV cache and up to 6× decode throughput at 1M. The counter-evidence: MiniMax M2 dropped linear/SWA variants because they hurt multi-hop reasoning and were fragile under low-precision state storage.
- **Attention sinks are now engineered.** Output gating (arXiv:2505.06708) removes the first-token sink (46.7% → 4.8% mass) and improves RULER by >10 points; gpt-oss instead learns an explicit per-head sink logit in the softmax denominator. Activation spikes in GDN hybrids cluster just before full-attention layers (arXiv:2608.12149, single source).
- **Linearizing pretrained models.** STILL (arXiv:2602.02180) routes salient tokens to softmax and summarizes the rest linearly with ~0.04B training tokens; MiniCPM-SALA (arXiv:2602.11761, secondary) converts a Transformer into a 1:3 sparse:linear hybrid at ~25% of from-scratch cost.
- **Serving consequences.** DSA-style indexers add an "indexer K cache" that complicates prefill/decode KV transfer; KV transform coding (PCA + adaptive quantization + entropy coding, ICLR 2026) reports up to 20× compression; attention-FFN disaggregation (arXiv:2605.28302) shows models with cheap attention can give ≥90% of GPUs to FFN experts.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does GQA reduce memory, and what does it trade off?"** KV-cache factor of $h/g$; minor quality loss vs MQA's larger loss.
- **"Explain how FlashAttention avoids the $O(n^2)$ memory."** Online softmax + tiling in SRAM; it's IO-aware, exact, memory-bound reasoning.
- **"How big is the KV cache for a given model/context?"** Compute $2\cdot \text{layers}\cdot g\cdot d_k\cdot n\cdot \text{bytes}$ — show you can size it.
- **"How would you serve a 1M-token context model?"** Ring/sequence parallelism, paged + quantized KV, prefix caching — and, in 2026, an architecture whose KV is sub-linear in practice (3:1 linear hybrid or compressed top-$k$ sparse attention).
- **"Derive the gated DeltaNet update and say why it beats additive linear attention."** $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$: it overwrites the slot for $k_t$ instead of accumulating, so the state stays bounded and recall improves.
- **"DSA vs a 3:1 delta-rule hybrid — which would you pick?"** DSA keeps exact softmax over a selected subset (good retrieval, but an $O(L^2)$ indexer and an extra cache); hybrids give $O(1)$-state layers and tiny KV but risk multi-hop/recall gaps (MiniMax's argument).

## Open Problems
Whether sub-quadratic attention (linear attention, SSMs) can fully match softmax attention on retrieval/in-context tasks remains open — 2025–26 hybrids match benchmarks but MiniMax's multi-hop results and the precision sensitivity of recurrent state keep the question alive. Optimal hybrid ratios of global vs local attention, and trainable sparsity that matches dense quality, are active research. KV-cache compression without recall degradation is not solved.

## References
- Dao, T. et al. (2022). *FlashAttention.* arXiv:2205.14135.
- Dao, T. (2023). *FlashAttention-2.* arXiv:2307.08691.
- Shah, J. et al. (2024). *FlashAttention-3.* arXiv:2407.08608.
- Ainslie, J. et al. (2023). *GQA.* arXiv:2305.13245.
- Liu, H. et al. (2023). *Ring Attention.* arXiv:2310.01889.
- Kwon, W. et al. (2023). *PagedAttention / vLLM.* arXiv:2309.06180.
- Yang, S. et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464.
- Yuan, J. et al. (2025). *Native Sparse Attention.* arXiv:2502.11089.
- Lu, E. et al. (2025). *MoBA: Mixture of Block Attention for Long-Context LLMs.* arXiv:2502.13189.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *DeepSeek-V3.2* (DeepSeek Sparse Attention). arXiv:2512.02556.
