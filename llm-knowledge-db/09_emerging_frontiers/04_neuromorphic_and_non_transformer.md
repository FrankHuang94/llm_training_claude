# Non-Transformer Architectures

> **Last Updated:** 2026-06-18
> **Related Files:** [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Test-Time Training](03_test_time_training.md) · [Long Context & Memory](00_long_context_and_memory.md)
> **Key Papers:** Gu & Dao 2023 Mamba ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) · Dao & Gu 2024 Mamba-2 / SSD ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)) · Peng et al. 2023 RWKV ([arXiv:2305.13048](https://arxiv.org/abs/2305.13048)) · Sun et al. 2023 RetNet ([arXiv:2307.08621](https://arxiv.org/abs/2307.08621))

## Overview
The Transformer's $O(n^2)$ attention and large KV cache motivate a search for **sub-quadratic** sequence models that match its quality. The leading challengers are **state-space models (SSMs)** — especially **Mamba** — along with **linear attention** variants (RWKV, RetNet, Hyena). These aim for the "holy grail": **Transformer-quality with RNN-like linear-time inference and constant memory** (no growing KV cache). After years of linear-attention approaches underperforming, Mamba (2023) reignited the field by closing much of the quality gap, and 2024–2026 saw **hybrid** attention-SSM models reach production. Understanding the SSM formulation, the attention-vs-SSM tradeoffs, and why hybrids dominate is increasingly expected knowledge.

## Core Concepts
**State-space models (SSMs).** A continuous linear system $h'(t)=Ah(t)+Bx(t)$, $y(t)=Ch(t)$, discretized to a recurrence $h_t=\bar A h_{t-1}+\bar B x_t$, $y_t=\bar C h_t$. Key duality: it can be computed as a **recurrence** (linear-time, constant-memory inference, like an RNN) **or** as a **convolution** (parallelizable training). Earlier SSMs (**S4**) used clever structured/HiPPO initialization to capture long-range dependencies.

**Mamba (selective SSM).** The breakthrough: make the SSM parameters $(\bar B, \bar C, \Delta)$ **input-dependent** ("selective"), so the model can *choose* what to remember or forget based on content — overcoming the key weakness of linear-time-invariant SSMs (they couldn't do content-based reasoning like attention). This costs the convolution view, so Mamba uses a **hardware-aware parallel scan** for efficient training. Mamba matched Transformers at small-mid scale with linear-time, constant-memory inference.

**Mamba-2 / SSD.** Establishes a theoretical bridge ("**state-space duality**") showing SSMs and (a form of) attention are closely related, enabling faster, more hardware-friendly implementations and larger state sizes.

**Linear attention family.**
- **RWKV:** an RNN/Transformer hybrid with linear attention reformulated for parallel training and recurrent inference; the "RNN that scales," with RWKV-7 adding expressive, dynamic state.
- **RetNet:** "retention" mechanism with parallel (train), recurrent (infer), and chunkwise forms.
- **Hyena:** replaces attention with long implicit convolutions + gating (sub-quadratic).
- All share the goal of $O(n)$ inference; recent variants (**DeltaNet, Gated DeltaNet**) add delta-rule/gated updates for stronger associative recall, overlapping with TTT (see [Test-Time Training](03_test_time_training.md)).

**The fundamental tradeoff.** Attention keeps *all* tokens (perfect recall, quadratic cost); SSMs compress history into a *fixed-size state* (linear cost, lossy recall). SSMs excel at long sequences and throughput but are **weaker at precise recall / copying / in-context retrieval** — exactly what attention does best.

## Key Challenges
- **Recall / copying weakness.** Fixed-state models struggle with tasks needing exact retrieval from long context (associative recall, needle tests) — attention's strength.
- **In-context learning gap.** Some ICL/induction-head behaviors are harder for pure SSMs.
- **Scaling validation.** Most decisive comparisons are at small/mid scale; frontier-scale parity is less established (and labs keep using attention or hybrids).
- **Ecosystem maturity.** Decades of attention optimization (FlashAttention, kernels, tooling) advantage Transformers; SSM kernels/tooling are newer.
- **Hybrid design.** Choosing the attention:SSM ratio and placement is empirical.

## Solutions & Current Best Practices
**Hybrids win in practice.** Interleave a *few* full-attention layers (for recall/ICL) with many SSM/linear layers (for efficiency) — e.g., **Jamba** (Mamba-Transformer-MoE), **Zamba**, **Nemotron-H**, **Samba**. This captures attention's recall where it matters while getting SSM throughput/long-context efficiency elsewhere. For pure long-sequence throughput with modest recall needs, Mamba-2/RWKV-7 are strong. Most frontier *flagship* models remain attention-based or attention-heavy hybrids; pure SSMs are gaining in efficiency-critical and edge/long-context niches.

## Lab Perspectives
- **Academia (CMU/Princeton — Gu & Dao):** Mamba/Mamba-2, the leading SSM line.
- **AI21:** **Jamba** — first production-scale hybrid SSM-Transformer-MoE.
- **NVIDIA:** Nemotron-H and hybrid research; efficient long-context.
- **RWKV/open community:** RWKV (linear-attention RNN), broad open ecosystem and edge focus.
- **Frontier labs (OpenAI/Anthropic/Google/Meta):** predominantly attention/hybrid; watch and selectively adopt SSM ideas for efficiency (e.g., some Gemini/Llama research on hybrids).

## Latest Developments (2023–2026)
**Mamba (2023)** revived the field; **Mamba-2/SSD** unified SSMs and attention theoretically. **Hybrids** (Jamba, Zamba, Nemotron-H, Samba) became the pragmatic sweet spot and reached production scale. **Expressive recurrent state** (Gated DeltaNet, RWKV-7, Titans) blurred SSMs with **test-time learning/fast weights**. The consensus crystallized: **hybrids, not pure replacements** — attention for recall, SSMs for efficiency — though the optimal mix and whether pure SSMs can ever fully match attention at the frontier remain open.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What problem do SSMs/Mamba solve vs Transformers?"** Sub-quadratic, constant-memory inference (no growing KV cache) for long sequences.
- **"What made Mamba work where earlier SSMs didn't?"** Input-dependent ('selective') parameters enable content-based remembering/forgetting; hardware-aware parallel scan.
- **"What's the fundamental SSM-vs-attention tradeoff?"** Fixed-state compression (linear, lossy recall) vs keep-all-tokens (quadratic, perfect recall).
- **"Why do hybrids dominate?"** A few attention layers restore recall/ICL; SSM layers give efficiency.
- **"Where do SSMs struggle?"** Exact copying/associative recall/needle-in-haystack.

## Open Problems
Whether a **pure** sub-quadratic architecture can match attention on **recall/ICL at frontier scale** is unresolved. Optimal **hybrid ratios**, closing the recall gap, mature kernels/tooling, and the deeper theory linking SSMs, attention, fast weights, and test-time learning are active. The field's likely near-term answer is *efficient hybrids*, but a clean Transformer successor remains an open prize.

## References
- Gu, A., Dao, T. (2023). *Mamba.* arXiv:2312.00752.
- Dao, T., Gu, A. (2024). *Mamba-2 / State Space Duality.* arXiv:2405.21060.
- Peng, B. et al. (2023). *RWKV.* arXiv:2305.13048.
- Sun, Y. et al. (2023). *RetNet.* arXiv:2307.08621.
- Lieber, O. et al. (2024). *Jamba.* arXiv:2403.19887.
