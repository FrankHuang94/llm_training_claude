# Non-Transformer Architectures

> **Last Updated:** 2026-11-04
> **Related Files:** [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Diffusion & Hybrid LMs](06_diffusion_and_hybrid_language_models.md) · [Test-Time Training](03_test_time_training.md) · [Long Context & Memory](00_long_context_and_memory.md)
> **Key Papers:** Gu & Dao 2023 Mamba ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) · Dao & Gu 2024 Mamba-2 / SSD ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)) · Peng et al. 2023 RWKV ([arXiv:2305.13048](https://arxiv.org/abs/2305.13048)) · Sun et al. 2023 RetNet ([arXiv:2307.08621](https://arxiv.org/abs/2307.08621)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) · Mamba-3 2026 ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569))

## Overview
The Transformer's $O(n^2)$ attention and large KV cache motivate a search for **sub-quadratic** sequence models that match its quality. The leading challengers are **state-space models (SSMs)** — especially **Mamba** — along with **linear attention** variants (RWKV, RetNet, Hyena). These aim for the "holy grail": **Transformer-quality with RNN-like linear-time inference and constant memory** (no growing KV cache). After years of linear-attention approaches underperforming, Mamba (2023) reignited the field by closing much of the quality gap, and 2024–2026 saw **hybrid** attention-SSM / attention-linear models reach production — by 2026 several open *flagships* (Qwen3.5, Kimi K3, Nemotron 3) are hybrids. Understanding the SSM formulation, the attention-vs-SSM tradeoffs, and why hybrids dominate is increasingly expected knowledge.

## Core Concepts
**State-space models (SSMs).** A continuous linear system $h'(t)=Ah(t)+Bx(t)$, $y(t)=Ch(t)$, discretized to a recurrence $h_t=\bar A h_{t-1}+\bar B x_t$, $y_t=\bar C h_t$. Key duality: it can be computed as a **recurrence** (linear-time, constant-memory inference, like an RNN) **or** as a **convolution** (parallelizable training). Earlier SSMs (**S4**) used clever structured/HiPPO initialization to capture long-range dependencies.

**Mamba (selective SSM).** The breakthrough: make the SSM parameters $(\bar B, \bar C, \Delta)$ **input-dependent** ("selective"), so the model can *choose* what to remember or forget based on content — overcoming the key weakness of linear-time-invariant SSMs (they couldn't do content-based reasoning like attention). This costs the convolution view, so Mamba uses a **hardware-aware parallel scan** for efficient training. Mamba matched Transformers at small-mid scale with linear-time, constant-memory inference.

**Mamba-2 / SSD.** Establishes a theoretical bridge ("**state-space duality**") showing SSMs and (a form of) attention are closely related, enabling faster, more hardware-friendly implementations and larger state sizes.

**Linear attention family.**
- **RWKV:** an RNN/Transformer hybrid with linear attention reformulated for parallel training and recurrent inference; the "RNN that scales," with RWKV-7 adding expressive, dynamic state.
- **RetNet:** "retention" mechanism with parallel (train), recurrent (infer), and chunkwise forms.
- **Hyena:** replaces attention with long implicit convolutions + gating (sub-quadratic).
- All share the goal of $O(n)$ inference; recent variants (**DeltaNet, Gated DeltaNet**) add delta-rule/gated updates for stronger associative recall, overlapping with TTT (see [Test-Time Training](03_test_time_training.md)). **Kimi Delta Attention (KDA)** refines Gated DeltaNet with channel-wise (rather than head-wise) gating — conceptually, the scalar per-head decay $\alpha_t$ becomes a per-channel diagonal $\mathrm{Diag}(\alpha_t)$ — giving finer control over what each state dimension forgets.

**Mamba-3.** Inference-first redesign: exponential-trapezoidal discretization, **complex-valued** state transitions (restoring state-tracking ability that real, non-negative decays lack), and a **MIMO** update that raises decode arithmetic intensity on GPUs.

**The fundamental tradeoff.** Attention keeps *all* tokens (perfect recall, quadratic cost); SSMs compress history into a *fixed-size state* (linear cost, lossy recall). SSMs excel at long sequences and throughput but are **weaker at precise recall / copying / in-context retrieval** — exactly what attention does best.

## Key Challenges
- **Recall / copying weakness.** Fixed-state models struggle with tasks needing exact retrieval from long context (associative recall, needle tests) — attention's strength.
- **In-context learning gap.** Some ICL/induction-head behaviors are harder for pure SSMs.
- **Scaling validation.** Most decisive comparisons are at small/mid scale; frontier-scale parity is less established (and labs keep using attention or hybrids).
- **Ecosystem maturity.** Decades of attention optimization (FlashAttention, kernels, tooling) advantage Transformers; SSM kernels/tooling are newer.
- **Hybrid design.** Choosing the attention:SSM ratio and placement is empirical; massive activations in GDN hybrids reportedly cluster just before full-attention layers (single source).
- **Numerics.** Recurrent states are sensitive to low-precision storage (MiniMax's reason to abandon linear attention in M2); Nemotron 3 keeps Mamba output projections in MXFP8 even under NVFP4 pretraining.
- **Long-chain reasoning.** Hybrids can match full attention on standard benchmarks yet lose on multi-hop/long-chain reasoning (MiniMax M2 report).

## Solutions & Current Best Practices
**Hybrids win in practice.** Interleave a *few* full-attention layers (for recall/ICL) with many SSM/linear layers (for efficiency) — e.g., **Jamba** (Mamba-Transformer-MoE), **Zamba**, **Nemotron-H / Nano 2 / Nemotron 3**, **Samba**, and the 2025–26 production recipe of **~3:1 linear:full** (Qwen3-Next/3.5: Gated DeltaNet + gated attention; Kimi Linear/K3: KDA + MLA). This captures attention's recall where it matters while getting SSM throughput/long-context efficiency elsewhere. For pure long-sequence throughput with modest recall needs, Mamba-2/3 and RWKV-7 are strong. A competing sub-quadratic route keeps *softmax* attention but makes it **sparse** (learned top-k: DSA/CSA, MiniMax MSA) — see [Diffusion & Hybrid LMs](06_diffusion_and_hybrid_language_models.md). Pretrained Transformers can also be **linearized/hybridized by continued training** (MiniCPM-SALA, STILL) at a fraction of from-scratch cost. Pure SSMs remain niche at the frontier; closed labs do not disclose architectures (gpt-oss, the one open OpenAI design, uses alternating sliding-window/dense attention with learned sinks).

## Lab Perspectives
- **Academia (CMU/Princeton — Gu & Dao):** Mamba/Mamba-2, the leading SSM line.
- **AI21:** **Jamba** — first production-scale hybrid SSM-Transformer-MoE.
- **NVIDIA:** Nemotron-H → Nemotron Nano 2 (only ~8% attention layers) → **Nemotron 3** (Mamba-2 + MoE + few attention layers; Nano 30B-A3B, Super 120B-A12B pretrained in NVFP4), released fully open with data and recipes.
- **Qwen / Moonshot:** production linear hybrids (Qwen3-Next, Qwen3.5; Kimi Linear, Kimi K3).
- **MiniMax:** the counter-example — lightning-attention hybrid in M1, **back to full attention** in M2, then block-sparse attention (MSA) in M3.
- **RWKV/open community:** RWKV (linear-attention RNN), broad open ecosystem and edge focus.
- **US frontier labs (OpenAI/Anthropic/Google/Meta):** architectures of closed models are undisclosed; public signals are attention-centric (gpt-oss sliding-window + sinks), with Google publishing the most recurrent-memory research (Titans/ATLAS/HOPE).

## Latest Developments (2023–Nov 2026)
**Mamba (2023)** revived the field; **Mamba-2/SSD** unified SSMs and attention theoretically. **Hybrids** (Jamba, Zamba, Nemotron-H, Samba) became the pragmatic sweet spot and reached production scale. **Expressive recurrent state** (Gated DeltaNet, RWKV-7, Titans) blurred SSMs with **test-time learning/fast weights**. The consensus crystallized: **hybrids, not pure replacements** — attention for recall, SSMs for efficiency — though the optimal mix and whether pure SSMs can ever fully match attention at the frontier remain open.

**Mid-2025 → Nov 2026 update:**
- **Nemotron Nano 2 (2025-08) → Nemotron 3 (2025-12/2026-03).** 62 layers = 28 Mamba-2 + 28 FFN + only 6 attention; 20T tokens in FP8; up to 6.3× throughput vs Qwen3-8B on reasoning workloads ([arXiv:2508.14444](https://arxiv.org/abs/2508.14444)). Nemotron 3 interleaves Mamba-2 with MoE ([arXiv:2512.20856](https://arxiv.org/abs/2512.20856)); Super (120B-A12B) is pretrained in NVFP4 on ~25T tokens ([arXiv:2604.12374](https://arxiv.org/abs/2604.12374)).
- **Qwen3-Next (2025-09) and Qwen3.5 (2026-02).** 48 layers = 12 × [3 Gated DeltaNet + 1 gated attention] in an 80B-A3B, 512-expert MoE; >10× prefill throughput vs Qwen3-32B beyond 32K. Qwen3.5-397B-A17B reportedly keeps the GDN hybrid (no tech report; secondary).
- **Kimi Linear (2025-10) → Kimi K3 (2026-07).** KDA interleaved 3:1 with MLA (48B-A3B) — claimed first linear hybrid to beat full attention under fair comparison across short-context, long-context and RL regimes, with up to 75% less KV and 6× decode at 1M ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)); K3 scales it to 2.8T total / 104B active with 69 KDA + 24 MLA layers.
- **MiniMax's dissent (2025-10).** M2 abandoned M1's lightning-attention hybrid: equal on standard benchmarks but worse on multi-hop/long-chain reasoning, sensitive to low-precision state, immature infra (MiniMax blog) — the canonical interview counterpoint.
- **Mamba-3 (2026-03, ICLR 2026 oral).** Complex-valued transitions, exponential-trapezoidal discretization, MIMO decode; Mamba-2 perplexity at half the state size ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)).
- **Retrofitting (2026-02).** MiniCPM-SALA converts a pretrained Transformer into a 1:3 sparse:linear hybrid via continual training (~75% cheaper than from scratch; secondary), and STILL routes salient tokens to softmax and the rest to linear attention within a layer with ~0.04B training tokens ([arXiv:2602.02180](https://arxiv.org/abs/2602.02180)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"What problem do SSMs/Mamba solve vs Transformers?"** Sub-quadratic, constant-memory inference (no growing KV cache) for long sequences.
- **"What made Mamba work where earlier SSMs didn't?"** Input-dependent ('selective') parameters enable content-based remembering/forgetting; hardware-aware parallel scan.
- **"What's the fundamental SSM-vs-attention tradeoff?"** Fixed-state compression (linear, lossy recall) vs keep-all-tokens (quadratic, perfect recall).
- **"Why do hybrids dominate?"** A few attention layers restore recall/ICL; SSM layers give efficiency.
- **"Where do SSMs struggle?"** Exact copying/associative recall/needle-in-haystack.
- **"Linear hybrid (Kimi Linear) vs sparse attention (DSA) vs full attention (MiniMax M2) — which would you pick for a 1M-context agent model?"** Bounded state and 3:1 ratio vs exact-but-selected softmax over compressed KV vs simplicity and multi-hop robustness; discuss KV cost, precision sensitivity, kernel maturity, RL behavior.

## Open Problems
Whether a **pure** sub-quadratic architecture can match attention on **recall/ICL at frontier scale** is unresolved. Optimal **hybrid ratios** (why ~3:1 keeps recurring), whether hybrids degrade long-chain reasoning (MiniMax) or not (Kimi), closing the recall gap, mature kernels/tooling, and the deeper theory linking SSMs, attention, fast weights, and test-time learning are active. The field's likely near-term answer is *efficient hybrids*, but a clean Transformer successor remains an open prize.

## References
- Gu, A., Dao, T. (2023). *Mamba.* arXiv:2312.00752.
- Dao, T., Gu, A. (2024). *Mamba-2 / State Space Duality.* arXiv:2405.21060.
- Peng, B. et al. (2023). *RWKV.* arXiv:2305.13048.
- Sun, Y. et al. (2023). *RetNet.* arXiv:2307.08621.
- Lieber, O. et al. (2024). *Jamba.* arXiv:2403.19887.
- NVIDIA (2025). *Nemotron Nano 2.* arXiv:2508.14444.
- NVIDIA (2025). *Nemotron 3.* arXiv:2512.20856.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- CMU/Princeton/Together AI (Gu, Dao et al.) (2026). *Mamba-3.* arXiv:2603.15569.
