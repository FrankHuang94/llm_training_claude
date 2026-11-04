# Diffusion & Hybrid Language Models

> **Last Updated:** 2026-11-04
> **Related Files:** [Non-Transformer Architectures](04_neuromorphic_and_non_transformer.md) · [Long Context & Memory](00_long_context_and_memory.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Speculative Decoding](../06_efficiency/04_speculative_decoding.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md)
> **Key Papers:** Nie et al. 2025 LLaDA ([arXiv:2502.09992](https://arxiv.org/abs/2502.09992)) · inclusionAI 2025 LLaDA2.0 ([arXiv:2512.15745](https://arxiv.org/abs/2512.15745)) · Inception Labs 2025 Mercury ([arXiv:2506.17298](https://arxiv.org/abs/2506.17298)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) · DeepSeek-AI 2025 DeepSeek-V3.2 / DSA ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556))

## Overview
Two assumptions of the 2023-era LLM are now being relaxed in production. (1) **Left-to-right, one-token-at-a-time generation**: *discrete diffusion LMs* (dLLMs) generate by iteratively denoising a masked sequence, predicting many tokens per step, and *multi-token prediction* (MTP) adds parallel future-token heads to AR models. (2) **Dense softmax attention in every layer**: *hybrid* models mix a minority of full-attention layers with linear-recurrent (Gated DeltaNet, KDA, Mamba) layers, or replace dense attention with **learned top-k sparse attention** over (compressed) KV. Both lines target the same bottleneck — decode latency and long-context cost — and by 2026 both had 100B+-parameter instantiations (LLaDA2.0-flash; Qwen3-Next/3.5, Kimi K3, DeepSeek-V4, Nemotron 3). This file covers the mechanisms, the scaling evidence, and the dissenting results.

## Core Concepts
**Masked (absorbing-state) diffusion LMs.** Forward process: mask each token independently with probability $t\sim U[0,1]$. A bidirectional Transformer predicts all masked tokens; the loss $\mathcal{L}=\mathbb{E}_{t,x_0,x_t}\big[\tfrac{1}{t}\sum_{i:\,x_t^i=[M]}-\log p_\theta(x_0^i\mid x_t)\big]$ upper-bounds the NLL (D3PM → SEDD → MDLM lineage). Sampling starts fully masked and *unmasks* tokens over $S$ steps (often by confidence), so $S\ll L$ steps can emit $L$ tokens — the source of the speed advantage. **LLaDA** (8B, trained from scratch) showed this is competitive with AR models of similar size, including on in-context learning and the "reversal curse."

**Block diffusion / semi-AR.** Generate blocks left-to-right, diffusing *within* each block: recovers KV caching and variable length, interpolating between AR (block size 1) and full diffusion (block = sequence). LLaDA2.0 and BlockGen exploit this continuum.

**AR→diffusion conversion.** Instead of pretraining from scratch, convert a pretrained AR model: warm up with block diffusion at growing block sizes, train full-sequence diffusion, then decay to small blocks for efficient inference (LLaDA2.0's WSD-style schedule).

**Hybrid linear attention.** Replace most attention layers with a fixed-size recurrent state, e.g. the gated delta rule $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ (Gated DeltaNet; **KDA** makes the decay channel-wise), keeping ~1 in 4 layers as full softmax attention (or MLA) for exact recall. KV cache then scales with only the full-attention layers.

**Learned sparse attention.** Keep softmax but attend to a selected subset: **DSA** uses a cheap "lightning indexer" (few heads, ReLU, FP8) to score past tokens and runs MLA over the top-$k$; the indexer is still $O(L^2)$ but cheap, and main attention becomes $O(Lk)$. **CSA** (DeepSeek-V4) first compresses each $m$-token block into one KV entry, then selects top-$k$ compressed entries; **HCA** applies dense attention over much more heavily compressed KV. MiniMax **MSA** selects top-16 128-token blocks per GQA group.

**Multi-token prediction (MTP).** Auxiliary heads/modules predict tokens $t{+}2,\dots,t{+}k$; used as extra training signal and as a built-in speculative-decoding draft (DeepSeek-V3 style; now standard in Qwen3-Next, GLM-4.5/5).

## Key Challenges
- **dLLM quality/compute gap.** Earlier scaling studies reported that masked diffusion needs more compute than AR to reach equal likelihood, and likelihood is only a bound; parallel unmasking of dependent tokens causes incoherence (the "factorization" error of predicting tokens independently in one step).
- **dLLM inference engineering.** Bidirectional attention breaks standard KV caching; variable-length output and early stopping are awkward; the speed advantage depends on how many tokens can be committed per step without quality loss.
- **Reasoning and RL for dLLMs.** AR RL machinery (token log-probs, GRPO ratios) does not transfer directly because sequence likelihood is intractable.
- **Hybrids lose on multi-hop.** MiniMax found linear hybrids matched full attention on standard benchmarks but lost on multi-hop/long-chain reasoning; linear states are sensitive to low-precision storage.
- **Sparse-attention systems cost.** Indexers add an "indexer K cache" that complicates prefill/decode-disaggregated KV transfer; training needs a dense warm-up so the indexer learns what main attention wants.

## Solutions & Current Best Practices
**dLLMs:** convert from strong AR checkpoints (LLaDA2.0) rather than pretrain from scratch; use **block diffusion** for caching and variable length; add a **confidence loss** so the model can commit many high-confidence tokens per step; prefer masked diffusion for likelihood, but test uniform-state variants on reasoning. **Hybrids:** the converging recipe is **~3:1 linear:full** (Qwen3-Next: 12×[3 GDN + 1 gated attention]; Kimi Linear/K3: 3 KDA : 1 MLA; Nemotron: mostly Mamba-2 with ~8% attention), output-gated attention to remove sinks, and higher-precision storage for recurrent states/projections. **Sparse attention:** train DSA in two stages — dense warm-up aligning the indexer to the main attention distribution with a KL loss, then sparse continued training — and combine with KV compression (CSA/HCA) for 1M context. **Retrofitting:** convert pretrained Transformers into hybrids by continued training (MiniCPM-SALA, STILL) at a fraction of from-scratch cost. **MTP:** keep MTP modules for both training signal and self-speculative decoding.

## Lab Perspectives
- **Google DeepMind:** Gemini Diffusion (2025-05, experimental, reportedly on par with Gemini 2.0 Flash-Lite); no public updates since (secondary).
- **Inception Labs:** first commercial dLLMs (Mercury Coder Mini/Small); Mercury 2 (2026-02) billed as the first diffusion reasoning LM at ~1,000 tok/s on Blackwell (self-reported).
- **inclusionAI / Ant:** LLaDA lineage; LLaDA2.0 scales AR→diffusion conversion to 100B MoE.
- **Qwen / Moonshot:** production linear hybrids (Qwen3-Next, Qwen3.5; Kimi Linear, Kimi K3 with KDA+MLA 3:1).
- **DeepSeek / Zhipu:** learned sparse attention (DSA in V3.2, adopted by GLM-5; CSA+HCA in V4).
- **MiniMax:** hybrid in M1 → full attention in M2 → block-sparse MSA in M3 — the field's most explicit public reversal.
- **NVIDIA:** Mamba-2/Transformer(/MoE) hybrids (Nemotron-H, Nano 2, Nemotron 3), released with data and recipes.

## Latest Developments (2023–Nov 2026)
Discrete diffusion matured from D3PM to SEDD and MDLM (2023–24), then **LLaDA** (2025-02) scaled masked diffusion to 8B. Mamba/Mamba-2 and Gated DeltaNet made linear-recurrent layers competitive, and Jamba/Nemotron-H showed hybrids work at scale; DeepSeek-V3 made MTP mainstream and NSA/MoBA made sparse attention trainable.

**Mid-2025 → Nov 2026 update:**
- **Commercial and 100B-scale dLLMs (2025-05 → 2026-02).** Gemini Diffusion (I/O 2025, secondary); Mercury predicts many tokens in parallel per step ([arXiv:2506.17298](https://arxiv.org/abs/2506.17298)); **LLaDA2.0** converts AR MoE models to diffusion at 16B (mini) and 100B (flash) via block-diffusion warm-up → full-sequence → small-block decay, with an auxiliary confidence loss for aggressive parallel decoding ([arXiv:2512.15745](https://arxiv.org/abs/2512.15745)).
- **Diffusion scaling laws (2026-02).** "Scaling Beyond Masked Diffusion LMs" compares masked, uniform-state and interpolating diffusion against AR: masked diffusion has the best likelihood scaling among diffusion variants, yet uniform-state diffusion does better on GSM8K despite worse perplexity ([arXiv:2602.15014](https://arxiv.org/abs/2602.15014)) — perplexity is not the right yardstick for dLLMs. BlockGen (2026-06) trains over mixed block sizes so likelihood interpolates smoothly between AR and diffusion (single source).
- **Linear hybrids in production (2025-09 → 2026-07).** Qwen3-Next 80B-A3B (3 GDN : 1 gated attention; >10× prefill throughput beyond 32K vs Qwen3-32B); **Kimi Linear** (KDA, 48B-A3B) claims to beat full attention under fair comparison incl. RL, with up to 75% less KV and 6× decode at 1M ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)); **Kimi K3** scales KDA+MLA 3:1 to 2.8T/104B active (69 KDA + 24 MLA layers, MLA reportedly NoPE).
- **MiniMax returns to full attention (2025-10).** M2 dropped M1's lightning-attention hybrid (multi-hop losses, low-precision state sensitivity, immature infra); M3 (~2026-06) then adopted block-sparse MSA with claimed 9× prefill / 15× decode speedups over M2 at 1M.
- **DSA → CSA/HCA (2025-09 → 2026-04).** DeepSeek-V3.2-Exp isolated DSA by matching V3.1-Terminus's training config (quality roughly on par, API price cut >50%); GLM-5 adopted DSA at 744B ([arXiv:2602.15763](https://arxiv.org/abs/2602.15763)); DeepSeek-V4 (1.6T/49B active) reportedly needs ~27% of V3.2's per-token FLOPs and ~10% of its KV at 1M (report listed as arXiv:2606.19348; ID/date inconsistent, verify).
- **SSM hybrids (2025-08 → 2026-03).** Nemotron Nano 2 (28 Mamba-2 + 28 FFN + 6 attention; up to 6.3× throughput vs Qwen3-8B; [arXiv:2508.14444](https://arxiv.org/abs/2508.14444)); Nemotron 3 Mamba-2/MoE family with Super pretrained in NVFP4 ([arXiv:2512.20856](https://arxiv.org/abs/2512.20856)); **Mamba-3** adds complex-valued transitions and MIMO decode ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)).
- **MTP beyond next-$k$ tokens (2025-10 → 2026-02).** Future Summary Prediction predicts a compact summary of the long-range future, beating NTP and MTP at 3B/8B ([arXiv:2510.14751](https://arxiv.org/abs/2510.14751)); self-distillation converts an NTP model into a standalone MTP decoder, >3× faster on GSM8K with <5% accuracy loss ([arXiv:2602.06019](https://arxiv.org/abs/2602.06019)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Write the masked-diffusion training loss and explain the $1/t$ weight."** ELBO under absorbing-state noise; each masked position contributes CE weighted by the inverse masking rate.
- **"Why might a diffusion LM be faster at decode, and what limits it?"** Many tokens committed per denoising step; limited by dependency between simultaneously unmasked tokens, lack of KV caching (fixed by block diffusion), and steps needed for quality.
- **"Qwen3-Next vs DeepSeek-V3.2 vs MiniMax-M2 — three answers to long context. Compare."** Linear hybrid (bounded state) vs learned top-k sparse softmax vs full attention; KV, FLOPs, multi-hop robustness, precision sensitivity, kernel maturity.
- **"How is DSA's indexer trained?"** Dense warm-up with a KL loss aligning indexer scores to the main attention distribution, then sparse training with top-$k$.
- **"MTP vs speculative decoding vs diffusion?"** MTP = AR with extra heads (self-drafting, exact via verification); diffusion = non-AR parallel denoising (not exact w.r.t. any AR distribution).

## Open Problems
Whether dLLMs close the quality gap at frontier scale (and whether RL/reasoning transfers to them), the right yardstick when perplexity misleads, how to do exact-likelihood RL for diffusion, whether ~3:1 hybrids or sparse softmax better preserve long-chain reasoning (Kimi vs MiniMax), the optimal compression/selection hierarchy for 1M+ context, and how these designs interact with FP4 training and disaggregated serving are all open.

## References
- Austin, J. et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).* arXiv:2107.03006.
- Lou, A. et al. (2023). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD).* arXiv:2310.16834.
- Sahoo, S. et al. (2024). *Simple and Effective Masked Diffusion Language Models (MDLM).* arXiv:2406.07524.
- Nie, S. et al. (2025). *Large Language Diffusion Models (LLaDA).* arXiv:2502.09992.
- Arriola, M. et al. (2025). *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.* arXiv:2503.09573.
- Inception Labs (2025). *Mercury: Ultra-Fast Language Models Based on Diffusion.* arXiv:2506.17298.
- inclusionAI (2025). *LLaDA2.0.* arXiv:2512.15745.
- Multiple authors (2026). *Scaling Beyond Masked Diffusion Language Models.* arXiv:2602.15014.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- NVIDIA (2025). *Nemotron Nano 2.* arXiv:2508.14444.
- Gloeckle, F. et al. (2024). *Better & Faster Large Language Models via Multi-token Prediction.* arXiv:2404.19737.
