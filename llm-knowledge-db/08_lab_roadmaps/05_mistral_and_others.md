# Mistral and Other Labs

> **Last Updated:** 2026-11-04
> **Related Files:** [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Optimizer Choices](../02_pretraining/03_optimizer_choices.md) · [DeepSeek](04_deepseek_roadmap.md) · [Lab Comparison Matrix](06_lab_comparison_matrix.md)
> **Key Papers:** Jiang et al. 2023 Mistral 7B ([arXiv:2310.06825](https://arxiv.org/abs/2310.06825)) · Jiang et al. 2024 Mixtral ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)) · Kimi Team 2025 Kimi K2 ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) · Groeneveld et al. 2024 OLMo ([arXiv:2402.00838](https://arxiv.org/abs/2402.00838))

## Overview
Beyond the "big five," a second tier of labs now sets much of the agenda — and, by 2026, **leads the open-weight frontier**. The most important groups are **Mistral AI** (Europe's frontier challenger, which shipped the first Western ~1T-parameter open model in Oct 2026) and the **Chinese open-weight labs** — **Alibaba Qwen**, **Moonshot (Kimi)**, **Zhipu/Z.ai (GLM)**, **MiniMax** and **Xiaomi MiMo** (DeepSeek has its [own file](04_deepseek_roadmap.md)) — whose open MoEs reached **1–2.8T total parameters** with 1M-token context between Jul 2025 and Aug 2026. Around them sit **xAI**, **NVIDIA (Nemotron)**, **AI2 (OLMo)**, **Thinking Machines**, **Microsoft (phi/MAI)** and **Cohere**. Collectively they drive open-model competition, sub-quadratic attention, low-precision training, Muon-family optimizers, sovereign AI, and fully open science.

## Mistral AI
**Overview.** Founded 2023 in Paris by ex-DeepMind/Meta researchers (Mensch, Lample, Lacroix); Europe's leading frontier lab, positioned around **open weights**, **efficiency**, and **European AI sovereignty**.

**Model lineage & innovations.**
- **Mistral 7B (2023):** punched far above its size; popularized **Sliding-Window Attention (SWA)** and **GQA** in a clean Apache 2.0 open model.
- **Mixtral 8×7B / 8×22B (2023–2024):** mainstreamed **open sparse MoE** (top-2 of 8 experts; ~13B active of 47B).
- **2024 → mid-2025:** Mistral Large / Small / Nemo (12B, with NVIDIA) / **Codestral** / **Pixtral**; Mistral Small 3 and Medium 3; **Magistral** reasoning models and **Devstral** coding agents (2025).
- **Mistral 3 (2025-12-02):** **Mistral Large 3**, a "granular" sparse MoE with **675B total / 41B active** (≈673B/39B LM + 2.5B vision encoder), 256K context, Apache 2.0 for base and instruct (deprecated 2026-10-31); **Ministral 3** at 3B/8B/14B, each with base, instruct and reasoning variants.
- **Mistral Medium 3.5 (2026-04-29):** **128B dense** open weights (modified MIT) merging instruct, reasoning and coding into one checkpoint, replacing Magistral and Devstral 2; image input (context reported as 256K or 128K).
- **Mistral Large 4, "Le Chonk" (preview 2026-10-06; weights 2026-10-29):** **~1.05T total / 52B active** MoE (49B is per-token expert compute; 52B also counts embeddings and output layers) plus a 1.6B vision encoder, under the **Mistral Ecosystem Open-Weight License** (modified MIT; Model-as-a-Service providers need a commercial license, no revenue threshold). Described as the top open model outside China, still trailing the Chinese leaders overall.

**Differentiation.** Efficiency-first architecture (SWA, GQA, granular MoE), consolidation of modes into single checkpoints, European data-sovereignty positioning, and on-prem/enterprise deployment. **Compute:** EU-based NVIDIA clusters (a "Mistral Compute" platform was announced with NVIDIA in mid-2025), far smaller than US gigawatt contracts. **Strategy:** be the open, sovereign alternative for enterprises and governments. **Strengths:** openness at ~1T scale, efficiency, EU regulatory/trust positioning. **Weaknesses:** compute budget, narrower research breadth, and a gap to both US closed models and Chinese open models.

## Chinese Open-Weight Labs
- **Alibaba Qwen.** The broadest open family: **Qwen3** (Apr 2025; dense 0.6B–32B plus 30B-A3B and 235B-A22B MoEs, hybrid thinking) → **Qwen3-Next-80B-A3B** (2025-09; **3:1 Gated DeltaNet : gated attention** hybrid, 512-expert MoE with 10 routed + 1 shared, 262K context; >10× throughput beyond 32K vs Qwen3-32B at <1/10 training cost) → **Qwen3.5-397B-A17B** (2026-02; GDN hybrid, native VL, Apache 2.0) → **Qwen3.6** (2026-04; 35B-A3B and 27B dense open) → **Qwen3.8-2.4T-A95B** (open weights 2026-08; 512 experts, 262K native context extensible to ~1M; text-only open checkpoint with thinking always on, custom license). The closed **Qwen3-Max** (>1T, 36T tokens) and Qwen3.8-Max-Preview (claimed "second only to Fable 5") are hosted only.
- **Moonshot AI (Kimi).** **Kimi K2** (2025-07; ~1.04T / 32B active, 15.5T tokens with **MuonClip** — Muon + QK-clip that rescales Q/K projections per head when max logits exceed a threshold — and zero loss spikes); **Kimi Linear** (2025-10; **Kimi Delta Attention**, channel-wise-gated Gated DeltaNet interleaved 3:1 with MLA; up to 75% less KV cache and 6× decode at 1M); **K2 Thinking** (2025-11; **INT4 weight-only QAT** on MoE weights during post-training, all benchmarks reported at INT4); **Kimi K3** (hosted 2026-07-16, weights 07-27; **2.8T / 104B active**, 896 experts with 16 routed + 2 shared, 69 KDA + 24 gated-MLA layers, block-level Attention Residuals, native vision, 1M context, MXFP4 weights; custom license requiring negotiation above $20M revenue); **K3.1 Pro** (2026-10-23, API; open weights promised for 11-04, not yet confirmed, with cyber training data reportedly removed). Also **MoBA** block-sparse attention for long-context serving.
- **Zhipu / Z.ai (GLM).** **GLM-4.5** (2025-07; 355B / 32B active) → **GLM-4.6** (2025-09; 200K context, MIT) → **GLM-5** (2026-02-11; **744B / 40B active**, **adopts DeepSeek Sparse Attention**, 28.5T tokens, asynchronous agent RL on the **slime** framework; reportedly trained entirely on **Huawei Ascend + MindSpore**, unconfirmed by Z.ai) → **GLM-5.1** (2026-04; post-training refresh).
- **MiniMax.** M1's hybrid lightning (linear) attention gave way in **MiniMax-M2** (2025-10; ~230B / 10B active) to **full attention**, because SWA and linear variants hurt multi-hop reasoning — the key counterexample to the hybrid trend; M2.5 (2026-02); **M3** (~2026-06; **MiniMax Sparse Attention**: an indexer scores 128-token blocks and keeps the top-16 per GQA group; 1M context; claimed 9× prefill / 15× decode vs M2 at 1M; open weights promised).
- **Xiaomi MiMo.** **MiMo-V2-Flash** (2025-12; 309B / 15B active, hybrid sliding-window attention) → **MiMo-V2.5 / V2.5-Pro** (2026-04; MIT, native vision and audio, 1M context; reportedly 5:1 SWA:global with a 128-token window).

## Other Significant Labs
- **xAI (Grok).** Musk-founded (2023); Grok models integrated with X; the **Colossus** supercluster (~100k→200k H100s) built at remarkable speed — a compute-first bet; Grok 3 (Feb 2025) was competitive on reasoning. Later releases (Grok 4 / 4.1, a Grok 5 in training), Colossus 2 and a corporate combination with SpaceX are widely reported but **not verified in our sources**.
- **NVIDIA — Nemotron 3.** Fully open (weights, data, recipes) **hybrid Mamba-2/Transformer MoE** family: **Nano** 30B-A3B (2025-12; 23 Mamba-2/MoE + 6 attention layers, 1M context), **Super** 120B / 12B active (2026-03; **first in the family pretrained in NVFP4**, ~25T tokens), and **Ultra** (~mid-2026, NVFP4 on Blackwell; final size unconfirmed). NVIDIA's NVFP4 paper (12B on 10T tokens in 4-bit, close to FP8) documents the recipe.
- **AI2 — OLMo.** **Fully open** models (weights, data, code, logs); Tülu 3 (open post-training, coined RLVR); **OLMo 3** (2025-11; 7B/32B Base and Think, Dolma 3 ~6T tokens, Apache 2.0) and OLMo 3.1 (~2025-12) — the reproducible-science reference point.
- **Thinking Machines Lab.** Founded 2025 by Mira Murati; influential post-training research blogs — **"LoRA Without Regret"** (2025-09: all-layer LoRA matches full FT; rank-1 suffices for RL since policy gradients absorb ~1 bit/episode) and **"On-Policy Distillation"** (2025-10: student samples, teacher gives per-token reverse-KL reward; ~70% AIME24 at ~1/10 the GPU-hours of RL). A fine-tuning API (Tinker) is reported but not verified here.
- **Microsoft — phi / MAI.** phi-1→phi-4 showed the synthetic "textbook-quality" data thesis for small models; **MAI-1-preview** (2025-08) was Microsoft's first fully in-house foundation model, and a reasoning model (MAI-Thinking-1, 2026-06) was reportedly released — a hedge against dependence on OpenAI.
- **Cohere.** Enterprise/RAG-focused (Command R/R+, Command A), multilingual (Aya); business-oriented rather than consumer.
- **Others:** 01.AI (Yi), Baidu (ERNIE); Google's **Gemma 4** and OpenAI's **gpt-oss** are covered in their labs' files.

## Key Challenges (shared)
- **Compute disadvantage** vs US labs contracting tens of gigawatts; for Chinese labs, **export controls** and the move to domestic accelerators.
- **Gap to restricted US frontier models** (Mythos/Fable, GPT-6 Astra): roughly one release cycle; open checkpoints sometimes ship with capabilities removed (cyber data in K3.1 open weights).
- **Monetization** of open models — hence "MIT, but" licenses with MaaS or revenue carve-outs (Kimi, Qwen3.8, Mistral Large 4) and hosted-first, weights-later releases.
- **Serving trillion-parameter MoEs** requires rack-scale NVLink domains (NVIDIA publishes GB300 NVL72 guides for Qwen3.8), raising the bar for "open" in practice.

## Lab Perspectives (how they differ)
- **Mistral:** open + efficient + European sovereignty; now at ~1T scale with mode-merged checkpoints.
- **Qwen:** breadth across every size, linear-attention hybrids (Gated DeltaNet), and the largest open MoE expert counts.
- **Kimi:** optimizer and attention innovation (MuonClip, KDA, AttnRes) and the largest open model (K3, 2.8T).
- **GLM:** DSA adoption, async agentic RL (slime), reportedly all-domestic training stack.
- **MiniMax:** contrarian on attention — full attention in M2, learned block-sparse in M3.
- **NVIDIA:** fully open hybrids that showcase NVFP4 pretraining and its hardware.
- **xAI:** compute-first, fast-scaling, X-integrated.
- **OLMo / Thinking Machines:** open, reproducible science and post-training methods.

## Latest Developments (2023–Nov 2026)
Open models (Qwen, Mistral, DeepSeek, OLMo) closed much of the gap to closed frontier in 2023–mid-2025; **reasoning** spread to open releases (QwQ, Magistral, R1). xAI's **Colossus** showed compute can be assembled extremely fast, and **Muon** adoption (Kimi) began.

**Mid-2025 → Nov 2026 update:**
- **Trillion-parameter open MoEs (2025-07 → 2026-10):** largest open weights went 1T (Kimi K2) → 1.6T (DeepSeek V4-Pro) → 2.8T (Kimi K3) → 2.4T (Qwen3.8) → ~1.05T (Mistral Large 4, first Western); active parameters stay ~13–104B, so sparsity keeps rising (K3: 16 of 896 experts).
- **Sub-quadratic attention becomes standard in Chinese flagships:** ~3:1 linear-to-full hybrids (Qwen3-Next/3.5 with Gated DeltaNet; Kimi Linear/K3 with KDA + MLA), learned top-k sparse attention (DSA in GLM-5; MSA in MiniMax M3), and SWA hybrids (MiMo) — with MiniMax M2's return to full attention as the counterexample. (arXiv:2510.26692)
- **1M-token context as default spec:** Kimi K3/K3.1, MiniMax M3, MiMo-V2.5 and Nemotron 3 all ship 1M; the selling point is KV/FLOP reduction, not window size.
- **Training at 4 bits:** K2 Thinking (INT4 QAT), K3 (MXFP4 weights), Nemotron 3 Super/Ultra (**NVFP4 pretraining**; Hadamard transforms, stochastic rounding, BF16 tail layers). (arXiv:2509.25149)
- **Muon in production:** MuonClip let K2 train 15.5T tokens with zero loss spikes; Muon is reported for GLM-5 and DeepSeek-V4; stability work shifts to sparsity (K3's "Quantile Balancing"). (arXiv:2507.20534)
- **Release pattern:** hosted preview first, weights 1.5–4 weeks later (K3, Qwen3.8, Mistral Large 4), with open checkpoints sometimes differing from the hosted product (Qwen3.8 text-only; K3.1 cyber data removed).
- **Post-training recipes from small labs:** on-policy distillation and all-layer LoRA (Thinking Machines) and fully open RL pipelines (OLMo 3 Think) became standard references.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why was Mistral 7B impressive, and what is sliding-window attention?"** Strong small open model; local windowed attention for $O(n\cdot w)$ cost + GQA (see [Attention](../00_foundations/01_attention_mechanisms.md)).
- **"What did Mixtral demonstrate about MoE — and how have open MoEs evolved since?"** Top-2-of-8 → fine-grained 512–896 experts with shared experts and ~2–5% activation.
- **"Compare Gated DeltaNet/KDA hybrids with DSA/MSA-style sparse attention. Why did MiniMax M2 go back to full attention?"** Recurrent-state capacity vs learned top-k retrieval; multi-hop reasoning and low-precision state sensitivity.
- **"What is MuonClip / QK-clip and why was it needed?"** Muon at scale caused attention-logit blow-up; per-head rescaling of Q/K weights caps max logits (see [Optimizer Choices](../02_pretraining/03_optimizer_choices.md)).
- **"How would you pretrain in NVFP4?"** Block scaling, random Hadamard transforms on Wgrad inputs, stochastic rounding, high-precision sensitive layers.
- **"What makes OLMo different?"** Fully open data+code+weights → reproducible science.

## Open Problems / Watch Items
Whether open labs can match restricted US frontier models on long-horizon agents, whether ~3:1 hybrids or compressed-sparse attention win at 1M context, the economics of trillion-parameter open weights, license fragmentation ("MIT, but"), the geopolitics of Chinese open models in Western deployment and their hardware provenance, Mistral's ability to fund the next scale, and whether xAI's compute-first bet translates into verifiable frontier leadership.

## References
- Jiang, A. et al. (2023). *Mistral 7B.* arXiv:2310.06825.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Bai, J. et al. (2023). *Qwen Technical Report.* arXiv:2309.16609.
- Yang, A. et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- GLM-5 Team (2026). *GLM-5.* arXiv:2602.15763.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
- Groeneveld, D. et al. (2024). *OLMo.* arXiv:2402.00838.
- Abdin, M. et al. (2024). *Phi-3.* arXiv:2404.14219.
