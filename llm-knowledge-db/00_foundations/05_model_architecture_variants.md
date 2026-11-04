# Model Architecture Variants

> **Last Updated:** 2026-11-04
> **Related Files:** [Transformer Architecture](00_transformer_architecture.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md) · [Attention Mechanisms](01_attention_mechanisms.md)
> **Key Papers:** Chowdhery et al. 2022 PaLM ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) · Touvron et al. 2023 LLaMA 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · Jiang et al. 2024 Mixtral ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)) · DeepSeek-AI 2024 DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))

## Overview
Although nearly all frontier LLMs share the decoder-only Transformer skeleton, they differ in a set of well-defined "knobs": normalization type and placement, activation function, attention variant, positional scheme, dense-vs-MoE FFNs, and various stability tricks. These choices are not cosmetic — they materially affect training stability, inference cost, and quality-per-FLOP. This file catalogs the design space and the converged "modern recipe," then maps how flagship model families instantiate it.

The story of 2020→2026 is a gradual standardization (pre-norm + RMSNorm + RoPE + SwiGLU + GQA) followed by a re-diversification driven by efficiency: MoE FFNs, latent attention (MLA), attention–SSM and linear-attention hybrids, learned sparse attention, and — by 2026 — trillion-parameter open MoEs with 4-bit weights.

## Core Concepts
**Normalization.** Original Transformers used **post-norm** LayerNorm (unstable at depth). Modern models use **pre-norm** ($x+\text{Sublayer}(\text{Norm}(x))$) for stable gradients, almost always with **RMSNorm** (no mean subtraction). Some very large runs add **QK-norm** (normalize queries/keys before attention) and **sandwich/double norm** for stability.

**Activations.** ReLU → GeLU (BERT/GPT-2/3) → **gated linear units**, especially **SwiGLU** (LLaMA, PaLM, Mistral). Gated activations consistently improve quality per parameter; the FFN hidden dim is reduced to ~$\frac{8}{3}d$ to hold parameter count constant given the extra gate matrix.

**Attention variants.** MHA → **MQA** (PaLM) → **GQA** (LLaMA-2 70B+, Mistral) for KV-cache reduction; **sliding-window** (Mistral); **MLA** (DeepSeek) for low-rank KV compression. See [Attention Mechanisms](01_attention_mechanisms.md).

**Dense vs MoE FFN.** Dense models activate all parameters per token. **Sparse MoE** replaces the FFN with $E$ experts and a router selecting top-$k$ (2 in Mixtral-era designs; 8–16 out of hundreds in 2025–26 fine-grained designs), so active parameters $\ll$ total. Mixtral 8×7B (47B total, ~13B active), DeepSeek-V3 (671B total, 37B active), Llama 4 (MoE), Kimi K2 (1T/32B, 384 experts, 8 active) and Kimi K3 (2.8T/104B, 896 experts, 16 active + 2 shared) embody this. MoE decouples capacity from compute — see [MoE](../06_efficiency/05_mixture_of_experts.md).

**Parallel layers.** PaLM computes attention and FFN from the *same* normalized input in parallel (rather than sequentially), saving a normalization and improving TPU utilization.

**Depth vs width.** Given a parameter budget, aspect ratio (layers vs $d_{model}$) trades expressivity for parallelism; deeper-narrower helps reasoning but hurts hardware efficiency and stability.

## Key Challenges
- **Stability at scale.** Some choices (post-norm, certain activations, large LR) cause loss spikes; mitigations interact non-trivially.
- **MoE instability and load imbalance.** Routers can collapse to a few experts; require auxiliary load-balancing losses or loss-free balancing.
- **Inference vs training optimality diverge.** GQA/MQA/MoE help inference but complicate training; the best training architecture isn't the best serving architecture.
- **Reproducibility.** Many design choices are justified only by internal ablations that aren't published.

## Solutions & Current Best Practices
The **2026 dense default**: decoder-only, pre-norm RMSNorm, SwiGLU, RoPE, GQA, no biases, untied embeddings at scale, AdamW (increasingly Muon for matrix parameters) + WSD/cosine. The **MoE default** (2026): fine-grained experts (128–896) with 1–2 shared experts, top-4 to top-16 routing, auxiliary-loss-free or low-coefficient balancing, expert/tensor parallelism, an MTP head, and increasingly Muon (with QK-Clip) plus 4-bit expert weights (MXFP4/NVFP4/INT4 QAT). Token mixers are now chosen per layer: hybrid attention–Mamba blocks (Jamba, Zamba, Nemotron-H/Nemotron 3), ~3:1 delta-rule hybrids, or learned sparse attention.

| Model (date) | Total / active | Experts (routed+shared) | Token mixer |
|---|---|---|---|
| gpt-oss-120b (2025-08) | 116.8B / 5.1B | 128, top-4 | GQA; alternating banded/dense; learned sinks |
| Kimi K2 (2025-07) | ~1T / 32B | 384, top-8 | MLA |
| Qwen3-Next (2025-09) | 80B / ~3B | 512 (10+1) | 3 Gated DeltaNet : 1 gated attention |
| MiniMax-M2 (2025-10) | ~230B / 10B | 256 | full attention |
| GLM-5 (2026-02) | 744B / 40B | — | DSA (DeepSeek Sparse Attention) |
| DeepSeek-V4-Pro (2026-04) | 1.6T / 49B | — | CSA + HCA (compressed sparse attention) |
| Gemma 4 (2026-04) | 26B-A4B MoE; 31B dense | — | — |
| Kimi K3 (2026-07) | 2.8T / 104B | 896 (16+2) | 69 KDA : 24 gated MLA (3:1) |
| Qwen3.8 open (2026-08) | 2.4T / 95B | 512 (10+1) | — |
| Mistral Large 4 (2026-10) | ~1.05T / 52B | — | — |

(—: not publicly specified or not verified. Most figures are vendor-reported; some come from secondary sources.)

## Lab Perspectives
- **OpenAI:** frontier models closed; GPT-4 widely believed to be MoE; o-series adds long reasoning rollouts. gpt-oss (Apache 2.0) is its open reference: 128-expert top-4 MoE with MXFP4 expert weights (~4.25 bits/param), so the 120b fits on one 80 GB GPU.
- **Google DeepMind:** Gemini is natively multimodal and MoE; Gemma 1–3 followed a clean dense recipe with GQA + RoPE, and Gemma 4 (Apache 2.0) adds a 26B-A4B MoE beside a 31B dense model and per-layer-embedding edge models.
- **Meta:** Llama 1–3 dense (standard-setting recipe); Llama 4 moved to MoE (Scout/Maverick/Behemoth); since Apr 2026 its frontier Muse models are closed, with architecture undisclosed.
- **DeepSeek:** distinctive MLA + fine-grained MoE + multi-token prediction (MTP) training objective, then DSA (V3.2) and CSA/HCA (V4, 1M context) with Muon and FP4 QAT on routed experts (reported).
- **Moonshot / Qwen / Zhipu / MiniMax:** the Chinese open frontier now leads on scale and attention design — KDA hybrids (Kimi), Gated DeltaNet hybrids with ultra-sparse 512-expert MoE (Qwen), DSA adoption (GLM-5), and an explicit reversal to full attention (MiniMax M2) before block-sparse MSA (M3).
- **NVIDIA:** Nemotron 3 is a Mamba-2/MoE hybrid with few attention layers, pretrained in NVFP4 (Super, Ultra), released with data and recipes.
- **Mistral:** sliding-window + GQA dense (7B) and MoE (Mixtral); efficiency-first. Mistral Large 3 (675B/41B "granular" MoE) and Large 4 (~1.05T/52B, open weights Oct 2026) are the largest Western open MoEs; Medium 3.5 is a 128B *dense* model.

## Latest Developments (2023–Nov 2026)
MoE has become mainstream at the frontier (DeepSeek-V3, Llama 4, Qwen-MoE, Mixtral). **Multi-token prediction** (predicting several future tokens) is used as an auxiliary objective (DeepSeek-V3, Meta research) for better data efficiency and as a basis for speculative decoding. **Attention–SSM hybrids** target long context. **MLA** spread as a KV-efficient alternative to GQA. Normalization research (QK-norm, OLMo-2's reordered norm) continues to chase billion-dollar-run stability.

**Mid-2025 → Nov 2026 update:**
- **Open MoEs crossed 1T and sparsity kept rising (2025-07 → 2026-10).** Kimi K2 (1T/32B; arXiv:2507.20534) → DeepSeek-V4-Pro (1.6T/49B) → Kimi K3 (2.8T/104B) → Qwen3.8 (2.4T/95B) → Mistral Large 4 (~1.05T/52B). Active parameters stay ~10–104B while expert counts grew to 512–896 with top-10/16 routing; Qwen3-Next activates ~3B of 80B.
- **Three competing token-mixer families.** (i) ~3:1 delta-rule hybrids (Qwen3-Next, Qwen3.5, Kimi Linear arXiv:2510.26692, K3); (ii) learned top-$k$ sparse attention (DSA arXiv:2512.02556 → V4 CSA/HCA; GLM-5 arXiv:2602.15763; MiniMax M3 MSA with 128-token blocks, top-16 per GQA group); (iii) sliding-window hybrids (gpt-oss; MiMo-V2.5 reportedly 5:1 SWA:global). MiniMax M2's return to full attention (multi-hop losses, low-precision state fragility) is the standing counterexample.
- **SSM hybrids at production scale.** Nemotron Nano 2 (2025-08; arXiv:2508.14444): 62 layers = 28 Mamba-2 + 28 FFN + 6 attention, 20T FP8 tokens, pruned 12B→9B, up to 6.3× throughput vs Qwen3-8B on reasoning workloads. Nemotron 3 (arXiv:2512.20856) interleaves Mamba-2 and MoE; Mamba-3 (arXiv:2603.15569, ICLR 2026 oral) adds complex-valued state transitions and MIMO decoding, matching Mamba-2 perplexity at half the state size.
- **Stability moved into gates, optimizers and residual geometry.** Output-gated attention (arXiv:2505.06708) removes attention sinks; QK-Clip in MuonClip prevents logit explosion (K2: zero loss spikes over 15.5T tokens); mHC (arXiv:2512.24880) constrains widened residual streams to doubly-stochastic mixing; K3 credits "SiTU-GLU", "Quantile Balancing" and Attention Residuals (reported, not independently verified).
- **Number format is now an architecture choice.** gpt-oss ships MXFP4 MoE weights; Kimi K2 Thinking uses INT4 weight-only QAT (≈2× generation speed); DeepSeek-V4 reportedly applies FP4 QAT to routed experts; K3 ships MXFP4 weights; Nemotron 3 Super/Ultra are *pretrained* in NVFP4 (arXiv:2604.12374).
- **Non-autoregressive variants.** LLaDA2.0 (arXiv:2512.15745) converts AR MoEs into block-diffusion LMs up to 100B with a warm-up → full-sequence → small-block schedule; a scaling comparison (arXiv:2602.15014) finds masked diffusion has the best likelihood scaling among diffusion variants, yet uniform-state diffusion does better on GSM8K.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What's in the 'modern LLM recipe' and why?"** Pre-norm RMSNorm, SwiGLU, RoPE, GQA — justify each.
- **"Dense vs MoE: when and why?"** Decoupling capacity from active compute; serving throughput vs memory footprint and routing complexity.
- **"What is MLA and how does it differ from GQA?"** Low-rank latent KV compression vs head sharing.
- **"Why parallel attention+FFN layers in PaLM?"** Fewer norms, better hardware utilization, negligible quality loss at scale.
- **"Why did open MoEs go from 8 experts/top-2 to 512–896 experts/top-10–16?"** Finer granularity gives more expert combinations per FLOP and better specialization; shared experts absorb common knowledge; the costs are all-to-all communication, routing stability and memory for total parameters (rack-scale NVLink domains).
- **"Name three 2026 routes to an affordable 1M-token context and their failure modes."** 3:1 linear hybrid (recall/multi-hop), learned sparse over compressed KV (indexer cost, extra caches), SWA hybrid (long-range loss in local layers).

## Open Problems
There is no first-principles theory for most of these choices — they are won by ablation. Open questions include the optimal expert granularity and routing in MoE (and the expert:attention compute split, arXiv:2603.10379), whether hybrid SSM/linear-attention designs will displace pure attention or plateau on multi-hop reasoning, the right depth/width law, how far 4-bit weights can go without quality loss, and how to make multi-trillion-parameter runs stable without heuristic norms. The "best architecture" likely depends on the deployment regime, which the field is only beginning to formalize.

## References
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.
- Lieber, O. et al. (2024). *Jamba.* arXiv:2403.19887.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- NVIDIA (2025). *NVIDIA Nemotron Nano 2.* arXiv:2508.14444.
- DeepSeek-AI (2025). *DeepSeek-V3.2* (DeepSeek Sparse Attention). arXiv:2512.02556.
- Zhipu AI / Z.ai (2026). *GLM-5.* arXiv:2602.15763.
- Gu, A., Dao, T. et al. (2026). *Mamba-3.* arXiv:2603.15569.
