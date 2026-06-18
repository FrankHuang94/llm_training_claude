# Model Architecture Variants

> **Last Updated:** 2026-06-18
> **Related Files:** [Transformer Architecture](00_transformer_architecture.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md) · [Attention Mechanisms](01_attention_mechanisms.md)
> **Key Papers:** Chowdhery et al. 2022 PaLM ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) · Touvron et al. 2023 LLaMA 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · Jiang et al. 2024 Mixtral ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)) · DeepSeek-AI 2024 DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))

## Overview
Although nearly all frontier LLMs share the decoder-only Transformer skeleton, they differ in a set of well-defined "knobs": normalization type and placement, activation function, attention variant, positional scheme, dense-vs-MoE FFNs, and various stability tricks. These choices are not cosmetic — they materially affect training stability, inference cost, and quality-per-FLOP. This file catalogs the design space and the converged "modern recipe," then maps how flagship model families instantiate it.

The story of 2020→2026 is a gradual standardization (pre-norm + RMSNorm + RoPE + SwiGLU + GQA) followed by a re-diversification driven by efficiency: MoE FFNs, latent attention (MLA), and attention–SSM hybrids.

## Core Concepts
**Normalization.** Original Transformers used **post-norm** LayerNorm (unstable at depth). Modern models use **pre-norm** ($x+\text{Sublayer}(\text{Norm}(x))$) for stable gradients, almost always with **RMSNorm** (no mean subtraction). Some very large runs add **QK-norm** (normalize queries/keys before attention) and **sandwich/double norm** for stability.

**Activations.** ReLU → GeLU (BERT/GPT-2/3) → **gated linear units**, especially **SwiGLU** (LLaMA, PaLM, Mistral). Gated activations consistently improve quality per parameter; the FFN hidden dim is reduced to ~$\frac{8}{3}d$ to hold parameter count constant given the extra gate matrix.

**Attention variants.** MHA → **MQA** (PaLM) → **GQA** (LLaMA-2 70B+, Mistral) for KV-cache reduction; **sliding-window** (Mistral); **MLA** (DeepSeek) for low-rank KV compression. See [Attention Mechanisms](01_attention_mechanisms.md).

**Dense vs MoE FFN.** Dense models activate all parameters per token. **Sparse MoE** replaces the FFN with $E$ experts and a router selecting top-$k$ (usually 2), so active parameters $\ll$ total. Mixtral 8×7B (47B total, ~13B active), DeepSeek-V3 (671B total, 37B active), Llama 4 (MoE) embody this. MoE decouples capacity from compute — see [MoE](../06_efficiency/05_mixture_of_experts.md).

**Parallel layers.** PaLM computes attention and FFN from the *same* normalized input in parallel (rather than sequentially), saving a normalization and improving TPU utilization.

**Depth vs width.** Given a parameter budget, aspect ratio (layers vs $d_{model}$) trades expressivity for parallelism; deeper-narrower helps reasoning but hurts hardware efficiency and stability.

## Key Challenges
- **Stability at scale.** Some choices (post-norm, certain activations, large LR) cause loss spikes; mitigations interact non-trivially.
- **MoE instability and load imbalance.** Routers can collapse to a few experts; require auxiliary load-balancing losses or loss-free balancing.
- **Inference vs training optimality diverge.** GQA/MQA/MoE help inference but complicate training; the best training architecture isn't the best serving architecture.
- **Reproducibility.** Many design choices are justified only by internal ablations that aren't published.

## Solutions & Current Best Practices
The **2026 dense default**: decoder-only, pre-norm RMSNorm, SwiGLU, RoPE, GQA, no biases, untied embeddings at scale, AdamW + WSD/cosine. The **MoE default**: fine-grained experts with shared experts (DeepSeek), top-2 routing, auxiliary-loss-free or low-coefficient balancing, and expert/tensor parallelism. Hybrid attention–Mamba blocks (Jamba, Zamba, Nemotron-H) are emerging for long-context efficiency.

## Lab Perspectives
- **OpenAI:** closed; GPT-4 widely believed to be MoE; o-series adds long reasoning rollouts.
- **Google DeepMind:** Gemini is natively multimodal and MoE; Gemma open models follow a clean dense recipe with GQA + RoPE.
- **Meta:** Llama 1–3 dense (standard-setting recipe); Llama 4 moved to MoE (Scout/Maverick/Behemoth).
- **DeepSeek:** distinctive MLA + fine-grained MoE + multi-token prediction (MTP) training objective.
- **Mistral:** sliding-window + GQA dense (7B) and MoE (Mixtral); efficiency-first.

## Latest Developments (2023–2026)
MoE has become mainstream at the frontier (DeepSeek-V3, Llama 4, Qwen-MoE, Mixtral). **Multi-token prediction** (predicting several future tokens) is used as an auxiliary objective (DeepSeek-V3, Meta research) for better data efficiency and as a basis for speculative decoding. **Attention–SSM hybrids** target long context. **MLA** spread as a KV-efficient alternative to GQA. Normalization research (QK-norm, OLMo-2's reordered norm) continues to chase billion-dollar-run stability.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What's in the 'modern LLM recipe' and why?"** Pre-norm RMSNorm, SwiGLU, RoPE, GQA — justify each.
- **"Dense vs MoE: when and why?"** Decoupling capacity from active compute; serving throughput vs memory footprint and routing complexity.
- **"What is MLA and how does it differ from GQA?"** Low-rank latent KV compression vs head sharing.
- **"Why parallel attention+FFN layers in PaLM?"** Fewer norms, better hardware utilization, negligible quality loss at scale.

## Open Problems
There is no first-principles theory for most of these choices — they are won by ablation. Open questions include the optimal expert granularity and routing in MoE, whether hybrid SSM-attention will displace pure attention, the right depth/width law, and how to make trillion-parameter runs stable without heuristic norms. The "best architecture" likely depends on the deployment regime, which the field is only beginning to formalize.

## References
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.
- Lieber, O. et al. (2024). *Jamba.* arXiv:2403.19887.
