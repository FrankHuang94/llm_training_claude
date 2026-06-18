# Speculative Decoding

> **Last Updated:** 2026-06-18
> **Related Files:** [Knowledge Distillation](03_knowledge_distillation.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Quantization Techniques](02_quantization_techniques.md)
> **Key Papers:** Leviathan et al. 2023 Speculative Decoding ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) · Chen et al. 2023 Speculative Sampling (DeepMind) ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)) · Cai et al. 2024 Medusa ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)) · Li et al. 2024 EAGLE ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077))

## Overview
Autoregressive decoding is fundamentally **sequential and memory-bandwidth-bound**: each token requires a full forward pass that reads all model weights from HBM, but uses the GPU's compute only lightly (batch-of-one). Speculative decoding breaks this bottleneck by exploiting a key asymmetry: **verifying** several candidate tokens in parallel is nearly as cheap as generating one (both are one forward pass, and the pass is bandwidth-bound, so processing $k$ tokens at once costs almost the same as one). A small **draft** model proposes several tokens; the large **target** model verifies them in a single parallel pass; accepted tokens are kept. Crucially, this yields a **1.5–3× speedup with *identical* output distribution** — a lossless acceleration, which is why it's deployed everywhere.

## Core Concepts
**The draft-then-verify loop.**
1. A cheap **draft** model autoregressively proposes $\gamma$ candidate tokens $\hat x_{1..\gamma}$.
2. The **target** model runs **one forward pass** over all $\gamma$ candidates in parallel, yielding its true distributions $p(x_i\mid \cdot)$ at each position.
3. **Verification / acceptance:** accept each drafted token with a probability that guarantees the final samples are distributed *exactly* as if drawn from the target model. On the first rejection, **resample** that token from a corrected residual distribution and discard the rest.

**Why it's exact (rejection sampling).** For draft distribution $q$ and target $p$, accept token $x$ with probability $\min(1, p(x)/q(x))$; on rejection, sample from the normalized residual $(p-q)_+$. This **modified rejection sampling** provably produces samples from $p$ — the speedup is "free" (lossless) in expectation. Expected accepted tokens per step depends on how well $q$ approximates $p$ (the **acceptance rate**); the more aligned the draft, the bigger the speedup. Greedy decoding is the special case of accepting while draft==target argmax.

**Variants (eliminating/improving the draft).**
- **Self-drafting / Medusa** (Cai et al.): instead of a separate draft model, add multiple **extra decoding heads** to the target model that predict several future tokens at once; verify with a tree-structured attention. No separate model to train/serve; integrates with the base.
- **EAGLE** (Li et al.): draft at the **feature (hidden-state) level** with a lightweight autoregressive head, achieving high acceptance and large speedups; EAGLE-2/3 add dynamic draft trees. A current SOTA approach.
- **Lookahead decoding** (Fu et al.): draft-free; uses Jacobi/n-gram parallelism to generate and verify n-grams.
- **Tree/batched verification (SpecInfer):** verify a *tree* of candidate continuations to raise accepted-tokens-per-step.
- **Self-speculative / layer-skip:** use a subset of the model's own layers as the draft.

**Multi-token prediction synergy.** Models trained with **MTP heads** (DeepSeek-V3) come with a built-in draft mechanism — pretraining and inference acceleration co-designed (see [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md)).

## Key Challenges
- **Draft-target alignment.** Low acceptance rate (poor draft) kills the speedup; the draft must match the target's distribution well on the actual workload.
- **Draft model cost/training.** A separate draft adds memory and must be trained/distilled and kept in sync with the target.
- **Batching tension.** Speculative decoding helps most at **small batch** (latency-bound, bandwidth-bound); at **large batch** (throughput-bound, compute-saturated) the parallel-verify advantage shrinks and it can even hurt.
- **Tree/verification overhead.** Tree verification raises acceptance but adds compute and complexity.
- **Acceptance variance.** Hard/uncertain regions (reasoning, rare tokens) have lower acceptance.

## Solutions & Current Best Practices
Use **EAGLE-2/3 or Medusa** for self-drafting (no separate model) in latency-sensitive serving; a **distilled small draft** of the same family works well when available. Tune the **draft length $\gamma$** to the acceptance rate (longer drafts help when acceptance is high). Deploy via inference engines (vLLM, TensorRT-LLM, SGLang) that implement speculative decoding + tree verification. Combine with **quantization** (quantized target) and **continuous batching**; disable or adapt speculation at large batch sizes where it doesn't pay. For reasoning models with long outputs, speculative decoding is especially valuable (long sequences amplify per-token savings).

## Lab Perspectives
- **Google DeepMind/Google** (Leviathan, Chen) introduced speculative decoding/sampling; widely used in serving.
- **OpenAI/Anthropic:** use speculative-style acceleration in production serving (details proprietary); "predicted outputs" features resemble speculative verification.
- **DeepSeek:** **MTP**-trained models provide native draft heads (V3) — co-designed pretraining + inference speedup.
- **Open ecosystem:** Medusa, EAGLE, vLLM/TensorRT-LLM/SGLang ship speculative decoding broadly.

## Latest Developments (2023–2026)
**EAGLE-2/3** and dynamic **draft trees** pushed speedups to ~3–5× in favorable settings. **Self-drafting** (Medusa/EAGLE) removed the separate-draft burden. **MTP-as-draft** co-design (DeepSeek-V3) tied training objective to inference acceleration. Active work on **large-batch** speculative decoding, speculation for **reasoning/long-output** models, and combining speculation with quantization and disaggregated (prefill/decode-split) serving.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does speculative decoding speed up inference without changing outputs?"** Decoding is bandwidth-bound; verifying $k$ tokens in one parallel pass ≈ cost of one; modified rejection sampling guarantees the target distribution → lossless.
- **"Explain the acceptance/verification step."** Accept with $\min(1,p/q)$; on reject, resample from $(p-q)_+$ — provably samples from target $p$.
- **"What determines the speedup?"** Acceptance rate (draft-target alignment) × draft length, minus overhead.
- **"Medusa/EAGLE vs a separate draft model?"** Self-drafting heads/features avoid a separate model and improve acceptance.
- **"When does speculative decoding NOT help?"** Large-batch/compute-bound regimes; poorly aligned draft.

## Open Problems
Maximizing acceptance rate (better, cheap drafts) across diverse and **reasoning-heavy** workloads, making speculative decoding pay at **high batch sizes**, and optimally co-designing training (MTP) with inference speculation are open. Theoretical limits of multi-token parallelism in inherently sequential generation remain an interesting question.

## References
- Leviathan, Y. et al. (2023). *Fast Inference via Speculative Decoding.* arXiv:2211.17192.
- Chen, C. et al. (2023). *Accelerating LLM Decoding with Speculative Sampling.* arXiv:2302.01318.
- Cai, T. et al. (2024). *Medusa.* arXiv:2401.10774.
- Li, Y. et al. (2024). *EAGLE.* arXiv:2401.15077.
- Gloeckle, F. et al. (2024). *Multi-Token Prediction.* arXiv:2404.19737.
