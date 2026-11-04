# Speculative Decoding

> **Last Updated:** 2026-11-04
> **Related Files:** [Knowledge Distillation](03_knowledge_distillation.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Quantization Techniques](02_quantization_techniques.md)
> **Key Papers:** Leviathan et al. 2023 Speculative Decoding ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) · Chen et al. 2023 Speculative Sampling (DeepMind) ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)) · Cai et al. 2024 Medusa ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)) · Li et al. 2024 EAGLE ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077)) · Li et al. 2025 EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840))

## Overview
Autoregressive decoding is fundamentally **sequential and memory-bandwidth-bound**: each token requires a full forward pass that reads all model weights from HBM, but uses the GPU's compute only lightly (batch-of-one). Speculative decoding breaks this bottleneck by exploiting a key asymmetry: **verifying** several candidate tokens in parallel is nearly as cheap as generating one (both are one forward pass, and the pass is bandwidth-bound, so processing $k$ tokens at once costs almost the same as one). A small **draft** model proposes several tokens; the large **target** model verifies them in a single parallel pass; accepted tokens are kept. Crucially, this yields a **1.5–3× speedup (higher with modern drafters) with *identical* output distribution** — a lossless acceleration, which is why it's deployed everywhere. Long reasoning outputs and agentic workloads have made it more valuable, not less.

## Core Concepts
**The draft-then-verify loop.**
1. A cheap **draft** model autoregressively proposes $\gamma$ candidate tokens $\hat x_{1..\gamma}$.
2. The **target** model runs **one forward pass** over all $\gamma$ candidates in parallel, yielding its true distributions $p(x_i\mid \cdot)$ at each position.
3. **Verification / acceptance:** accept each drafted token with a probability that guarantees the final samples are distributed *exactly* as if drawn from the target model. On the first rejection, **resample** that token from a corrected residual distribution and discard the rest.

**Why it's exact (rejection sampling).** For draft distribution $q$ and target $p$, accept token $x$ with probability $\min(1, p(x)/q(x))$; on rejection, sample from the normalized residual $(p-q)_+$. This **modified rejection sampling** provably produces samples from $p$ — the speedup is "free" (lossless) in expectation. Expected accepted tokens per step depends on how well $q$ approximates $p$ (the **acceptance rate**); the more aligned the draft, the bigger the speedup. Greedy decoding is the special case of accepting while draft==target argmax.

**Variants (eliminating/improving the draft).**
- **Self-drafting / Medusa** (Cai et al.): instead of a separate draft model, add multiple **extra decoding heads** to the target model that predict several future tokens at once; verify with a tree-structured attention. No separate model to train/serve; integrates with the base.
- **EAGLE** (Li et al.): draft at the **feature (hidden-state) level** with a lightweight autoregressive head, achieving high acceptance and large speedups; EAGLE-2 adds dynamic draft trees. **EAGLE-3** drops feature regression for direct token prediction, fuses low/mid/high-level target features, and trains with "training-time test" (simulated multi-step drafting) — a current SOTA approach.
- **Lookahead decoding** (Fu et al.): draft-free; uses Jacobi/n-gram parallelism to generate and verify n-grams.
- **Tree/batched verification (SpecInfer):** verify a *tree* of candidate continuations to raise accepted-tokens-per-step.
- **Self-speculative / layer-skip:** use a subset of the model's own layers as the draft.

**Multi-token prediction synergy.** Models trained with **MTP heads** (DeepSeek-V3, and now Qwen3-Next, GLM-4.5/5) come with a built-in draft mechanism — pretraining and inference acceleration co-designed (see [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md)).

## Key Challenges
- **Draft-target alignment.** Low acceptance rate (poor draft) kills the speedup; the draft must match the target's distribution well on the actual workload.
- **Draft model cost/training.** A separate draft adds memory and must be trained/distilled and kept in sync with the target.
- **Batching tension.** Speculative decoding helps most at **small batch** (latency-bound, bandwidth-bound); at **large batch** (throughput-bound, compute-saturated) the parallel-verify advantage shrinks and it can even hurt.
- **Tree/verification overhead.** Tree verification raises acceptance but adds compute and complexity.
- **Acceptance variance.** Hard/uncertain regions (reasoning, rare tokens) have lower acceptance.
- **Stateful architectures.** In linear-attention/SSM hybrids, rejected tokens have already updated the recurrent state, so verification needs state checkpointing or recomputation rather than simple KV truncation.

## Solutions & Current Best Practices
Use **EAGLE-3, native MTP heads, or Medusa** for self-drafting (no separate model) in latency-sensitive serving; a **distilled small draft** of the same family works well when available. Train drafters on **more data** — with EAGLE-3 the speedup keeps scaling with draft training data. Tune the **draft length $\gamma$** to the acceptance rate (longer drafts help when acceptance is high). Deploy via inference engines (vLLM, TensorRT-LLM, SGLang) that implement speculative decoding + tree verification. Combine with **quantization** (quantized target) and **continuous batching**; disable or adapt speculation at large batch sizes where it doesn't pay. For reasoning models with long outputs, speculative decoding is especially valuable (long sequences amplify per-token savings).

## Lab Perspectives
- **Google DeepMind/Google** (Leviathan, Chen) introduced speculative decoding/sampling; widely used in serving; also previewed **Gemini Diffusion** (2025-05), a parallel-generation alternative.
- **OpenAI/Anthropic:** use speculative-style acceleration in production serving (details proprietary); "predicted outputs" features resemble speculative verification. OpenAI also pursues raw low-latency hardware (GPT-5.3-Codex-Spark on Cerebras WSE-3 at >1,000 tok/s).
- **DeepSeek:** **MTP**-trained models provide native draft heads (V3) — co-designed pretraining + inference speedup; **Qwen** and **Zhipu** adopted the same MTP modules.
- **Open ecosystem:** Medusa, EAGLE(-3), vLLM/TensorRT-LLM/SGLang ship speculative decoding broadly.
- **Inception Labs / inclusionAI:** bet on **diffusion LMs**, which decode many tokens per step natively.

## Latest Developments (2023–Nov 2026)
**EAGLE-2/3** and dynamic **draft trees** pushed speedups to ~3–5× in favorable settings. **Self-drafting** (Medusa/EAGLE) removed the separate-draft burden. **MTP-as-draft** co-design (DeepSeek-V3) tied training objective to inference acceleration. Active work on **large-batch** speculative decoding, speculation for **reasoning/long-output** models, and combining speculation with quantization and disaggregated (prefill/decode-split) serving.

**Mid-2025 → Nov 2026 update:**
- **EAGLE-3 (2025-03, NeurIPS 2025).** Removing the feature-regression constraint and predicting tokens directly, with multi-layer feature fusion and training-time test, makes speedup **scale with draft training data**: up to 6.5× speedup, and 1.38× throughput even at batch 64 in SGLang — evidence that speculation can pay at moderate batch sizes ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)).
- **MTP modules as default drafts (2025–26).** DeepSeek-V3-style MTP modules ship in Qwen3-Next and GLM-4.5/5, used both as a training signal and as a native speculative draft — no separate drafter to train or keep in sync.
- **Retrofitting MTP via self-distillation (2026-02).** A pretrained NTP model is converted into a standalone MTP decoder, reporting >3× faster decoding on GSM8K at <5% accuracy loss ([arXiv:2602.06019](https://arxiv.org/abs/2602.06019)) — note this is *lossy*, unlike verified speculation.
- **Diffusion LMs as the "draft-free" alternative (2025-06 → 2026-02).** Mercury predicts many tokens per step ([arXiv:2506.17298](https://arxiv.org/abs/2506.17298)); Mercury 2 claims ~1,000 tok/s on Blackwell (self-reported); LLaDA2.0 adds a confidence loss to enable aggressive parallel decoding at 100B ([arXiv:2512.15745](https://arxiv.org/abs/2512.15745)).
- **Serving architecture shifts (2025-09 → 2026-05).** Prefill/decode disaggregation (NVIDIA Rubin CPX for prefill) and attention–FFN disaggregation for MoE ([arXiv:2605.28302](https://arxiv.org/abs/2605.28302)) change where verification compute lands; speculation must be co-designed with these splits.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does speculative decoding speed up inference without changing outputs?"** Decoding is bandwidth-bound; verifying $k$ tokens in one parallel pass ≈ cost of one; modified rejection sampling guarantees the target distribution → lossless.
- **"Explain the acceptance/verification step."** Accept with $\min(1,p/q)$; on reject, resample from $(p-q)_+$ — provably samples from target $p$.
- **"What determines the speedup?"** Acceptance rate (draft-target alignment) × draft length, minus overhead.
- **"Medusa/EAGLE vs a separate draft model?"** Self-drafting heads/features avoid a separate model and improve acceptance.
- **"When does speculative decoding NOT help?"** Large-batch/compute-bound regimes; poorly aligned draft.
- **"What did EAGLE-3 change, and why does it now scale with data?"** Feature-regression loss constrained the drafter to imitate hidden states; predicting tokens directly from fused multi-layer features, trained with simulated multi-step drafting, removes that bottleneck so more draft data keeps raising acceptance.

## Open Problems
Maximizing acceptance rate (better, cheap drafts) across diverse and **reasoning-heavy** workloads, making speculative decoding pay at **high batch sizes**, speculation for **recurrent/hybrid** architectures, and optimally co-designing training (MTP) with inference speculation are open. Whether diffusion or other parallel decoders can match AR quality — making speculation unnecessary — and the theoretical limits of multi-token parallelism in inherently sequential generation remain interesting questions.

## References
- Leviathan, Y. et al. (2023). *Fast Inference via Speculative Decoding.* arXiv:2211.17192.
- Chen, C. et al. (2023). *Accelerating LLM Decoding with Speculative Sampling.* arXiv:2302.01318.
- Cai, T. et al. (2024). *Medusa.* arXiv:2401.10774.
- Li, Y. et al. (2024). *EAGLE.* arXiv:2401.15077.
- Gloeckle, F. et al. (2024). *Multi-Token Prediction.* arXiv:2404.19737.
- Li, Y. et al. (2025). *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test.* arXiv:2503.01840.
- Inception Labs (2025). *Mercury: Ultra-Fast Language Models Based on Diffusion.* arXiv:2506.17298.
