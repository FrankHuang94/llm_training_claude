# Training Stability

> **Last Updated:** 2026-11-04
> **Related Files:** [Optimizer Choices](03_optimizer_choices.md) · [Distributed Training](01_distributed_training.md) · [Mixed Precision & Quantization](04_mixed_precision_and_quantization.md)
> **Key Papers:** Yang et al. 2022 muP/μTransfer ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466)) · Chowdhery et al. 2022 PaLM (loss spikes) ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) · Zhang et al. 2022 OPT logbook ([arXiv:2205.01068](https://arxiv.org/abs/2205.01068)) · Wortsman et al. 2023 Small-scale proxies for stability ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322)) · Kimi Team 2025 Kimi K2 (MuonClip/QK-Clip) ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534))

## Overview
At frontier scale a single training run can cost tens of millions of dollars and take months, so **stability** — avoiding divergence, loss spikes, and silent degradation — is mission-critical. Instabilities that are invisible at small scale (a 1B model trains fine) can derail a 100B+ run. The published "logbooks" (OPT, BLOOM, GLM-130B) read like incident reports: spontaneous loss spikes, NaNs, hardware-induced bit flips, and the manual interventions (rollback, skip batches, lower LR) used to recover. Understanding the *causes* of instability and the *preventive* architectural/optimization choices is essential for any pretraining role — and since 2025 the same discipline extends to long **RL post-training** runs.

## Core Concepts
**Loss spikes.** Sudden jumps in training loss, sometimes recoverable, sometimes terminal. Causes include: large gradient norms hitting a sharp loss-landscape region, attention-logit growth (softmax saturation), numerical overflow in fp16, and correlations with specific data batches. The growth of **attention logits** ($QK^\top$ magnitude exploding) and **output logits** are common culprits.

**Gradient clipping.** Cap the global gradient norm: if $\|g\| > c$, rescale $g \leftarrow g\cdot c/\|g\|$. The standard first line of defense against spikes; $c=1.0$ is typical. Per-layer or adaptive clipping variants exist.

**Learning-rate schedule.** **Warmup** (linearly ramp LR over the first ~thousands of steps) avoids early instability when gradients are large and Adam's second-moment estimate is immature. **Cooldown/decay** (cosine or WSD) reduces LR late to settle into a minimum (see [Optimizers](03_optimizer_choices.md)).

**Initialization & μP.** Scaled initialization (e.g., $1/\sqrt{d}$, scaling residual branches by $1/\sqrt{2L}$) controls activation/gradient variance with depth. **Maximal Update Parametrization (μP)** (Yang et al.) parametrizes init and per-layer LR so that *optimal hyperparameters are invariant to width* — enabling **μTransfer**: tune HPs on a small model, transfer to the large one without re-tuning. This de-risks expensive runs.

**Numerical stabilizers.**
- **z-loss**: add $\lambda(\log Z)^2$ penalizing the softmax log-partition $Z$, keeping logits bounded (PaLM, used widely).
- **QK-norm**: apply LayerNorm/RMSNorm to queries and keys before attention, capping logit growth (helps very large/long-context runs).
- **QK-Clip** (Kimi K2): a weight-space alternative. After each optimizer step, for each head $h$ whose max pre-softmax logit $S^h_{\max}$ exceeds a threshold $\tau$, rescale $W_q^h, W_k^h$ by $\sqrt{\gamma_h}$ with $\gamma_h=\tau/S^h_{\max}$, so the logit bound is restored without touching the forward pass. It suits MLA, where per-head keys are not materialized at inference and QK-norm is awkward.
- **Output (sigmoid) gating** after attention: removes attention sinks and massive activations and tolerates larger LRs (Qwen gated attention).
- **bf16 over fp16**: wider dynamic range avoids overflow (see [Mixed Precision](04_mixed_precision_and_quantization.md)).
- **Embedding/logit soft-capping** (Gemma 2): tanh-cap logits to a fixed range.

## Key Challenges
- **Scale-dependent instabilities.** Pathologies that don't appear below ~10B params make small-scale debugging unreliable.
- **Forensics are hard.** Distinguishing a bad batch, a hardware fault (silent data corruption / bit flip), and a genuine optimization instability is nontrivial mid-run.
- **Hardware faults at scale.** With 10k+ GPUs, SDC (silent data corruption), ECC errors, and NCCL hangs are routine and can masquerade as model instability.
- **Costly interventions.** Rollback to a checkpoint wastes compute; skipping batches or lowering LR may not fix the root cause.
- **Optimizer-induced logit growth.** Faster optimizers (Muon) and wider residual streams (Hyper-Connections) open new routes to attention-logit or signal blow-up that AdamW-era recipes did not need to handle.
- **RL-specific instability.** Entropy collapse and trainer/rollout numerical mismatch make long RL runs drift or collapse, especially for MoE policies.

## Solutions & Current Best Practices
**Prevent**: bf16, μP/scaled init, residual scaling, LR warmup, gradient clipping, z-loss, and (for big/long runs) QK-norm or QK-Clip, output gating and logit soft-capping. **Detect**: monitor gradient norm, loss, per-head max attention logit, and per-layer activation statistics; use **small-scale proxies** (Wortsman et al. 2023) that reproduce instabilities cheaply to pick stable configs. **Recover**: checkpoint frequently, and on a spike, **roll back and skip the offending data batches** (PaLM's documented recipe) or briefly lower LR. Deterministic, resumable data loading is essential. **For RL**: match trainer and rollout numerics (FP16 or batch-invariant kernels, FP32 LM-head logits), monitor policy entropy, and use sequence-level ratios for MoE policies.

## Lab Perspectives
- **Google** documented loss spikes in PaLM and the skip-batch recovery; popularized z-loss and μP-style scaling.
- **Meta (OPT)** published a candid logbook of instabilities and manual interventions — a foundational reference.
- **DeepMind/Microsoft** advanced μP/μTransfer for hyperparameter transfer.
- **Moonshot** introduced QK-Clip (MuonClip) and reported a 15.5T-token, 1T-parameter run with zero loss spikes; Kimi K3's report reportedly credits "SiTU-GLU" and "Quantile Balancing" for stability at 2.8T.
- **Qwen** shipped output-gated attention (Qwen3-Next); **DeepSeek** proposed manifold-constrained hyper-connections (mHC) to stabilize widened residual streams.
- **OpenAI/Anthropic** keep recipes closed but clearly invest heavily in stability tooling and monitoring; Anthropic has written on the importance of reproducible, deterministic infrastructure.

## Latest Developments (2023–Nov 2026)
**μTransfer** is increasingly standard for de-risking HP choices. **QK-norm and logit soft-capping** (Gemma 2, OLMo 2) became common stabilizers. **fp8 training** (H100) reintroduced dynamic-range stability concerns, addressed via per-tensor scaling (Transformer Engine). Better **silent-data-corruption detection** and elastic recovery matured for 100k-GPU-class runs. "Spike-free" recipes and small-proxy stability prediction are active engineering areas.

**Mid-2025 → Nov 2026 update:**
- **QK-Clip makes Muon safe at 1T (2025-07).** Kimi K2's MuonClip rescales $W_q, W_k$ per head whenever the max attention logit exceeds a threshold, preventing logit explosion across 15.5T tokens of a 1T-parameter MoE with no loss spikes ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)).
- **Gated attention removes sinks (2025-05, NeurIPS 2025).** A head-specific sigmoid gate after SDPA adds non-linearity to the low-rank V–O path and input-dependent sparsity; first-token attention falls from 46.7% to 4.8%, larger LRs become stable, and RULER improves by >10 points ([arXiv:2505.06708](https://arxiv.org/abs/2505.06708)).
- **mHC for widened residual streams (2025-12).** Hyper-Connections break the identity mapping and destabilize training; projecting the stream-mixing matrices onto doubly-stochastic matrices (Birkhoff polytope, via Sinkhorn–Knopp) restores stable signal propagation at 27B ([arXiv:2512.24880](https://arxiv.org/abs/2512.24880)). Its use in DeepSeek-V4 is reported but disputed.
- **4-bit stability recipes (2025-09 → 2026-04).** NVFP4 pretraining stays within ~FP8 quality only with random Hadamard transforms on Wgrad inputs, stochastic rounding, and a few sensitive layers kept in BF16 ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)); in Nemotron 3, Mamba output projections had to stay in MXFP8 because of flush-to-zero.
- **RL numerics as a stability problem (2025-09/10).** BF16's 7-bit mantissa makes trainer and rollout-engine log-probs diverge; switching RL to FP16 largely removes the mismatch ([arXiv:2510.26788](https://arxiv.org/abs/2510.26788)). ScaleRL's recipe keeps LM-head logits in FP32 ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786)); GSPO's sequence-level importance ratio fixes token-level IS noise that collapsed large-MoE RL ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)).
- **Entropy collapse is predictable (2025-05).** Without intervention RL trades entropy for reward as $R=-a\,e^{H}+b$, driven by the covariance between log-probability and advantage; Clip-Cov / KL-Cov target the high-covariance tokens ([arXiv:2505.22617](https://arxiv.org/abs/2505.22617)).
- **Massive activations in hybrids (2026-08, single source).** Controlled Gated-DeltaNet hybrid pretraining up to 1.3B reports activation spikes clustering just before the full-attention layers ([arXiv:2608.12149](https://arxiv.org/abs/2608.12149)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"What causes training loss spikes and how do you handle them?"** Logit/gradient growth, bad batches, fp16 overflow; clip, z-loss, QK-norm, rollback + skip-batch.
- **"What is μP / μTransfer and why does it matter?"** Width-invariant optimal HPs → tune small, transfer large; de-risks big runs.
- **"Why bf16 over fp16 for pretraining?"** Wider exponent range avoids overflow without loss scaling.
- **"How do you tell a hardware fault from an optimization instability?"** Reproducibility checks, gradient/activation monitoring, ECC/SDC logs.
- **"QK-norm vs QK-Clip — why did Kimi need the latter?"** QK-norm normalizes activations in the forward pass (awkward with MLA's non-materialized keys); QK-Clip caps the max logit by rescaling weights after the update, directly targeting Muon-induced logit growth.
- **"Why would you run RL in FP16 when pretraining uses BF16?"** RL's importance ratios depend on trainer/sampler log-prob agreement; FP16's extra mantissa bits shrink the mismatch, and gradients are well-scaled after pretraining so range is less critical.

## Open Problems
There is no predictive *theory* of loss spikes — mitigations are largely empirical. Reliably forecasting large-scale instabilities from small proxies is improving but unsolved. As runs scale to 100k+ accelerators and lower precision (fp8/fp4), the interaction of numerical precision, optimizer dynamics (Muon-family), architecture (hybrids, widened residuals, extreme MoE sparsity), and hardware faults creates new, poorly-understood failure modes; RL-stage stability (entropy, mismatch, MoE routing drift) lacks a unified treatment.

## References
- Yang, G. et al. (2022). *Tensor Programs V: μTransfer.* arXiv:2203.03466.
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Zhang, S. et al. (2022). *OPT (logbook).* arXiv:2205.01068.
- Wortsman, M. et al. (2023). *Small-scale Proxies for Large-scale Transformer Training Instabilities.* arXiv:2309.14322.
- Dehghani, M. et al. (2023). *ViT-22B (QK-norm).* arXiv:2302.05442.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence (MuonClip).* arXiv:2507.20534.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- DeepSeek-AI (2025). *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880.
- Cui, G. et al. (2025). *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* arXiv:2505.22617.
