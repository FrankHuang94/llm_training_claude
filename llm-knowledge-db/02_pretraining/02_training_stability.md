# Training Stability

> **Last Updated:** 2026-06-18
> **Related Files:** [Optimizer Choices](03_optimizer_choices.md) · [Distributed Training](01_distributed_training.md) · [Mixed Precision & Quantization](04_mixed_precision_and_quantization.md)
> **Key Papers:** Yang et al. 2022 muP/μTransfer ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466)) · Chowdhery et al. 2022 PaLM (loss spikes) ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) · Zhang et al. 2022 OPT logbook ([arXiv:2205.01068](https://arxiv.org/abs/2205.01068)) · Wortsman et al. 2023 Small-scale proxies for stability ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322))

## Overview
At frontier scale a single training run can cost tens of millions of dollars and take months, so **stability** — avoiding divergence, loss spikes, and silent degradation — is mission-critical. Instabilities that are invisible at small scale (a 1B model trains fine) can derail a 100B+ run. The published "logbooks" (OPT, BLOOM, GLM-130B) read like incident reports: spontaneous loss spikes, NaNs, hardware-induced bit flips, and the manual interventions (rollback, skip batches, lower LR) used to recover. Understanding the *causes* of instability and the *preventive* architectural/optimization choices is essential for any pretraining role.

## Core Concepts
**Loss spikes.** Sudden jumps in training loss, sometimes recoverable, sometimes terminal. Causes include: large gradient norms hitting a sharp loss-landscape region, attention-logit growth (softmax saturation), numerical overflow in fp16, and correlations with specific data batches. The growth of **attention logits** ($QK^\top$ magnitude exploding) and **output logits** are common culprits.

**Gradient clipping.** Cap the global gradient norm: if $\|g\| > c$, rescale $g \leftarrow g\cdot c/\|g\|$. The standard first line of defense against spikes; $c=1.0$ is typical. Per-layer or adaptive clipping variants exist.

**Learning-rate schedule.** **Warmup** (linearly ramp LR over the first ~thousands of steps) avoids early instability when gradients are large and Adam's second-moment estimate is immature. **Cooldown/decay** (cosine or WSD) reduces LR late to settle into a minimum (see [Optimizers](03_optimizer_choices.md)).

**Initialization & μP.** Scaled initialization (e.g., $1/\sqrt{d}$, scaling residual branches by $1/\sqrt{2L}$) controls activation/gradient variance with depth. **Maximal Update Parametrization (μP)** (Yang et al.) parametrizes init and per-layer LR so that *optimal hyperparameters are invariant to width* — enabling **μTransfer**: tune HPs on a small model, transfer to the large one without re-tuning. This de-risks expensive runs.

**Numerical stabilizers.**
- **z-loss**: add $\lambda(\log Z)^2$ penalizing the softmax log-partition $Z$, keeping logits bounded (PaLM, used widely).
- **QK-norm**: apply LayerNorm/RMSNorm to queries and keys before attention, capping logit growth (helps very large/long-context runs).
- **bf16 over fp16**: wider dynamic range avoids overflow (see [Mixed Precision](04_mixed_precision_and_quantization.md)).
- **Embedding/logit soft-capping** (Gemma 2): tanh-cap logits to a fixed range.

## Key Challenges
- **Scale-dependent instabilities.** Pathologies that don't appear below ~10B params make small-scale debugging unreliable.
- **Forensics are hard.** Distinguishing a bad batch, a hardware fault (silent data corruption / bit flip), and a genuine optimization instability is nontrivial mid-run.
- **Hardware faults at scale.** With 10k+ GPUs, SDC (silent data corruption), ECC errors, and NCCL hangs are routine and can masquerade as model instability.
- **Costly interventions.** Rollback to a checkpoint wastes compute; skipping batches or lowering LR may not fix the root cause.

## Solutions & Current Best Practices
**Prevent**: bf16, μP/scaled init, residual scaling, LR warmup, gradient clipping, z-loss, and (for big/long runs) QK-norm and logit soft-capping. **Detect**: monitor gradient norm, loss, and per-layer activation statistics; use **small-scale proxies** (Wortsman et al. 2023) that reproduce instabilities cheaply to pick stable configs. **Recover**: checkpoint frequently, and on a spike, **roll back and skip the offending data batches** (PaLM's documented recipe) or briefly lower LR. Deterministic, resumable data loading is essential.

## Lab Perspectives
- **Google** documented loss spikes in PaLM and the skip-batch recovery; popularized z-loss and μP-style scaling.
- **Meta (OPT)** published a candid logbook of instabilities and manual interventions — a foundational reference.
- **DeepMind/Microsoft** advanced μP/μTransfer for hyperparameter transfer.
- **OpenAI/Anthropic** keep recipes closed but clearly invest heavily in stability tooling and monitoring; Anthropic has written on the importance of reproducible, deterministic infrastructure.

## Latest Developments (2023–2026)
**μTransfer** is increasingly standard for de-risking HP choices. **QK-norm and logit soft-capping** (Gemma 2, OLMo 2) became common stabilizers. **fp8 training** (H100) reintroduced dynamic-range stability concerns, addressed via per-tensor scaling (Transformer Engine). Better **silent-data-corruption detection** and elastic recovery matured for 100k-GPU-class runs. "Spike-free" recipes and small-proxy stability prediction are active engineering areas.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What causes training loss spikes and how do you handle them?"** Logit/gradient growth, bad batches, fp16 overflow; clip, z-loss, QK-norm, rollback + skip-batch.
- **"What is μP / μTransfer and why does it matter?"** Width-invariant optimal HPs → tune small, transfer large; de-risks big runs.
- **"Why bf16 over fp16 for pretraining?"** Wider exponent range avoids overflow without loss scaling.
- **"How do you tell a hardware fault from an optimization instability?"** Reproducibility checks, gradient/activation monitoring, ECC/SDC logs.

## Open Problems
There is no predictive *theory* of loss spikes — mitigations are largely empirical. Reliably forecasting large-scale instabilities from small proxies is improving but unsolved. As runs scale to 100k+ accelerators and lower precision (fp8/fp4), the interaction of numerical precision, optimizer dynamics, and hardware faults creates new, poorly-understood failure modes.

## References
- Yang, G. et al. (2022). *Tensor Programs V: μTransfer.* arXiv:2203.03466.
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Zhang, S. et al. (2022). *OPT (logbook).* arXiv:2205.01068.
- Wortsman, M. et al. (2023). *Small-scale Proxies for Large-scale Transformer Training Instabilities.* arXiv:2309.14322.
- Dehghani, M. et al. (2023). *ViT-22B (QK-norm).* arXiv:2302.05442.
