# LoRA and Variants

> **Last Updated:** 2026-06-18
> **Related Files:** [Parameter-Efficient Fine-Tuning](00_parameter_efficient_finetuning.md) · [Quantization Techniques](02_quantization_techniques.md) · [Supervised Fine-Tuning](../03_posttraining/01_supervised_finetuning_SFT.md)
> **Key Papers:** Hu et al. 2021 LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) · Dettmers et al. 2023 QLoRA ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)) · Liu et al. 2024 DoRA ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353)) · Hayou et al. 2024 LoRA+ ([arXiv:2402.12354](https://arxiv.org/abs/2402.12354))

## Overview
LoRA (Low-Rank Adaptation) is the single most important parameter-efficient fine-tuning method and one of the most-used techniques in all of applied LLM work. Its premise: the weight *update* needed to adapt a pretrained model to a task has **low intrinsic rank**, so instead of learning a full $d\times k$ update matrix, learn its low-rank factorization. This cuts trainable parameters by 100–10,000×, slashes optimizer-state memory, produces tiny (MB-scale) checkpoints, and — crucially — can be **merged back** into the base weights for **zero inference latency**. Combined with quantization (QLoRA), it enables fine-tuning 65B+ models on a single consumer GPU. The LoRA math and its tradeoffs vs full fine-tuning are common interview material.

## Core Concepts
**The decomposition.** For a frozen pretrained weight $W_0 \in \mathbb{R}^{d\times k}$, LoRA represents the update as a product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll \min(d,k)$$
Only $A$ and $B$ are trained ($W_0$ frozen). Trainable params drop from $dk$ to $r(d+k)$. The forward pass is $h = W_0 x + BA x$, scaled by $\frac{\alpha}{r}$ (a tunable factor). **Initialization:** $A$ random Gaussian, $B=0$, so $\Delta W=0$ at start (training begins exactly at the pretrained model). At inference, compute $W_0 + BA$ once and **merge** — no extra latency or parameters.

**Where to apply.** Originally the attention projections ($W_q, W_v$); modern practice applies LoRA to *all* linear layers (Q,K,V,O and the FFN/MLP matrices), which improves quality, especially at low rank.

**Hyperparameters.** **Rank $r$** (typically 8–64; higher for harder tasks/larger shifts), **$\alpha$** (scaling; common heuristic $\alpha=2r$), target modules, and LoRA dropout. Quality is fairly robust to $r$ above a task-dependent threshold; too-low $r$ underfits large shifts.

**QLoRA.** Dettmers et al.: quantize the **frozen base model to 4-bit** (NF4 — a normal-float datatype optimal for normally-distributed weights), keep LoRA adapters in bf16, and backprop through the quantized base. Innovations: **NF4**, **double quantization** (quantize the quantization constants), and **paged optimizers** (handle memory spikes). Result: fine-tune a 65B model on a single 48GB GPU with negligible quality loss — a landmark for accessibility.

**Major variants.**
- **DoRA** (Weight-Decomposed LoRA): decompose weights into **magnitude** and **direction**, apply LoRA to the direction and train magnitude separately — closes much of the gap to full fine-tuning, especially at low rank.
- **LoRA+**: use a **higher learning rate for $B$ than $A$** (they should be scaled differently); improves convergence/quality at no extra cost.
- **rsLoRA**: rank-stabilized scaling ($\alpha/\sqrt{r}$) for better high-rank behavior.
- **PiSSA / LoftQ**: initialize $A,B$ from the SVD of $W_0$ (principal components) rather than randomly — faster convergence and better quantized-init (LoftQ minimizes quantization error).
- **VeRA**: share frozen random $A,B$ across layers and train tiny scaling vectors — extreme parameter efficiency.

## Key Challenges
- **Capacity limits.** LoRA can underperform full fine-tuning on tasks requiring **large knowledge/capability additions** (e.g., new language, deep domain shift) — the low-rank update can't absorb enough.
- **Rank selection.** Too low → underfit; too high → diminishing returns and more memory; the right $r$ is task-dependent and empirical.
- **QLoRA precision/throughput.** 4-bit base adds dequantization overhead in training and some quality risk on sensitive tasks.
- **Merging conflicts.** Stacking/merging multiple LoRAs can interfere; adapter arithmetic is imperfect.
- **Optimization differences.** Vanilla LoRA's symmetric LR for $A,B$ is suboptimal (fixed by LoRA+).

## Solutions & Current Best Practices
**Default recipe:** LoRA on **all linear layers**, $r=16$–$64$, $\alpha=2r$, plus **QLoRA (NF4)** when memory-constrained. Use **DoRA or LoRA+** for a quality bump at low cost; **PiSSA/LoftQ** init for faster convergence (and better quantized init). For serving many tasks, use **multi-LoRA** systems (S-LoRA, Punica) that keep one base model and hot-swap adapters. Prefer **full fine-tuning** when maximal quality on a large distribution shift is required and resources allow. Validate that merged weights match the adapter-applied forward pass.

## Lab Perspectives
- **Microsoft** authored LoRA; it underpins much of Azure/OpenAI customization tooling.
- **Open-source ecosystem** (HuggingFace PEFT, Unsloth, Axolotl) made LoRA/QLoRA the default community fine-tuning method.
- **Frontier labs** use full fine-tuning for flagship post-training but offer **LoRA fine-tuning APIs** and rely on **multi-adapter serving** for customer customization.
- **Cloud/serving providers** standardized **multi-LoRA** to serve many tenants over a shared base.

## Latest Developments (2023–2026)
**QLoRA** democratized large-model fine-tuning; **DoRA/LoRA+/PiSSA/rsLoRA** narrowed the gap to full fine-tuning. **Multi-LoRA serving** became production-standard. LoRA-style adapters are increasingly used in **RL post-training** (cheaper GRPO/DPO) and **continual learning** (per-skill adapters). Research continues on **when LoRA suffices vs full fine-tuning** (LoRA learns less but forgets less — a useful regularization tradeoff) and on better initialization/optimization.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Derive LoRA and explain the parameter savings."** $\Delta W=BA$, rank $r$; $dk \to r(d+k)$; $B=0$ init so training starts at pretrained model; mergeable → no inference cost.
- **"What does QLoRA add?"** 4-bit NF4 frozen base + bf16 adapters + double quantization + paged optimizers → fine-tune 65B on one GPU.
- **"When does LoRA fail vs full fine-tuning?"** Large capability/knowledge shifts exceed low-rank capacity.
- **"What problem does DoRA / LoRA+ solve?"** DoRA: magnitude/direction decomposition closes the gap at low rank; LoRA+: different LRs for A vs B improve convergence.
- **"How do you serve many LoRA adapters efficiently?"** Shared base + batched multi-adapter kernels (S-LoRA/Punica).

## Open Problems
Predicting the **minimal rank** for a task and *when* LoRA matches full fine-tuning is unsolved. Robust **composition** of multiple adapters (LoRA arithmetic) without interference, optimal initialization, and understanding LoRA's regularization/forgetting tradeoff theoretically are active research areas.

## References
- Hu, E. et al. (2021). *LoRA.* arXiv:2106.09685.
- Dettmers, T. et al. (2023). *QLoRA.* arXiv:2305.14314.
- Liu, S.-Y. et al. (2024). *DoRA.* arXiv:2402.09353.
- Hayou, S. et al. (2024). *LoRA+.* arXiv:2402.12354.
- Sheng, Y. et al. (2023). *S-LoRA.* arXiv:2311.03285.
