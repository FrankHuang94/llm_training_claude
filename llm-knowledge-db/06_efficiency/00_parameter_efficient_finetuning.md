# Parameter-Efficient Fine-Tuning (PEFT)

> **Last Updated:** 2026-11-04
> **Related Files:** [LoRA & Variants](01_LoRA_and_variants.md) · [Quantization Techniques](02_quantization_techniques.md) · [Supervised Fine-Tuning](../03_posttraining/01_supervised_finetuning_SFT.md)
> **Key Papers:** Houlsby et al. 2019 Adapters ([arXiv:1902.00751](https://arxiv.org/abs/1902.00751)) · Li & Liang 2021 Prefix-Tuning ([arXiv:2101.00190](https://arxiv.org/abs/2101.00190)) · Lester et al. 2021 Prompt Tuning ([arXiv:2104.08691](https://arxiv.org/abs/2104.08691)) · Liu et al. 2022 (IA)³ ([arXiv:2205.05638](https://arxiv.org/abs/2205.05638))

## Overview
Full fine-tuning of a multi-billion-parameter model updates every weight, requiring optimizer states and gradients for the entire model (≈16 bytes/param with Adam) and producing a full-size checkpoint per task — prohibitive for most practitioners and unwieldy for serving many task-specific variants. With open-weight MoEs now at 1–2.8T total parameters, full fine-tuning is out of reach for nearly everyone outside the releasing lab. **Parameter-efficient fine-tuning (PEFT)** freezes the pretrained backbone and trains only a small number of new or selected parameters (often <1%), achieving near-full-fine-tuning quality at a fraction of the memory, storage, and compute. PEFT democratized adaptation of large models and is foundational to modern fine-tuning practice; LoRA (covered in detail in [LoRA & Variants](01_LoRA_and_variants.md)) is the dominant member of this family.

## Core Concepts
**Why PEFT works.** Empirically, the *task-adaptation* a pretrained model needs lies in a **low intrinsic dimension** (Aghajanyan et al. 2020) — you don't need to move all weights to specialize a capable base model. PEFT methods exploit this by restricting updates to a small subspace. RL post-training is an extreme case: a policy-gradient episode carries very little information (on the order of one bit per episode with a scalar reward), so the update it induces is intrinsically tiny.

**Method families.**
- **Additive — Adapters** (Houlsby et al.): insert small bottleneck MLP modules (down-project → nonlinearity → up-project) between transformer sublayers; train only these. Adds inference latency unless merged.
- **Reparameterization — LoRA** (Hu et al.): learn a low-rank update $\Delta W = BA$ added to frozen weights; *mergeable* at inference (no latency). The dominant approach — see [LoRA & Variants](01_LoRA_and_variants.md).
- **Prompt/Prefix tuning** (Lester, Li & Liang): prepend trainable continuous "soft prompt" / prefix vectors to the input or to keys/values at each layer; the model weights are untouched. Extremely parameter-light but harder to optimize and weaker on hard tasks.
- **Selective — BitFit** (Ben-Zaken et al.): fine-tune only the **bias** terms. Surprisingly competitive on some tasks; near-zero added parameters. A modern relative is **sparse memory finetuning**, which updates only the memory-layer slots most specific to the new data.
- **(IA)³** (Liu et al.): learn element-wise **rescaling vectors** for keys, values, and FFN activations — very few parameters, strong few-shot results, mergeable.

**Tradeoffs across methods.** Adapters and LoRA are the most robust to hard tasks; prompt/prefix tuning is lightest but task-sensitive and benefits from scale; selective methods (BitFit) are simplest but limited. LoRA's *mergeability* (zero inference overhead) is a key reason it won over adapters in practice.

## Key Challenges
- **Capacity ceiling.** For large *distribution shifts* (new domain/language, major capability addition), PEFT can underperform full fine-tuning — there isn't enough capacity in the small update.
- **Inference overhead (some methods).** Unmerged adapters/prefixes add latency and complicate serving.
- **Hyperparameter sensitivity.** Rank (LoRA), bottleneck size (adapters), prompt length, and learning rate need tuning; defaults don't always transfer.
- **Multi-task interference.** Composing/merging multiple PEFT modules can cause interference.
- **Where to put the parameters.** Attention-only adapters underperform on MoE/MLP-heavy models; placement matters as much as parameter count.

## Solutions & Current Best Practices
**LoRA/QLoRA is the default PEFT method** for instruction-tuning and domain adaptation: strong quality, mergeable, and combinable with 4-bit quantization for single-GPU fine-tuning of large models (see [LoRA & Variants](01_LoRA_and_variants.md) and [Quantization](02_quantization_techniques.md)). Apply it to **all layers, especially MLP/MoE layers**, with a learning rate roughly an order of magnitude above full fine-tuning. For **RL fine-tuning**, very low-rank LoRA is sufficient. Use **full fine-tuning** when the task demands large capability shifts or maximal quality and resources allow. **Multi-LoRA serving** (swap/stack task adapters on a shared base, e.g., S-LoRA, Punica) enables thousands of task variants on one base model — a major production advantage. The HuggingFace **PEFT** library standardizes these methods.

## Lab Perspectives
- **Industry/open-source** practitioners rely heavily on PEFT (LoRA/QLoRA) for customizing open models cheaply — the dominant fine-tuning mode outside frontier labs.
- **Frontier labs** (OpenAI/Anthropic/Google) generally **full-fine-tune** their own models for flagship post-training (max quality, ample compute) but expose **LoRA-style fine-tuning APIs** to customers and use adapter-based serving for customization.
- **Thinking Machines Lab** published the most systematic recent study of when LoRA matches full fine-tuning (SFT and RL).
- **Meta FAIR** proposed sparse memory-layer finetuning as a forgetting-resistant alternative to LoRA for continual learning.
- **Cloud providers** (multi-tenant) favor LoRA for **multi-tenant serving** — many customer adapters over one base model.

## Latest Developments (2023–Nov 2026)
**QLoRA** (4-bit base + LoRA) made large-model fine-tuning accessible on a single GPU and triggered an explosion of open fine-tunes. **Multi-LoRA serving** systems matured (S-LoRA, Punica) for serving many adapters efficiently. New variants (**DoRA**, **LoRA+**, **VeRA**, **PiSSA**) narrowed the gap to full fine-tuning. PEFT is also used in **continual learning** (per-task adapters to avoid forgetting, see [Continual Learning](../09_emerging_frontiers/01_continual_learning.md)) and increasingly in **RL post-training** for cost savings.

**Mid-2025 → Nov 2026 update:**
- **LoRA Without Regret (2025-09, Thinking Machines).** LoRA matches full fine-tuning when applied to **all layers — especially MLP and MoE layers** (attention-only LoRA underperforms); its optimal LR is ~10× the full-FT LR; and for policy-gradient RL even **rank-1** LoRA matches full FT, because RL absorbs only ~1 bit per episode. LoRA falls behind mainly when a large SFT dataset exceeds adapter capacity.
- **RL updates are intrinsically sparse (2026-03).** PRISM finds that midtraining restructures >90% of weights while RL makes sparse updates to ~5% of parameters ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)) — independent evidence that RL is a natural fit for PEFT.
- **Sparse memory finetuning (2025-10, Meta FAIR).** Memory-layer models update only the top-$t$ slots chosen by TF-IDF-like ratios (frequent in the batch, rare in pretraining). NaturalQuestions forgetting drops to 11%, versus 89% for full FT and 71% for LoRA, at equal new-knowledge acquisition ([arXiv:2510.15103](https://arxiv.org/abs/2510.15103)).
- **Test-time parameter updates (2025-12).** TTT-E2E updates only a subset of MLPs with next-token gradient steps on the context at test time, with a meta-learned initialization, giving constant per-token latency and 2.7× speedup at 128K at 3B scale ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)) — PEFT-style partial updates as a long-context memory mechanism.
- **Trillion-scale, low-bit bases (2025-07 → 2026-08).** Open bases now span 1–2.8T total parameters (Kimi K2/K3, DeepSeek-V4, Qwen3.8), and several ship QAT-native low-bit weights (Kimi K2 Thinking INT4, DeepSeek-V4 FP4-QAT experts, Kimi K3 MXFP4), so the QLoRA pattern — adapters over a frozen low-bit base — is often the only practical way to adapt them.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does PEFT work at all?"** Low intrinsic dimensionality of task adaptation; a capable base needs only a small update.
- **"Compare adapters, prefix tuning, and LoRA."** Additive bottleneck (latency) vs soft prompts (light, finicky) vs low-rank mergeable update (no latency) — LoRA wins on robustness + mergeability.
- **"When does PEFT underperform full fine-tuning?"** Large distribution shifts / major capability additions exceed the low-rank capacity.
- **"How do you serve 1000 fine-tuned variants cheaply?"** Multi-LoRA: shared frozen base + swappable per-task adapters.
- **"Why can rank-1 LoRA match full fine-tuning in RL but not in large SFT?"** RL's scalar reward conveys ~1 bit per episode, so total information is tiny; large SFT datasets carry far more bits than a low-rank adapter can store.

## Open Problems
Predicting *a priori* when PEFT suffices vs needs full fine-tuning is unsolved, though information-capacity arguments (bits per example vs adapter capacity) are a promising lens. Optimal method/rank selection, robust **composition/merging** of many PEFT modules without interference, PEFT for continual learning without forgetting, and closing the residual quality gap to full fine-tuning on the hardest tasks remain active research.

## References
- Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning (Adapters).* arXiv:1902.00751.
- Li, X., Liang, P. (2021). *Prefix-Tuning.* arXiv:2101.00190.
- Lester, B. et al. (2021). *The Power of Scale for Prompt Tuning.* arXiv:2104.08691.
- Liu, H. et al. (2022). *(IA)³ / T-Few.* arXiv:2205.05638.
- Aghajanyan, A. et al. (2020). *Intrinsic Dimensionality.* arXiv:2012.13255.
- Thinking Machines Lab (2025). *LoRA Without Regret.* Blog post.
- Lin, J. et al. (2025). *Continual Learning via Sparse Memory Finetuning.* arXiv:2510.15103.
