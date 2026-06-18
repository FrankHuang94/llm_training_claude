# Parameter-Efficient Fine-Tuning (PEFT)

> **Last Updated:** 2026-06-18
> **Related Files:** [LoRA & Variants](01_LoRA_and_variants.md) · [Quantization Techniques](02_quantization_techniques.md) · [Supervised Fine-Tuning](../03_posttraining/01_supervised_finetuning_SFT.md)
> **Key Papers:** Houlsby et al. 2019 Adapters ([arXiv:1902.00751](https://arxiv.org/abs/1902.00751)) · Li & Liang 2021 Prefix-Tuning ([arXiv:2101.00190](https://arxiv.org/abs/2101.00190)) · Lester et al. 2021 Prompt Tuning ([arXiv:2104.08691](https://arxiv.org/abs/2104.08691)) · Liu et al. 2022 (IA)³ ([arXiv:2205.05638](https://arxiv.org/abs/2205.05638))

## Overview
Full fine-tuning of a multi-billion-parameter model updates every weight, requiring optimizer states and gradients for the entire model (≈16 bytes/param with Adam) and producing a full-size checkpoint per task — prohibitive for most practitioners and unwieldy for serving many task-specific variants. **Parameter-efficient fine-tuning (PEFT)** freezes the pretrained backbone and trains only a small number of new or selected parameters (often <1%), achieving near-full-fine-tuning quality at a fraction of the memory, storage, and compute. PEFT democratized adaptation of large models and is foundational to modern fine-tuning practice; LoRA (covered in detail in [LoRA & Variants](01_LoRA_and_variants.md)) is the dominant member of this family.

## Core Concepts
**Why PEFT works.** Empirically, the *task-adaptation* a pretrained model needs lies in a **low intrinsic dimension** (Aghajanyan et al. 2020) — you don't need to move all weights to specialize a capable base model. PEFT methods exploit this by restricting updates to a small subspace.

**Method families.**
- **Additive — Adapters** (Houlsby et al.): insert small bottleneck MLP modules (down-project → nonlinearity → up-project) between transformer sublayers; train only these. Adds inference latency unless merged.
- **Reparameterization — LoRA** (Hu et al.): learn a low-rank update $\Delta W = BA$ added to frozen weights; *mergeable* at inference (no latency). The dominant approach — see [LoRA & Variants](01_LoRA_and_variants.md).
- **Prompt/Prefix tuning** (Lester, Li & Liang): prepend trainable continuous "soft prompt" / prefix vectors to the input or to keys/values at each layer; the model weights are untouched. Extremely parameter-light but harder to optimize and weaker on hard tasks.
- **Selective — BitFit** (Ben-Zaken et al.): fine-tune only the **bias** terms. Surprisingly competitive on some tasks; near-zero added parameters.
- **(IA)³** (Liu et al.): learn element-wise **rescaling vectors** for keys, values, and FFN activations — very few parameters, strong few-shot results, mergeable.

**Tradeoffs across methods.** Adapters and LoRA are the most robust to hard tasks; prompt/prefix tuning is lightest but task-sensitive and benefits from scale; selective methods (BitFit) are simplest but limited. LoRA's *mergeability* (zero inference overhead) is a key reason it won over adapters in practice.

## Key Challenges
- **Capacity ceiling.** For large *distribution shifts* (new domain/language, major capability addition), PEFT can underperform full fine-tuning — there isn't enough capacity in the small update.
- **Inference overhead (some methods).** Unmerged adapters/prefixes add latency and complicate serving.
- **Hyperparameter sensitivity.** Rank (LoRA), bottleneck size (adapters), prompt length, and learning rate need tuning; defaults don't always transfer.
- **Multi-task interference.** Composing/merging multiple PEFT modules can cause interference.

## Solutions & Current Best Practices
**LoRA/QLoRA is the default PEFT method** for instruction-tuning and domain adaptation: strong quality, mergeable, and combinable with 4-bit quantization for single-GPU fine-tuning of large models (see [LoRA & Variants](01_LoRA_and_variants.md) and [Quantization](02_quantization_techniques.md)). Use **full fine-tuning** when the task demands large capability shifts or maximal quality and resources allow. **Multi-LoRA serving** (swap/stack task adapters on a shared base, e.g., S-LoRA, Punica) enables thousands of task variants on one base model — a major production advantage. The HuggingFace **PEFT** library standardizes these methods.

## Lab Perspectives
- **Industry/open-source** practitioners rely heavily on PEFT (LoRA/QLoRA) for customizing open models cheaply — the dominant fine-tuning mode outside frontier labs.
- **Frontier labs** (OpenAI/Anthropic/Google) generally **full-fine-tune** their own models for flagship post-training (max quality, ample compute) but expose **LoRA-style fine-tuning APIs** to customers and use adapter-based serving for customization.
- **Cloud providers** (multi-tenant) favor LoRA for **multi-tenant serving** — many customer adapters over one base model.

## Latest Developments (2023–2026)
**QLoRA** (4-bit base + LoRA) made large-model fine-tuning accessible on a single GPU and triggered an explosion of open fine-tunes. **Multi-LoRA serving** systems matured (S-LoRA, Punica) for serving many adapters efficiently. New variants (**DoRA**, **LoRA+**, **VeRA**, **PiSSA**) narrowed the gap to full fine-tuning. PEFT is also used in **continual learning** (per-task adapters to avoid forgetting, see [Continual Learning](../09_emerging_frontiers/01_continual_learning.md)) and increasingly in **RL post-training** for cost savings.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does PEFT work at all?"** Low intrinsic dimensionality of task adaptation; a capable base needs only a small update.
- **"Compare adapters, prefix tuning, and LoRA."** Additive bottleneck (latency) vs soft prompts (light, finicky) vs low-rank mergeable update (no latency) — LoRA wins on robustness + mergeability.
- **"When does PEFT underperform full fine-tuning?"** Large distribution shifts / major capability additions exceed the low-rank capacity.
- **"How do you serve 1000 fine-tuned variants cheaply?"** Multi-LoRA: shared frozen base + swappable per-task adapters.

## Open Problems
Predicting *a priori* when PEFT suffices vs needs full fine-tuning is unsolved. Optimal method/rank selection, robust **composition/merging** of many PEFT modules without interference, and closing the residual quality gap to full fine-tuning on the hardest tasks remain active research.

## References
- Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning (Adapters).* arXiv:1902.00751.
- Li, X., Liang, P. (2021). *Prefix-Tuning.* arXiv:2101.00190.
- Lester, B. et al. (2021). *The Power of Scale for Prompt Tuning.* arXiv:2104.08691.
- Liu, H. et al. (2022). *(IA)³ / T-Few.* arXiv:2205.05638.
- Aghajanyan, A. et al. (2020). *Intrinsic Dimensionality.* arXiv:2012.13255.
