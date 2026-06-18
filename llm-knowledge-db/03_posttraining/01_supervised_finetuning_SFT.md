# Supervised Fine-Tuning (SFT)

> **Last Updated:** 2026-06-18
> **Related Files:** [Post-Training Overview](00_posttraining_overview.md) · [Instruction Tuning](02_instruction_tuning.md) · [Synthetic Data Generation](../01_data/04_synthetic_data_generation.md)
> **Key Papers:** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Zhou et al. 2023 LIMA ("Less Is More for Alignment") ([arXiv:2305.11206](https://arxiv.org/abs/2305.11206)) · Taori et al. 2023 Alpaca · Chung et al. 2022 Scaling Instruction-Finetuning (FLAN) ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416))

## Overview
Supervised fine-tuning is the first and cheapest post-training stage: continue next-token training on a curated dataset of (prompt, ideal-response) pairs so the base model learns the *format* and *behavior* of an assistant — following instructions, using a chat structure, refusing appropriately, and producing helpful, well-formatted answers. SFT is "behavioral cloning": the model imitates demonstrations. It establishes the prior that later RL/preference stages refine. A central, somewhat surprising finding (LIMA, Zhou et al. 2023) is that **a small number (~1k) of very high-quality, diverse examples** can produce strong instruction-following, suggesting base models already "know" most capabilities and SFT mainly *elicits style and format* — the "Superficial Alignment Hypothesis."

## Core Concepts
**Objective and loss masking.** SFT uses the standard cross-entropy LM loss, but **only on the assistant/response tokens** — the prompt/user tokens (and system prompt) are masked out of the loss. This prevents the model from learning to *generate* user turns and focuses capacity on producing good responses:
$$\mathcal{L}_{SFT}=-\sum_{i\in \text{response}}\log p_\theta(x_i\mid x_{<i})$$
Forgetting to mask the prompt is a classic bug that degrades quality.

**Chat templates.** Multi-turn conversations are serialized with special control tokens demarcating roles. Common formats:
- **ChatML** (OpenAI): `<|im_start|>role ... <|im_end|>`.
- **Llama-2/3 chat**: `[INST] ... [/INST]` (Llama 2) / Llama 3's header tokens `<|start_header_id|>role<|end_header_id|>`.
- **Mistral**: `[INST] ... [/INST]`.
The template must be **identical at train and inference**; mismatches cause severe degradation. Templates also encode system prompts and tool-call structure.

**Packing.** To avoid wasting compute on padding, multiple short examples are **packed** into one sequence up to the context length, with attention masking (or document separators / reset of position ids) so examples don't attend across boundaries. Improves throughput substantially.

**Multi-turn handling.** Each assistant turn contributes loss; earlier turns serve as context. Care is needed so the model learns to condition on full dialogue history.

## Key Challenges
- **Data quality dominates.** SFT is extremely sensitive to demonstration quality; a few bad examples (wrong format, hallucinated facts, sycophancy) propagate. Quality ≫ quantity (LIMA).
- **Imitation ceiling.** SFT can only reproduce demonstrated behavior; it can't optimize toward what's *preferred* beyond the data, nor learn from negative examples — motivating RL/DPO.
- **Distribution shift / exposure bias.** Trained on gold prefixes, the model conditions on its own outputs at inference.
- **Catastrophic forgetting.** Heavy SFT can erode base-model knowledge/capabilities ("alignment tax").
- **Template/format brittleness.** Inconsistent templates between training and serving break behavior.

## Solutions & Current Best Practices
**Curate aggressively** (dedupe, filter, verify; prefer fewer high-quality, diverse examples — LIMA/phi philosophy). Use **synthetic data** (Self-Instruct/Evol-Instruct/Orca) plus human-written gold data, with **execution/verification** for code and math (see [Synthetic Data](../01_data/04_synthetic_data_generation.md)). **Mask prompts**, **pack sequences**, use a **consistent chat template**, modest epochs (1–3) and low LR to limit forgetting. SFT is typically the **warm start** for DPO/RLHF; many pipelines also blend in pretraining data to preserve capabilities.

## Lab Perspectives
- **OpenAI:** InstructGPT's SFT on human demonstrations seeded the modern recipe; ChatML standardized templates.
- **Google:** FLAN/T0 scaled *instruction* SFT across thousands of tasks for zero-shot generalization (see [Instruction Tuning](02_instruction_tuning.md)).
- **Meta:** Llama 2/3 used curated SFT (tens of thousands to millions of examples) plus synthetic data and rejection-sampled responses.
- **Microsoft:** phi/Orca emphasize synthetic, high-quality, reasoning-rich SFT data.
- **Anthropic:** SFT as the supervised stage feeding Constitutional AI / RLAIF.

## Latest Developments (2023–2026)
The field moved toward **synthetic, verified, reasoning-dense SFT** data and **long-CoT SFT** (distilling reasoning traces from R1/o1-style teachers as a cold-start before RL — DeepSeek-R1 used a small high-quality long-CoT SFT to bootstrap RL). "**SFT then RL**" remains standard, with debate over how much SFT is needed if strong RL follows (R1-Zero skipped SFT entirely). Quality-over-quantity (LIMA) is now conventional wisdom.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why mask the loss on prompt tokens?"** Focus learning on responses; avoid learning to generate user turns.
- **"What is the Superficial Alignment Hypothesis (LIMA)?"** Base model holds the knowledge; SFT mostly elicits style/format → few high-quality examples suffice.
- **"What's sequence packing and why use it?"** Concatenate examples to fill context, mask cross-attention; big throughput win.
- **"SFT limitations that motivate RLHF/DPO?"** Imitation ceiling, no negative signal, can't optimize beyond demonstrations.

## Open Problems
How much SFT is truly necessary versus pure RL (R1-Zero) is unsettled. The precise mechanism of the Superficial Alignment Hypothesis, the limits of distilling reasoning via SFT alone, and how to SFT without inducing forgetting/alignment-tax are open. Optimal data selection ("which 1k examples?") lacks a principled theory.

## References
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Zhou, C. et al. (2023). *LIMA.* arXiv:2305.11206.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned LMs (FLAN).* arXiv:2210.11416.
- Taori, R. et al. (2023). *Stanford Alpaca.*
- DeepSeek-AI (2025). *DeepSeek-R1 (cold-start SFT).* arXiv:2501.12948.
