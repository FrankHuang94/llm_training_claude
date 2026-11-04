# Supervised Fine-Tuning (SFT)

> **Last Updated:** 2026-11-04
> **Related Files:** [Post-Training Overview](00_posttraining_overview.md) · [Instruction Tuning](02_instruction_tuning.md) · [Synthetic Data Generation](../01_data/04_synthetic_data_generation.md) · [LoRA & Variants](../06_efficiency/01_LoRA_and_variants.md)
> **Key Papers:** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Zhou et al. 2023 LIMA ("Less Is More for Alignment") ([arXiv:2305.11206](https://arxiv.org/abs/2305.11206)) · Taori et al. 2023 Alpaca · Chung et al. 2022 Scaling Instruction-Finetuning (FLAN) ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416)) · Midtraining Bridges Pretraining and Posttraining 2025 ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865))

## Overview
Supervised fine-tuning is the first and cheapest post-training stage: continue next-token training on a curated dataset of (prompt, ideal-response) pairs so the base model learns the *format* and *behavior* of an assistant — following instructions, using a chat structure, refusing appropriately, and producing helpful, well-formatted answers. SFT is "behavioral cloning": the model imitates demonstrations. It establishes the prior that later RL/preference stages refine. A central, somewhat surprising finding (LIMA, Zhou et al. 2023) is that **a small number (~1k) of very high-quality, diverse examples** can produce strong instruction-following, suggesting base models already "know" most capabilities and SFT mainly *elicits style and format* — the "Superficial Alignment Hypothesis." In 2026 pipelines SFT is increasingly sandwiched between **midtraining** (which sets capability ceilings) and **RL / on-policy distillation** (which corrects SFT's off-policy weaknesses).

## Core Concepts
**Objective and loss masking.** SFT uses the standard cross-entropy LM loss, but **only on the assistant/response tokens** — the prompt/user tokens (and system prompt) are masked out of the loss. This prevents the model from learning to *generate* user turns and focuses capacity on producing good responses:
$$\mathcal{L}_{SFT}=-\sum_{i\in \text{response}}\log p_\theta(x_i\mid x_{<i})$$
Forgetting to mask the prompt is a classic bug that degrades quality.

**Chat templates.** Multi-turn conversations are serialized with special control tokens demarcating roles. Common formats:
- **ChatML** (OpenAI): `<|im_start|>role ... <|im_end|>`.
- **Llama-2/3 chat**: `[INST] ... [/INST]` (Llama 2) / Llama 3's header tokens `<|start_header_id|>role<|end_header_id|>`.
- **Mistral**: `[INST] ... [/INST]`.
The template must be **identical at train and inference**; mismatches cause severe degradation. Templates also encode system prompts, tool-call structure, and — since hybrid reasoning models (e.g., DeepSeek-V3.1's Think/Non-Think modes, 2025-08) — **reasoning-mode switches and thinking blocks**, so a single checkpoint can be SFT'd for both modes.

**Packing.** To avoid wasting compute on padding, multiple short examples are **packed** into one sequence up to the context length, with attention masking (or document separators / reset of position ids) so examples don't attend across boundaries. Improves throughput substantially.

**Multi-turn handling.** Each assistant turn contributes loss; earlier turns serve as context. Care is needed so the model learns to condition on full dialogue history. For agent traces, tool outputs are treated like user turns (masked), and only the model's own actions/thoughts are trained.

**Off-policy vs on-policy distillation.** Classic SFT-distillation trains on *teacher* samples (off-policy), so the student never learns to recover from its own mistakes. **On-policy distillation** (Thinking Machines, 2025-10) instead has the student sample and the teacher score each token with reverse KL, $\mathcal{L}=\sum_t \mathrm{KL}\big(\pi_\theta(\cdot\mid y_{<t})\,\|\,\pi_{T}(\cdot\mid y_{<t})\big)$ over $y\sim\pi_\theta$ — dense supervision without exposure bias.

## Key Challenges
- **Data quality dominates.** SFT is extremely sensitive to demonstration quality; a few bad examples (wrong format, hallucinated facts, sycophancy) propagate. Quality ≫ quantity (LIMA).
- **Imitation ceiling.** SFT can only reproduce demonstrated behavior; it can't optimize toward what's *preferred* beyond the data, nor learn from negative examples — motivating RL/DPO.
- **Distribution shift / exposure bias.** Trained on gold prefixes, the model conditions on its own outputs at inference.
- **Catastrophic forgetting.** Heavy SFT can erode base-model knowledge/capabilities ("alignment tax").
- **Template/format brittleness.** Inconsistent templates between training and serving break behavior.
- **Teacher access.** Frontier labs now actively resist distillation (account-bound "preserved thinking," distillation classifiers), so long-CoT teacher traces from closed models are harder to obtain.

## Solutions & Current Best Practices
**Curate aggressively** (dedupe, filter, verify; prefer fewer high-quality, diverse examples — LIMA/phi philosophy). Use **synthetic data** (Self-Instruct/Evol-Instruct/Orca) plus human-written gold data, with **execution/verification** for code and math (see [Synthetic Data](../01_data/04_synthetic_data_generation.md)). **Mask prompts**, **pack sequences**, use a **consistent chat template**, modest epochs (1–3) and low LR to limit forgetting. SFT is typically the **warm start** for DPO/RLHF; many pipelines also blend in pretraining data to preserve capabilities. For parameter-efficient SFT, apply **LoRA to all layers (especially MLP/MoE)** with ~10× the full-FT learning rate. Follow SFT with **on-policy distillation** when a strong teacher exists, and use rejection fine-tuning on verified agent trajectories as a cold start for agentic RL.

## Lab Perspectives
- **OpenAI:** InstructGPT's SFT on human demonstrations seeded the modern recipe; ChatML standardized templates.
- **Google:** FLAN/T0 scaled *instruction* SFT across thousands of tasks for zero-shot generalization (see [Instruction Tuning](02_instruction_tuning.md)).
- **Meta:** Llama 2/3 used curated SFT (tens of thousands to millions of examples) plus synthetic data and rejection-sampled responses.
- **Microsoft:** phi/Orca emphasize synthetic, high-quality, reasoning-rich SFT data.
- **Anthropic:** SFT as the supervised stage feeding Constitutional AI / RLAIF; the 2026 constitution is used to generate Claude-written synthetic training data.
- **Thinking Machines:** LoRA-vs-full-FT and on-policy-distillation studies reframed what SFT should and shouldn't do.

## Latest Developments (2023–Nov 2026)
The field moved toward **synthetic, verified, reasoning-dense SFT** data and **long-CoT SFT** (distilling reasoning traces from R1/o1-style teachers as a cold-start before RL — DeepSeek-R1 used a small high-quality long-CoT SFT to bootstrap RL). "**SFT then RL**" remains standard, with debate over how much SFT is needed if strong RL follows (R1-Zero skipped SFT entirely). Quality-over-quantity (LIMA) is now conventional wisdom.

**Mid-2025 → Nov 2026 update:**
- **On-policy distillation displaces some SFT-distillation (2025-10 → 2026).** Student-sampled, teacher-scored reverse KL reached ~70% AIME24 at ~1/10 the GPU-hours of RL and recovered instruction-following after domain midtraining (Thinking Machines blog). Fixes for mode-seeking reverse KL followed: entropy-gated forward/reverse switching (EOPD, [arXiv:2603.07079](https://arxiv.org/abs/2603.07079)) and truncated top-K reverse KL ([arXiv:2603.25562](https://arxiv.org/abs/2603.25562)).
- **LoRA Without Regret (2025-09).** LoRA matches full fine-tuning when applied to all layers (attention-only LoRA underperforms), with an optimal LR ~10× full-FT's — making cheap SFT nearly lossless at post-training data scales (Thinking Machines blog).
- **Midtraining as the distribution bridge (2025-10).** When specialized data enters interacts with its mixture weight — early allows a high weight, late needs a low one; gains largest for math/code ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865)). PRISM ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)) finds midtraining restructures >90% of weights vs ~5% sparse updates from RL.
- **SFT interleaved inside RL.** ComputerRL's "Entropulse" alternates RL and SFT phases to recover entropy in long agentic RL runs ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040)); Nebius's SWE agents use rejection fine-tuning on real-repo trajectories before multi-turn DAPO ([arXiv:2508.03501](https://arxiv.org/abs/2508.03501)).
- **Hybrid-mode SFT.** One set of weights now serves thinking and non-thinking modes (DeepSeek-V3.1, 2025-08; V4 Thinking/Non-Thinking, 2026-04), shifting template design toward explicit mode tokens.
- **Distillation defenses (2026).** Anthropic's Opus 5.5 "preserved thinking" blocks API users from editing prior context to extract reasoning, and Fable 5's classifiers route distillation-like requests away — reducing the supply of frontier long-CoT traces for SFT.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why mask the loss on prompt tokens?"** Focus learning on responses; avoid learning to generate user turns.
- **"What is the Superficial Alignment Hypothesis (LIMA)?"** Base model holds the knowledge; SFT mostly elicits style/format → few high-quality examples suffice.
- **"What's sequence packing and why use it?"** Concatenate examples to fill context, mask cross-attention; big throughput win.
- **"SFT limitations that motivate RLHF/DPO?"** Imitation ceiling, no negative signal, can't optimize beyond demonstrations.
- **"SFT-distillation vs on-policy distillation?"** Off-policy teacher samples suffer exposure bias; OPD trains on the student's own samples with dense per-token teacher KL — fixes compounding error; watch reverse-KL mode-seeking.

## Open Problems
How much SFT is truly necessary versus pure RL (R1-Zero) is unsettled. The precise mechanism of the Superficial Alignment Hypothesis, the limits of distilling reasoning via SFT alone, and how to SFT without inducing forgetting/alignment-tax are open. Optimal data selection ("which 1k examples?") lacks a principled theory. The division of labor between midtraining, SFT, and on-policy distillation is now an open design question.

## References
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Zhou, C. et al. (2023). *LIMA.* arXiv:2305.11206.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned LMs (FLAN).* arXiv:2210.11416.
- Taori, R. et al. (2023). *Stanford Alpaca.*
- DeepSeek-AI (2025). *DeepSeek-R1 (cold-start SFT).* arXiv:2501.12948.
- CMU (2025). *Midtraining Bridges Pretraining and Posttraining Distributions.* arXiv:2510.14865.
- Thinking Machines Lab (2025). *LoRA Without Regret; On-Policy Distillation.* Blog posts.
