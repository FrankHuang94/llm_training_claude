# Paper Discussion Questions

> **Last Updated:** 2026-06-18
> **Related Files:** [Common Technical Questions](01_common_technical_questions.md) · [Post-Training Overview](../03_posttraining/00_posttraining_overview.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md)

## Overview
Research-oriented interviews (especially at Anthropic, OpenAI, DeepMind) often center on **deep discussion of landmark papers**: "Walk me through the DPO derivation," "What was the key insight of Chinchilla?", "What would you change about InstructGPT?" The interviewer probes whether you understand the *contribution*, the *method*, the *limitations*, and the *follow-up work* — not just the abstract. This file gives a **must-know paper list** with, for each, the one-line contribution, the key mechanism, limitations, and likely follow-up questions. For any paper, be ready to discuss: **(1) the problem, (2) the key idea, (3) the method/math, (4) the evidence, (5) limitations, (6) what came after.**

## The Must-Know Papers

### Attention Is All You Need (Vaswani et al., 2017)
- **Contribution:** replaced recurrence with pure self-attention → parallelizable, scalable sequence modeling.
- **Key mechanism:** scaled dot-product + multi-head attention; encoder-decoder with positional encodings.
- **Limitations:** $O(n^2)$ attention; original post-norm unstable; sinusoidal positions.
- **Follow-ups to discuss:** decoder-only dominance, FlashAttention, RoPE, GQA. (See [Transformer Architecture](../00_foundations/00_transformer_architecture.md).)

### Chinchilla / Training Compute-Optimal LLMs (Hoffmann et al., 2022)
- **Contribution:** compute-optimal training scales $N$ and $D$ ~equally (~20 tokens/param); prior models undertrained.
- **Key mechanism:** IsoFLOP sweeps; the $L(N,D)$ power law.
- **Limitations:** ignores inference cost (→ overtraining paradigm); data-wall implications.
- **Follow-ups:** Kaplan discrepancy, data-constrained scaling, inference-time scaling. (See [Scaling Laws](../00_foundations/04_scaling_laws.md).)

### InstructGPT (Ouyang et al., 2022)
- **Contribution:** RLHF makes models useful; 1.3B aligned > 175B base in preference.
- **Key mechanism:** SFT → reward model (Bradley-Terry) → PPO with KL penalty.
- **Limitations:** human-label cost, reward hacking, alignment tax, sycophancy.
- **Follow-ups:** Constitutional AI, DPO, GRPO. (See [RLHF](../03_posttraining/03_RLHF.md).)

### Constitutional AI (Bai et al., 2022)
- **Contribution:** align with AI feedback guided by written principles, reducing human labels.
- **Key mechanism:** SL critique-revise + RL with an AI preference model (RLAIF).
- **Limitations:** whose values? labeler bias; ceiling at labeler competence.
- **Follow-ups:** RLAIF generality, scalable oversight, Model Spec/rule-based rewards. (See [CAI & RLAIF](../03_posttraining/07_constitutional_AI_and_RLAIF.md).)

### DPO (Rafailov et al., 2023)
- **Contribution:** preference alignment without an explicit RM or RL — a single supervised loss.
- **Key mechanism:** closed-form optimal policy → invert for reward → substitute into Bradley-Terry → $Z(x)$ cancels.
- **Limitations:** offline/off-policy; likelihood displacement; length bias.
- **Follow-ups:** IPO/KTO/SimPO/ORPO; online DPO; DPO-vs-PPO debate. (See [DPO](../03_posttraining/05_DPO_and_preference_optimization.md).)

### DeepSeek-R1 (DeepSeek-AI, 2025)
- **Contribution:** open reasoning model via pure RL with verifiable rewards; long CoT emerges (R1-Zero, no SFT).
- **Key mechanism:** GRPO (critic-free, group-relative advantage) + rule-based rewards; cold-start SFT + final RLHF for R1.
- **Limitations:** verifiable domains only; readability/language-mixing (R1-Zero); credit assignment.
- **Follow-ups:** GRPO variants (DAPO, Dr. GRPO), reasoning distillation, RLVR for agents. (See [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md).)

### LoRA (Hu et al., 2021)
- **Contribution:** parameter-efficient fine-tuning via low-rank weight updates; mergeable.
- **Key mechanism:** $\Delta W=BA$, rank $r$; freeze base.
- **Limitations:** capacity ceiling on large shifts; rank selection.
- **Follow-ups:** QLoRA, DoRA, LoRA+, multi-LoRA serving. (See [LoRA](../06_efficiency/01_LoRA_and_variants.md).)

### FlashAttention (Dao et al., 2022)
- **Contribution:** exact, IO-aware attention removing $O(n^2)$ memory; large speedups.
- **Key mechanism:** tiling + online softmax in SRAM.
- **Limitations:** hardware-specific kernels; superseded by FA-2/3.
- **Follow-ups:** FA-2/3 (Hopper, fp8), long-context training. (See [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md).)

### Llama 3 Technical Report (Grattafiori et al., 2024)
- **Contribution:** open, detailed recipe for a frontier model (data, scale, post-training, infra, fault tolerance).
- **Key mechanism:** ~15T tokens, dense + GQA + RoPE, SFT + rejection sampling + DPO, 16k-GPU fault-tolerant training.
- **Limitations:** dense (no MoE at launch), reasoning gap.
- **Follow-ups:** Llama 4 MoE; comparisons to DeepSeek/Qwen. (See [Meta Roadmap](../08_lab_roadmaps/03_meta_ai_roadmap.md).)

### Honorable mentions to be ready for
Scaling Monosemanticity / SAEs (interpretability), Let's Verify Step by Step (process supervision), Mixtral / DeepSeek-V3 (MoE), Mamba (SSMs), GPT-4 / Gemini reports, "Let's think step by step" (zero-shot CoT), Toy Models of Superposition.

## How to Discuss a Paper (the rubric interviewers use)
1. **Problem & motivation** — what was broken before?
2. **Key insight** — the one idea in a sentence.
3. **Method/math** — derive or sketch the core equation.
4. **Evidence** — what experiment proved it; any caveats in the eval?
5. **Limitations** — be honest and specific.
6. **Impact & follow-ups** — what it enabled and what improved on it.
7. **Critique** — what you'd do differently / what's still open.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Walk me through the DPO derivation."** Closed-form optimal policy → reward inversion → BT substitution → partition cancels.
- **"What was Chinchilla's key correction to Kaplan?"** LR schedule per horizon + parameter counting → equal $N,D$ scaling.
- **"What would you change about InstructGPT?"** E.g., DPO/GRPO to cut RM fragility; better length-debiasing; online iteration.
- **"Why was R1 a big deal scientifically?"** Reasoning emerges from pure RLVR; open, reproducible blueprint.

## References
All papers are cited with arXiv IDs in their respective deep-dive files (linked above).
