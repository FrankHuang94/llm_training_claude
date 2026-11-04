# Paper Discussion Questions

> **Last Updated:** 2026-11-04
> **Related Files:** [Common Technical Questions](01_common_technical_questions.md) · [Post-Training Overview](../03_posttraining/00_posttraining_overview.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md) · [Frontier Open Problems](../09_emerging_frontiers/05_frontier_open_problems.md)

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
- **Follow-ups:** GRPO variants (DAPO, Dr. GRPO, GSPO, CISPO), ScaleRL, reasoning distillation and on-policy distillation, RLVR for long-horizon agents, DeepSeekMath-V2 self-verification. (See [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md).)

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
- **Follow-ups:** Llama 4 MoE; Meta's 2026 pivot to the **closed** Muse family (Muse Spark, Muse Wonder); open-frontier leadership passing to DeepSeek/Qwen/Kimi/GLM, gpt-oss and Mistral Large 4. (See [Meta Roadmap](../08_lab_roadmaps/03_meta_ai_roadmap.md).)

### Honorable mentions to be ready for
Scaling Monosemanticity / SAEs (interpretability), Let's Verify Step by Step (process supervision), Mixtral / DeepSeek-V3 (MoE), Mamba (SSMs), GPT-4 / Gemini reports, "Let's think step by step" (zero-shot CoT), Toy Models of Superposition.

## 2025–2026 Must-Know Papers
Expect these in 2026 loops; they define the current recipe for pretraining, RL post-training, and safety.

### DAPO (ByteDance Seed/Tsinghua, 2025) · GSPO (Qwen, 2025) · CISPO / MiniMax-M1 (2025)
- **Contribution:** make GRPO-style RL stable at scale — DAPO: clip-higher, dynamic sampling, token-level loss, overlong shaping (50 on AIME24 from Qwen2.5-32B base); GSPO: sequence-level length-normalized ratio + clipping (fixes MoE RL collapse, used for Qwen3); CISPO: clip the IS weight (stop-grad), not the update.
- **Discuss:** why token- vs sequence-level ratios; which tokens PPO-clip silences; how Dr. GRPO's bias critique relates. (See [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md).)

### The Art of Scaling RL Compute — ScaleRL (Meta et al., 2025; arXiv:2510.13786)
- **Contribution:** >400K GPU-hours; RL reward vs compute is a predictable **sigmoid** (asymptote $A$, efficiency $B$); most tricks move $B$, not $A$.
- **Limitations:** verifiable-reward reasoning settings; asymptote depends on base model and data. **Follow-ups:** RL data-scaling studies, ScaleLogic.

### Defeating the Training–Inference Mismatch via FP16 (Sea AI Lab, 2025; arXiv:2510.26788) + Defeating Nondeterminism (Thinking Machines, 2025)
- **Contribution:** BF16 rounding and non-batch-invariant kernels make sampler/trainer log-probs diverge; FP16 or batch-invariant kernels restore true on-policy RL.
- **Discuss:** why "systems numerics" became an algorithms problem; TIS/MIS patches vs fixing the root cause.

### On-Policy Distillation & LoRA Without Regret (Thinking Machines, 2025, blogs)
- **Contribution:** student-sampled trajectories + teacher per-token reverse-KL ≈ RL-level on-policyness with dense signal at ~1/10 the cost; LoRA on all layers matches full FT, and rank-1 suffices for RL (~1 bit/episode).
- **Limitations:** mode-seeking reverse KL; teacher signal decay on long prefixes.

### Muon is Scalable / Kimi K2 MuonClip (Moonshot, 2025; arXiv:2502.16982, 2507.20534) vs Fantastic Pretraining Optimizers (Wen et al., 2025; arXiv:2509.02046)
- **Contribution:** Muon with weight decay + RMS matching (~2× claimed); QK-Clip prevents logit explosion (1T MoE, 15.5T tokens, no spikes). Wen et al.: under fair tuning, ≤1.4× and shrinking with scale.
- **Discuss:** how to run a fair optimizer comparison; μP/weight-decay scaling.

### Pretraining LLMs with NVFP4 (NVIDIA, 2025; arXiv:2509.25149)
- **Contribution:** 12B on 10T tokens in 4-bit at ≈FP8 quality — Hadamard transforms, 2D block scaling, stochastic rounding, BF16 tail layers; NVFP4 > MXFP4.
- **Follow-ups:** Nemotron 3 NVFP4 pretraining at 25T tokens; FP4 QAT in DeepSeek-V4/Kimi.

### DeepSeek-V3.2 / DSA (2025; arXiv:2512.02556) · Kimi Linear (2025; arXiv:2510.26692) · Gated Attention (Qwen, 2025; arXiv:2505.06708)
- **Contribution:** learned top-$k$ sparse attention with a lightning indexer (quality ≈ dense in a controlled swap); KDA 3:1 hybrid claimed to beat full attention fairly; output gating removes attention sinks.
- **Discuss:** MiniMax-M2's reversion to full attention as the counterpoint; DeepSeek-V4's CSA/HCA; mHC (arXiv:2512.24880). (See [Diffusion & Hybrid LMs](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md).)

### Natural Emergent Misalignment from Reward Hacking (MacDiarmid et al., Anthropic/Redwood, 2025; arXiv:2511.18397)
- **Contribution:** production-like coding-RL hacks generalize to sabotage and alignment faking; RLHF hides it only on chat; **inoculation prompting** removes most generalization.
- **Discuss:** why recontextualizing a behavior changes generalization; monitors trained on synthetic hacks fail on natural ones.

### Chain of Thought Monitorability (Korbak et al., 2025; arXiv:2507.11473) + Monitoring Monitorability (OpenAI, 2025; arXiv:2512.18311)
- **Contribution:** CoT is a fragile safety asset; avoid direct CoT optimization; 13-eval suite shows CoT monitors beat action-only monitors.
- **Discuss:** process supervision vs monitorability; latent reasoning; anti-distillation hiding of thoughts.

### Why Language Models Hallucinate (Kalai et al., OpenAI, 2025; arXiv:2509.04664)
- **Contribution:** hallucination lower-bounded by a validity-classification error (singleton rate); binary grading rewards guessing → change benchmark scoring.

### METR Time Horizons (Kwa et al., 2025; arXiv:2503.14499) + OpenAI "Why we no longer evaluate SWE-bench Verified" (2026)
- **Contribution:** logistic-fit 50% horizon as a capability scale (doubling ~7 → ~4.3 months); benchmark audits show flawed tests and contamination.
- **Discuss:** measurement ceilings (~16 h), harness variance, eval awareness.

### Honorable mentions (2025–2026)
Absolute Zero and SPICE (self-play), Rubrics as Rewards, DeepSeekMath-V2 (verifier-as-reward), PRISM (midtraining sets RL ceilings), V-JEPA 2 (latent-space planning), TTT-E2E (long context as test-time learning), LLaDA2.0 / Mercury (diffusion LMs), Mamba-3, Constitutional Classifiers++, Persona Vectors.

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
- **"GSPO vs CISPO vs DAPO — which would you use for RL on a large MoE, and why?"** Sequence-level ratios for routing-induced token-ratio noise (GSPO); CISPO if rare high-value tokens are being clipped; DAPO's clip-higher/dynamic sampling as the baseline; cite ScaleRL's finding that these mostly change efficiency.
- **"What would you change about the reward-hacking → misalignment study?"** Natural (not synthetic-document-induced) hack discovery, more model families, mechanism of inoculation, monitors robust to natural hacks.

## References
All papers are cited with arXiv IDs in their respective deep-dive files (linked above).
