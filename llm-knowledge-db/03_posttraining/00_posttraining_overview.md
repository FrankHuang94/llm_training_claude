# Post-Training Overview

> **Last Updated:** 2026-06-18
> **Related Files:** [Supervised Fine-Tuning](01_supervised_finetuning_SFT.md) · [RLHF](03_RLHF.md) · [GRPO & RLVR](06_GRPO_and_RLVR.md)
> **Key Papers:** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Bai et al. 2022 HH-RLHF ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862)) · Touvron et al. 2023 Llama 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948))

## Overview
**Post-training** (a.k.a. alignment or fine-tuning) is everything done to a *pretrained base model* to turn it into a useful, safe, instruction-following assistant. A base model trained only on next-token prediction is a powerful but unwieldy text-completer; post-training is what produced the qualitative leap from GPT-3 to ChatGPT. The InstructGPT insight (Ouyang et al. 2022) was decisive: a 1.3B aligned model was preferred by humans over the 175B base model — *alignment, not scale, unlocked usability*. Post-training is now where much of the differentiation between frontier labs lives, and it is the **highest-yield area for interviews**.

## Core Concepts
**The canonical pipeline.** The classic three-stage recipe:
1. **SFT (Supervised Fine-Tuning):** imitate high-quality demonstrations (instruction→response), teaching format, instruction-following, and chat behavior. See [SFT](01_supervised_finetuning_SFT.md).
2. **Reward Modeling (RM):** train a model of human preferences from pairwise comparisons (Bradley-Terry). See [Reward Modeling](04_reward_modeling.md).
3. **RL optimization (RLHF/PPO):** optimize the policy against the RM with a KL penalty to the SFT reference. See [RLHF](03_RLHF.md).

**The modern, expanded stack.** Since 2023 the pipeline has broadened:
- **Preference optimization without RL:** DPO and variants directly optimize on preference pairs, no separate RM or rollout. See [DPO](05_DPO_and_preference_optimization.md).
- **RLVR / GRPO:** RL with *verifiable* rewards (math/code correctness) drives reasoning; DeepSeek-R1 showed pure RL can induce long chain-of-thought. See [GRPO & RLVR](06_GRPO_and_RLVR.md).
- **Constitutional AI / RLAIF:** replace human feedback with AI feedback guided by principles. See [CAI & RLAIF](07_constitutional_AI_and_RLAIF.md).
- **Rejection sampling / best-of-N / iterative SFT:** generate, filter by reward/verifier, retrain (RAFT, STaR, "rejection sampling fine-tuning"). See [Rejection Sampling](08_rejection_sampling_and_best_of_N.md).

**Why each stage.** SFT sets the behavioral prior cheaply but can only imitate; RL/preference methods optimize *beyond* demonstrations toward what humans/verifiers actually prefer, fixing issues imitation can't (calibrated refusals, helpfulness-harmlessness tradeoffs, reasoning that exceeds any single demonstration).

## Key Challenges
- **Alignment tax.** Aligning can degrade raw capability/diversity; balancing helpfulness, harmlessness, and honesty is a multi-objective problem.
- **Reward hacking / overoptimization.** Optimizing a proxy reward (RM) too hard exploits its errors (see [Reward Modeling](04_reward_modeling.md)).
- **Data quality & cost.** Human preference/demonstration data is expensive and noisy; annotator disagreement is high.
- **Evaluation.** Measuring "alignment" is hard; benchmarks lag real chat quality, and preference is subjective.

## Solutions & Current Best Practices
The 2026 default stack at most labs: **SFT on curated/synthetic instructions → preference optimization** (DPO or PPO-class, often *online/iterative*) **+ RLVR for reasoning** + safety-specific RL/CAI, with heavy use of **synthetic data**, **LLM-as-judge** evaluation, and **iterative rounds** (Llama 2/3 ran multiple RLHF rounds). Reasoning models add a large RL phase on verifiable tasks. The exact ordering and weighting are key differentiators.

## Lab Perspectives
- **OpenAI:** invented InstructGPT/RLHF-PPO; o-series adds massive RLVR-style reasoning RL.
- **Anthropic:** Constitutional AI / RLAIF, safety-first, helpful-honest-harmless framing.
- **Google DeepMind:** RLHF + RLAIF (Sparrow, Gemini); strong on verifiable reasoning (AlphaProof).
- **Meta:** documented iterative RLHF (rejection sampling + PPO/DPO) in Llama 2/3, open recipes.
- **DeepSeek:** pioneered **GRPO** and pure-RL reasoning (R1-Zero/R1), distilling reasoning into small models.

## Latest Developments (2023–2026)
The biggest shift is **RL for reasoning (RLVR/GRPO)** — post-training now *creates new capabilities* (long CoT) rather than just eliciting them. **DPO vs PPO** debates matured toward "online/iterative beats offline." **Synthetic + AI feedback** largely displaced pure human labeling at scale. Post-training compute budgets have grown to rival or exceed pretraining for reasoning models.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Walk through the full post-training pipeline."** SFT → RM → RL, plus the modern DPO/RLVR/CAI extensions and why each exists.
- **"Why was InstructGPT a big deal?"** 1.3B aligned > 175B base in human preference; alignment unlocks usability.
- **"SFT vs RL — what does RL add that SFT can't?"** Optimize beyond imitation toward preferred/verifiable behavior; better calibration and reasoning.
- **"How has post-training changed with reasoning models?"** RLVR creates capabilities; post-training compute now huge.

## Open Problems
How to align models *more capable than their supervisors* (scalable oversight, see [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)) is the central open problem. Robustly preventing reward hacking, balancing the alignment tax, generalizing alignment beyond the training distribution, and aligning open-ended (non-verifiable) behavior all remain unsolved.

## References
- Ouyang, L. et al. (2022). *Training LMs to Follow Instructions (InstructGPT).* arXiv:2203.02155.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
