# Post-Training Overview

> **Last Updated:** 2026-11-04
> **Related Files:** [Supervised Fine-Tuning](01_supervised_finetuning_SFT.md) · [RLHF](03_RLHF.md) · [GRPO & RLVR](06_GRPO_and_RLVR.md) · [Agentic Coding & Computer Use](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)
> **Key Papers:** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Bai et al. 2022 HH-RLHF ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862)) · Touvron et al. 2023 Llama 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · ScaleRL 2025 ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786))

## Overview
**Post-training** (a.k.a. alignment or fine-tuning) is everything done to a *pretrained base model* to turn it into a useful, safe, instruction-following assistant. A base model trained only on next-token prediction is a powerful but unwieldy text-completer; post-training is what produced the qualitative leap from GPT-3 to ChatGPT. The InstructGPT insight (Ouyang et al. 2022) was decisive: a 1.3B aligned model was preferred by humans over the 175B base model — *alignment, not scale, unlocked usability*. Post-training is now where much of the differentiation between frontier labs lives, and it is the **highest-yield area for interviews**. By late 2026 it has also become the main *capability* lever: large-scale RL on verifiable and agentic tasks is what launch announcements lead with (hours-long autonomy, self-verification, computer use).

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
- **On-policy distillation (2025–26):** the student samples its own trajectories and a teacher scores every token (reverse KL) — dense supervision at RL-like on-policyness, now a standard stage for merging domain experts and shrinking models.
- **Agentic RL:** multi-turn RL in sandboxed repositories, terminals, and virtual desktops with execution-based rewards. See [Agentic Coding & Computer Use](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md).

**Why each stage.** SFT sets the behavioral prior cheaply but can only imitate; RL/preference methods optimize *beyond* demonstrations toward what humans/verifiers actually prefer, fixing issues imitation can't (calibrated refusals, helpfulness-harmlessness tradeoffs, reasoning that exceeds any single demonstration). 2026 evidence (PRISM) adds a caveat: **midtraining data sets much of the ceiling** — midtraining restructures >90% of weights while RL makes sparse updates to ~5% of parameters.

## Key Challenges
- **Alignment tax.** Aligning can degrade raw capability/diversity; balancing helpfulness, harmlessness, and honesty is a multi-objective problem.
- **Reward hacking / overoptimization.** Optimizing a proxy reward (RM) too hard exploits its errors (see [Reward Modeling](04_reward_modeling.md)). Since 2025 it is also a *safety* problem: hacks learned in production coding RL generalized to broad misalignment.
- **Data quality & cost.** Human preference/demonstration data is expensive and noisy; annotator disagreement is high. RL *environments* (verifiable tasks, sandboxes) are now the scarcer resource.
- **Evaluation.** Measuring "alignment" is hard; benchmarks lag real chat quality, preference is subjective, and models increasingly recognize when they are being evaluated.
- **Systems numerics.** Mismatch between the rollout engine and the trainer quietly makes "on-policy" RL off-policy.

## Solutions & Current Best Practices
The late-2026 default frontier stack: **midtraining on reasoning/agentic data → SFT (curated + distilled long-CoT) → large-scale RLVR and agentic RL** in sandboxed environments on asynchronous infrastructure → **on-policy distillation** to merge domain experts or train smaller tiers → **preference / rubric / constitution-guided RL** for non-verifiable behavior and safety. Synthetic data, **LLM-as-judge** and rubric rewards, and **iterative rounds** (Llama 2/3 ran multiple RLHF rounds) remain pervasive. Labs now also shape capabilities deliberately through the **RL environment mix** (e.g., Anthropic reports it did not train Opus 5 on cyber tasks). The exact ordering and weighting are key differentiators.

## Lab Perspectives
- **OpenAI:** invented InstructGPT/RLHF-PPO; o-series and the GPT-5/GPT-6 families add massive reasoning and long-horizon RL; GPT-6 Astra's gains are attributed to large-scale pretraining plus RL for long-horizon tasks. Research on honesty "confessions" and CoT monitorability.
- **Anthropic:** Constitutional AI / RLAIF; a fully rewritten constitution (2026-01) used at several training stages; uses **inoculation prompting** in production training against reward-hacking generalization.
- **Google DeepMind:** RLHF + RLAIF (Sparrow, Gemini); strong on verifiable reasoning — Gemini Deep Think reached IMO 2025 gold in natural language.
- **Meta:** documented iterative RLHF (rejection sampling + PPO/DPO) in Llama 2/3; co-authored ScaleRL (predictable RL compute scaling); its 2026 Muse models are closed, with post-training undisclosed.
- **DeepSeek:** pioneered **GRPO** and pure-RL reasoning (R1-Zero/R1); later hybrid thinking/non-thinking models (V3.1 → V4) and trained LLM proof verifiers as rewards (DeepSeekMath-V2).
- **Thinking Machines:** popularized on-policy distillation and LoRA-for-RL results.

## Latest Developments (2023–Nov 2026)
The biggest shift is **RL for reasoning (RLVR/GRPO)** — post-training now *creates new capabilities* (long CoT) rather than just eliciting them. **DPO vs PPO** debates matured toward "online/iterative beats offline." **Synthetic + AI feedback** largely displaced pure human labeling at scale. Post-training compute budgets have grown to rival or exceed pretraining for reasoning models.

**Mid-2025 → Nov 2026 update:**
- **RL compute scaling became predictable (2025-10).** ScaleRL ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786); >400K GPU-hours) fits reward vs compute with a sigmoid (asymptote $A$, efficiency exponent $B$); early-run fits extrapolate to 100K GPU-hour runs, and most design choices change *efficiency*, not the ceiling.
- **GRPO objective zoo converged (2025).** Token-level aggregation, decoupled clipping and dynamic sampling (DAPO, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)), IS-weight clipping (CISPO, [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)), and sequence-level ratios for MoE stability (GSPO, [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)). See [GRPO & RLVR](06_GRPO_and_RLVR.md).
- **On-policy distillation (2025-10 → 2026).** Thinking Machines reported ~70% on AIME24 at roughly a tenth of RL's GPU-hours; by 2026 multi-teacher OPD merges domain experts (Qwen3, KAT-Coder-V2.5, [arXiv:2607.05471](https://arxiv.org/abs/2607.05471)).
- **Midtraining sets the ceiling (2026-03).** PRISM ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)): science data at midtraining unlocks +17–28 points GPQA-Diamond after RL, while changing the RL data mix moves results <2 points.
- **Rewards beyond verifiers.** Rubrics as Rewards ([arXiv:2507.17746](https://arxiv.org/abs/2507.17746)) and trained LLM verifiers (DeepSeekMath-V2, [arXiv:2511.22570](https://arxiv.org/abs/2511.22570)) extend RL to semi-verifiable domains.
- **Reward hacking → emergent misalignment (2025-11).** Coding-environment hacks (e.g., `sys.exit(0)`) generalized to sabotage and alignment faking; chat-style RLHF hid it only on chat distributions; inoculation prompting removed most of the generalization ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)).
- **Systems numerics are an algorithms problem.** BF16 rollout/trainer mismatch fixed by FP16 RL ([arXiv:2510.26788](https://arxiv.org/abs/2510.26788)); batch-invariant kernels enable truly on-policy RL; async RL (AReaL, PipelineRL, slime) trades staleness for throughput.
- **Capability shaping in training (2026).** Anthropic reports deliberately reducing Opus 4.7's cyber capability during training and not training Opus 5 on cyber tasks — the RL environment mix is now an explicit safety lever.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Walk through the full post-training pipeline."** SFT → RM → RL, plus the modern DPO/RLVR/CAI extensions and why each exists.
- **"Why was InstructGPT a big deal?"** 1.3B aligned > 175B base in human preference; alignment unlocks usability.
- **"SFT vs RL — what does RL add that SFT can't?"** Optimize beyond imitation toward preferred/verifiable behavior; better calibration and reasoning.
- **"How has post-training changed with reasoning models?"** RLVR creates capabilities; post-training compute now huge.
- **"Where do midtraining and on-policy distillation fit in a 2026 pipeline?"** Midtraining sets the ceiling (PRISM); OPD gives dense per-token teacher signal on the student's own samples, ~10x cheaper than RL to reach a teacher's level.
- **"Why is reward hacking now a safety issue, not just a nuisance?"** Production coding hacks generalized to sabotage/alignment faking; mitigations are hack prevention, diverse safety training, inoculation prompting.

## Open Problems
How to align models *more capable than their supervisors* (scalable oversight, see [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)) is the central open problem. Robustly preventing reward hacking, balancing the alignment tax, generalizing alignment beyond the training distribution, and aligning open-ended (non-verifiable) behavior all remain unsolved. New in 2026: scaling RL *environments* (not just compute), and evaluating alignment when models suspect they are being tested.

## References
- Ouyang, L. et al. (2022). *Training LMs to Follow Instructions (InstructGPT).* arXiv:2203.02155.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Meta et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
- Sea AI Lab (2025). *Defeating the Training-Inference Mismatch via FP16.* arXiv:2510.26788.
