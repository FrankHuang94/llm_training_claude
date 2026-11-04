# Rejection Sampling and Best-of-N

> **Last Updated:** 2026-11-04
> **Related Files:** [GRPO & RLVR](06_GRPO_and_RLVR.md) · [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md) · [Reward Modeling](04_reward_modeling.md)
> **Key Papers:** Dong et al. 2023 RAFT ([arXiv:2304.06767](https://arxiv.org/abs/2304.06767)) · Zelikman et al. 2022 STaR ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) · Touvron et al. 2023 Llama 2 (rejection sampling) ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · Nakano et al. 2021 WebGPT (best-of-N) ([arXiv:2112.09332](https://arxiv.org/abs/2112.09332))

## Overview
Rejection sampling and best-of-N (BoN) are the simplest and most robust tools in the alignment toolbox: **generate many candidate responses, keep the best (by a reward model or verifier), and either return it (inference time) or train on it (post-training)**. They sit between pure SFT and full RL — capturing much of RL's "optimize toward what's preferred" benefit with far less complexity and instability. As a *training* method ("rejection sampling fine-tuning," RAFT, STaR), they create a self-improvement loop; as an *inference* method (BoN), they trade compute for quality. Both are heavily used in production (Llama 2/3) and are conceptual ancestors of modern inference-time scaling and RLVR. In 2025–26 the same idea reappears at agent scale: rejection fine-tuning on verified repository trajectories cold-starts coding-agent RL, and "parallel test-time compute" is BoN over whole agent runs.

## Core Concepts
**Best-of-N (inference).** Sample $N$ responses from the policy, score each with a reward model or verifier, return the top-scoring one. Quality improves with $N$, but so does the **KL from the base policy**: BoN's effective KL grows like $\log N - \frac{N-1}{N}$, meaning it implicitly performs reward maximization with a KL cost — making BoN a strong *baseline* for what RLHF should achieve. It is simple, parallelizable, and needs no training, but multiplies inference cost by $N$.

**Rejection sampling fine-tuning (training).** Run BoN to collect the best responses across many prompts, then **SFT the policy on those winners** (optionally discarding low-reward samples entirely — "rejection"). This distills BoN's gains back into the weights so future single samples are better — an offline, stable alternative to PPO. Iterating (generate → filter → fine-tune → repeat) yields **expert iteration / iterative self-improvement**.

**STaR (Self-Taught Reasoner).** For tasks with checkable answers: sample reasoning + answer, **keep only traces that reach the correct answer** (verifier = answer match), and fine-tune on them. For failures, optionally provide the answer as a hint to generate a "rationalized" correct trace. This bootstraps reasoning from a model's own successful rollouts — a direct precursor to RLVR (see [GRPO & RLVR](06_GRPO_and_RLVR.md)).

**RAFT (Reward-rAnked Fine-Tuning).** Formalizes the generate→rank-by-reward→fine-tune loop as a general, stable alignment algorithm, competitive with PPO on many tasks while being simpler.

**Relation to RL.** Rejection sampling is essentially **one step of policy improvement with a hard (top-k) filter** instead of a gradient through the reward. RLVR/GRPO can be seen as a softer, on-policy generalization; in fact many reasoning pipelines alternate rejection-sampling SFT and RL. Filtering also lives *inside* modern RL: DAPO's dynamic sampling and ScaleRL's zero-variance filtering discard prompt groups whose samples are all correct or all wrong, since their group-relative advantage is zero.

## Key Challenges
- **Compute cost.** Generating $N$ samples per prompt (often $N=4$–$64$+) is expensive, especially for long reasoning chains and multi-hour agent runs.
- **Reward-model dependence.** BoN/rejection are only as good as the scorer; a biased RM selects biased "best" samples (length bias is common). Verifiers avoid this where available, but flawed tests select flawed solutions.
- **Diversity collapse / over-optimization.** Repeatedly training on self-generated winners can narrow the distribution and amplify the RM's blind spots (a Goodhart risk).
- **Coverage limits.** If the base policy never samples a good response (low probability of success), filtering can't help — you can't select what you don't generate.

## Solutions & Current Best Practices
Use **verifiers** (math/code/exact-match) as the selector wherever possible; otherwise a robust RM with **length-debiasing** and ensembling, or a trained **self-verifier**. **Iterate** generate→filter→SFT in rounds (expert iteration), often interleaved with DPO/RL. Maintain **diversity** (temperature, varied prompts) to keep coverage. **Llama 2/3** used rejection sampling on the best of many samples (scored by RMs) as a core part of alignment, alternating with PPO/DPO. For reasoning, STaR-style "keep-correct-traces" SFT plus RLVR is the modern recipe; for agents, rejection fine-tuning on execution-verified trajectories precedes multi-turn RL. When a strong teacher exists, **on-policy distillation** replaces the hard filter with dense per-token teacher signal. At inference, BoN and **weighted/self-consistency voting** are standard test-time-compute levers (see [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md)).

## Lab Perspectives
- **Meta:** Llama 2/3 prominently used **rejection-sampling fine-tuning** (best-of-N by RM) as a major alignment stage, openly documented.
- **OpenAI:** WebGPT used best-of-N against a reward model; rejection sampling underlies much practical data generation; o-series leans on verified self-generated reasoning.
- **DeepSeek:** R1 used rejection sampling of correct/high-quality traces to build SFT data between RL stages; DeepSeekMath-V2 scales *verification* compute to select and label hard proofs.
- **Anthropic:** reports agentic-coding scores with and without "parallel test-time compute" (BoN-style selection over agent attempts) and trains for self-verification of the model's own code.
- **Google DeepMind/AllenAI:** expert-iteration and STaR-style self-improvement for reasoning and formal math.

## Latest Developments (2023–Nov 2026)
Rejection sampling became a **standard bridge** between SFT and RL: generate with the current model, **verify/score, keep winners, SFT, repeat** — used to bootstrap reasoning data for R1/o1-style models and to create distillation corpora. Combined with **inference-time scaling** research (how BoN/self-consistency/verifier-guided search trade compute for accuracy) and with **RLVR** (the on-policy, gradient-based generalization). The simplicity and stability of rejection sampling keep it popular even as RL methods mature.

**Mid-2025 → Nov 2026 update:**
- **Rejection fine-tuning for SWE agents (2025-08).** Nebius trained long-context, multi-turn SWE agents with rejection fine-tuning on real-repository trajectories followed by DAPO-style multi-turn RL — now a common baseline ([arXiv:2508.03501](https://arxiv.org/abs/2508.03501)).
- **BoN over agent runs (2025-09).** Claude Sonnet 4.5 reported 77.2% on SWE-bench Verified, rising to 82.0% with parallel test-time compute (vendor-reported) — selection over complete agent trajectories.
- **Filtering inside RL (2025).** Dynamic sampling (DAPO, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) and zero-variance filtering plus no-positive-resampling (ScaleRL, [arXiv:2510.13786](https://arxiv.org/abs/2510.13786)) turn rejection into a compute-efficiency device for GRPO.
- **Scaled self-verification as the selector (2025-11 → 2026).** DeepSeekMath-V2 rewards a generator for finding and fixing its own errors and scales verifier compute ([arXiv:2511.22570](https://arxiv.org/abs/2511.22570)); Anthropic reports Opus 4.8 is ~4× less likely than Opus 4.7 to let flaws in its own code pass.
- **Attacking the coverage limit with self-play.** Absolute Zero ([arXiv:2505.03335](https://arxiv.org/abs/2505.03335)) has one model propose maximally learnable code-reasoning tasks and solve them with an executor as verifier; SPICE ([arXiv:2510.24684](https://arxiv.org/abs/2510.24684)) grounds proposer/solver self-play in a document corpus to avoid collapse.
- **Synthesized verifiable tasks (2026).** KAT-Coder-V2.5 uses agent-built verifiable repository tasks plus multi-teacher on-policy distillation ([arXiv:2607.05471](https://arxiv.org/abs/2607.05471)) — rejection sampling's selector problem moves into environment construction.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How does best-of-N relate to RLHF?"** BoN implicitly maximizes reward with a $\log N$-scaling KL cost — a strong baseline for what RL should beat.
- **"What is rejection sampling fine-tuning and why use it over PPO?"** Generate→filter-by-reward→SFT on winners; simpler, stable, offline, no value net.
- **"Explain STaR."** Keep only self-generated reasoning traces that reach the correct answer; fine-tune on them to bootstrap reasoning.
- **"Limitation of rejection sampling?"** Can't select responses the policy never generates; RM bias and compute cost.
- **"Why does GRPO drop all-correct or all-wrong groups?"** Their group-relative advantages are identically zero — no gradient, only cost; filtering them (dynamic sampling) is rejection sampling at the prompt level.

## Open Problems
The coverage limit (you can only filter what you can sample) bounds self-improvement — how to push the policy into genuinely new behaviors without external signal is open (and connects to the RLVR "elicit vs create" debate). Avoiding diversity collapse and Goodharting under repeated self-training, and the optimal interleaving of rejection sampling, DPO, and RL, remain active questions. Whether self-play task generation can sustain open-ended improvement without collapse is unresolved.

## References
- Dong, H. et al. (2023). *RAFT.* arXiv:2304.06767.
- Zelikman, E. et al. (2022). *STaR.* arXiv:2203.14465.
- Touvron, H. et al. (2023). *Llama 2 (rejection sampling).* arXiv:2307.09288.
- Nakano, R. et al. (2021). *WebGPT (best-of-N).* arXiv:2112.09332.
- Anthony, T. et al. (2017). *Thinking Fast and Slow (Expert Iteration).* arXiv:1705.08439.
- Zhao, A. et al. (2025). *Absolute Zero.* arXiv:2505.03335.
- Nebius (2025). Long-context multi-turn SWE agents trained with rejection fine-tuning + RL (descriptive title). arXiv:2508.03501.
