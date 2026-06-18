# Rejection Sampling and Best-of-N

> **Last Updated:** 2026-06-18
> **Related Files:** [GRPO & RLVR](06_GRPO_and_RLVR.md) · [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md) · [Reward Modeling](04_reward_modeling.md)
> **Key Papers:** Dong et al. 2023 RAFT ([arXiv:2304.06767](https://arxiv.org/abs/2304.06767)) · Zelikman et al. 2022 STaR ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) · Touvron et al. 2023 Llama 2 (rejection sampling) ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · Nakano et al. 2021 WebGPT (best-of-N) ([arXiv:2112.09332](https://arxiv.org/abs/2112.09332))

## Overview
Rejection sampling and best-of-N (BoN) are the simplest and most robust tools in the alignment toolbox: **generate many candidate responses, keep the best (by a reward model or verifier), and either return it (inference time) or train on it (post-training)**. They sit between pure SFT and full RL — capturing much of RL's "optimize toward what's preferred" benefit with far less complexity and instability. As a *training* method ("rejection sampling fine-tuning," RAFT, STaR), they create a self-improvement loop; as an *inference* method (BoN), they trade compute for quality. Both are heavily used in production (Llama 2/3) and are conceptual ancestors of modern inference-time scaling and RLVR.

## Core Concepts
**Best-of-N (inference).** Sample $N$ responses from the policy, score each with a reward model or verifier, return the top-scoring one. Quality improves with $N$, but so does the **KL from the base policy**: BoN's effective KL grows like $\log N - \frac{N-1}{N}$, meaning it implicitly performs reward maximization with a KL cost — making BoN a strong *baseline* for what RLHF should achieve. It is simple, parallelizable, and needs no training, but multiplies inference cost by $N$.

**Rejection sampling fine-tuning (training).** Run BoN to collect the best responses across many prompts, then **SFT the policy on those winners** (optionally discarding low-reward samples entirely — "rejection"). This distills BoN's gains back into the weights so future single samples are better — an offline, stable alternative to PPO. Iterating (generate → filter → fine-tune → repeat) yields **expert iteration / iterative self-improvement**.

**STaR (Self-Taught Reasoner).** For tasks with checkable answers: sample reasoning + answer, **keep only traces that reach the correct answer** (verifier = answer match), and fine-tune on them. For failures, optionally provide the answer as a hint to generate a "rationalized" correct trace. This bootstraps reasoning from a model's own successful rollouts — a direct precursor to RLVR (see [GRPO & RLVR](06_GRPO_and_RLVR.md)).

**RAFT (Reward-rAnked Fine-Tuning).** Formalizes the generate→rank-by-reward→fine-tune loop as a general, stable alignment algorithm, competitive with PPO on many tasks while being simpler.

**Relation to RL.** Rejection sampling is essentially **one step of policy improvement with a hard (top-k) filter** instead of a gradient through the reward. RLVR/GRPO can be seen as a softer, on-policy generalization; in fact many reasoning pipelines alternate rejection-sampling SFT and RL.

## Key Challenges
- **Compute cost.** Generating $N$ samples per prompt (often $N=4$–$64$+) is expensive, especially for long reasoning chains.
- **Reward-model dependence.** BoN/rejection are only as good as the scorer; a biased RM selects biased "best" samples (length bias is common). Verifiers avoid this where available.
- **Diversity collapse / over-optimization.** Repeatedly training on self-generated winners can narrow the distribution and amplify the RM's blind spots (a Goodhart risk).
- **Coverage limits.** If the base policy never samples a good response (low probability of success), filtering can't help — you can't select what you don't generate.

## Solutions & Current Best Practices
Use **verifiers** (math/code/exact-match) as the selector wherever possible (non-hackable); otherwise a robust RM with **length-debiasing** and ensembling. **Iterate** generate→filter→SFT in rounds (expert iteration), often interleaved with DPO/RL. Maintain **diversity** (temperature, varied prompts) to keep coverage. **Llama 2/3** used rejection sampling on the best of many samples (scored by RMs) as a core part of alignment, alternating with PPO/DPO. For reasoning, STaR-style "keep-correct-traces" SFT plus RLVR is the modern recipe. At inference, BoN and **weighted/self-consistency voting** are standard test-time-compute levers (see [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md)).

## Lab Perspectives
- **Meta:** Llama 2/3 prominently used **rejection-sampling fine-tuning** (best-of-N by RM) as a major alignment stage, openly documented.
- **OpenAI:** WebGPT used best-of-N against a reward model; rejection sampling underlies much practical data generation; o-series leans on verified self-generated reasoning.
- **DeepSeek:** R1 used rejection sampling of correct/high-quality traces to build SFT data between RL stages.
- **Google DeepMind/AllenAI:** expert-iteration and STaR-style self-improvement for reasoning and formal math.

## Latest Developments (2023–2026)
Rejection sampling became a **standard bridge** between SFT and RL: generate with the current model, **verify/score, keep winners, SFT, repeat** — used to bootstrap reasoning data for R1/o1-style models and to create distillation corpora. Combined with **inference-time scaling** research (how BoN/self-consistency/verifier-guided search trade compute for accuracy) and with **RLVR** (the on-policy, gradient-based generalization). The simplicity and stability of rejection sampling keep it popular even as RL methods mature.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How does best-of-N relate to RLHF?"** BoN implicitly maximizes reward with a $\log N$-scaling KL cost — a strong baseline for what RL should beat.
- **"What is rejection sampling fine-tuning and why use it over PPO?"** Generate→filter-by-reward→SFT on winners; simpler, stable, offline, no value net.
- **"Explain STaR."** Keep only self-generated reasoning traces that reach the correct answer; fine-tune on them to bootstrap reasoning.
- **"Limitation of rejection sampling?"** Can't select responses the policy never generates; RM bias and compute cost.

## Open Problems
The coverage limit (you can only filter what you can sample) bounds self-improvement — how to push the policy into genuinely new behaviors without external signal is open (and connects to the RLVR "elicit vs create" debate). Avoiding diversity collapse and Goodharting under repeated self-training, and the optimal interleaving of rejection sampling, DPO, and RL, remain active questions.

## References
- Dong, H. et al. (2023). *RAFT.* arXiv:2304.06767.
- Zelikman, E. et al. (2022). *STaR.* arXiv:2203.14465.
- Touvron, H. et al. (2023). *Llama 2 (rejection sampling).* arXiv:2307.09288.
- Nakano, R. et al. (2021). *WebGPT (best-of-N).* arXiv:2112.09332.
- Anthony, T. et al. (2017). *Thinking Fast and Slow (Expert Iteration).* arXiv:1705.08439.
