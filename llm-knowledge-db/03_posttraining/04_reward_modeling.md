# Reward Modeling

> **Last Updated:** 2026-06-18
> **Related Files:** [RLHF](03_RLHF.md) · [GRPO & RLVR](06_GRPO_and_RLVR.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)
> **Key Papers:** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Gao et al. 2022 Reward Overoptimization ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)) · Lightman et al. 2023 "Let's Verify Step by Step" (PRM) ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) · Coste et al. 2023 Reward Model Ensembles ([arXiv:2310.02743](https://arxiv.org/abs/2310.02743))

## Overview
The reward model (RM) is the learned proxy for human (or AI) preference that turns subjective judgments into a differentiable training signal. It is the linchpin — and the weakest link — of the RLHF pipeline: the policy can only be as aligned as the RM is accurate, and every RM is an imperfect proxy that a powerful optimizer will try to exploit (Goodhart's law in action). Reward modeling has expanded from single scalar preference models to **process reward models (PRMs)** for reasoning, **verifiable rewards** for math/code, and **generative reward models / LLM-as-judge**.

## Core Concepts
**Bradley-Terry preference model.** The standard RM predicts a scalar reward $r_\phi(x,y)$ such that
$$P(y_w \succ y_l \mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),$$
trained on human pairwise comparisons via the logistic loss $-\log\sigma(r_w-r_l)$. The RM is usually the SFT model with the LM head replaced by a scalar value head, trained on the same backbone. Rewards are only meaningful *relatively* (the scale/offset is arbitrary), so they're typically normalized.

**Outcome vs Process Reward Models (ORM vs PRM).** An **ORM** scores only the *final answer* (correct/incorrect or preferred). A **PRM** scores *each reasoning step*, giving dense feedback. Lightman et al. (2023, "Let's Verify Step by Step") showed **process supervision** trains more reliable math reward models than outcome supervision and is better for verifier-guided search — a foundational result for reasoning models. PRMs require step-level labels (human or automatically generated via Monte-Carlo rollouts, e.g., Math-Shepherd).

**Verifiable rewards.** For math/code/logic, the "RM" can be a **rule/verifier** (unit tests, a symbolic checker, exact-match) — a perfect, non-hackable reward for that domain. This underpins RLVR/GRPO (see [GRPO & RLVR](06_GRPO_and_RLVR.md)) and sidesteps RM fragility where applicable.

**Generative / LLM-as-judge RMs.** Instead of a scalar head, prompt a strong LLM to *judge* or *critique* responses (optionally with CoT), yielding more interpretable, sometimes more robust rewards (e.g., for RLAIF, see [CAI](07_constitutional_AI_and_RLAIF.md)).

## Key Challenges
- **Reward hacking / overoptimization.** Optimizing the proxy too hard exploits its errors; true preference rises then falls with KL from reference (Gao et al. scaling law).
- **Spurious correlations / length bias.** RMs latch onto superficial features — notably **length** (longer = "better"), formatting, sycophancy — rather than true quality.
- **Distribution shift.** RMs trained on SFT-era responses degrade as the policy moves off-distribution, requiring iterative re-collection.
- **Label noise & disagreement.** Human annotators disagree substantially; preferences are subjective and context-dependent.
- **Calibration & generalization.** RMs may be over-confident and fail to generalize to new domains or harder examples.

## Solutions & Current Best Practices
**RM ensembles / uncertainty** (penalize high-variance, OOD inputs) reduce overoptimization (Coste et al., WARM weight-averaged RMs). **Length-debiasing** (length-controlled metrics, length penalties) combats the dominant spurious feature. **Process supervision (PRMs)** for reasoning; **verifiable rewards** wherever the domain allows (the most robust option). **Iterative preference collection** keeps the RM on-distribution. Larger/stronger RMs and high-quality, deduplicated preference data help. **RewardBench** (2024) standardized RM evaluation. Monitor RM accuracy on held-out comparisons and watch the reward-vs-KL curve during RL.

## Lab Perspectives
- **OpenAI** pioneered scalar RMs (InstructGPT) and **process supervision** (Lightman et al.) feeding o-series reasoning.
- **Anthropic** uses preference models for HH and **AI-feedback** RMs (Constitutional AI).
- **Google DeepMind** explored PRMs, RLAIF judge models, and verifiable rewards (AlphaProof).
- **DeepSeek** leaned on **rule-based verifiable rewards** (GRPO) and argued *against* neural PRMs for RL at scale due to hacking/cost (in the R1 report).
- **Meta** documented iterative RM training and length-bias mitigation in Llama 2/3.

## Latest Developments (2023–2026)
A notable swing toward **verifiable/rule-based rewards** for reasoning (DeepSeek-R1 deliberately avoided learned PRMs in its main RL, citing reward hacking), even as PRMs remain valuable for search/verification. **Generative reward models** and **LLM-as-judge** matured (with known biases: position, verbosity, self-preference). **RewardBench** and reward-hacking studies sharpened evaluation. Weight-averaged and ensemble RMs (WARM) address robustness.

## Interview Angles
> 💡 **What labs actually ask:**
- **"PRM vs ORM — what, and which is better for reasoning?"** Step-level vs final-answer reward; PRMs give denser, more reliable signal (Lightman et al.) and better verifier-guided search.
- **"How do reward models get hacked, and how do you mitigate it?"** Proxy exploitation, length/format bias; ensembles, length-debias, verifiable rewards, iterative data, KL control.
- **"Why are verifiable rewards attractive?"** Perfect, non-hackable signal for math/code → enables stable, scalable RL (RLVR).
- **"Write and explain the Bradley-Terry RM loss."** $-\log\sigma(r_w-r_l)$; rewards meaningful only relatively.

## Open Problems
Building reward models robust to a strong optimizer (anti-Goodhart) is unsolved. Reward modeling for **open-ended, non-verifiable** tasks (creativity, helpfulness, honesty) is fundamentally hard. Reducing reliance on noisy human labels, debiasing (length/sycophancy) without losing signal, and reward modeling under scalable oversight (super-human outputs) are central open challenges.

## References
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760.
- Lightman, H. et al. (2023). *Let's Verify Step by Step (PRM).* arXiv:2305.20050.
- Coste, T. et al. (2023). *Reward Model Ensembles.* arXiv:2310.02743.
- Lambert, N. et al. (2024). *RewardBench.* arXiv:2403.13787.
