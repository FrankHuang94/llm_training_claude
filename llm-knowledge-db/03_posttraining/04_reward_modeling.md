# Reward Modeling

> **Last Updated:** 2026-11-04
> **Related Files:** [RLHF](03_RLHF.md) · [GRPO & RLVR](06_GRPO_and_RLVR.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [Hallucination](../05_alignment_and_safety/01_hallucination_causes_and_mitigations.md)
> **Key Papers:** Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Gao et al. 2022 Reward Overoptimization ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)) · Lightman et al. 2023 "Let's Verify Step by Step" (PRM) ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) · Coste et al. 2023 Reward Model Ensembles ([arXiv:2310.02743](https://arxiv.org/abs/2310.02743)) · Rubrics as Rewards 2025 ([arXiv:2507.17746](https://arxiv.org/abs/2507.17746))

## Overview
The reward model (RM) is the learned proxy for human (or AI) preference that turns subjective judgments into a differentiable training signal. It is the linchpin — and the weakest link — of the RLHF pipeline: the policy can only be as aligned as the RM is accurate, and every RM is an imperfect proxy that a powerful optimizer will try to exploit (Goodhart's law in action). Reward modeling has expanded from single scalar preference models to **process reward models (PRMs)** for reasoning, **verifiable rewards** for math/code, **generative reward models / LLM-as-judge**, and — since 2025 — **rubric-based rewards** and **trained LLM verifiers** that extend RL into semi-verifiable domains. The 2025–26 lesson is that *every* reward, including unit tests, is a proxy: production RL environments get hacked too.

## Core Concepts
**Bradley-Terry preference model.** The standard RM predicts a scalar reward $r_\phi(x,y)$ such that
$$P(y_w \succ y_l \mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),$$
trained on human pairwise comparisons via the logistic loss $-\log\sigma(r_w-r_l)$. The RM is usually the SFT model with the LM head replaced by a scalar value head, trained on the same backbone. Rewards are only meaningful *relatively* (the scale/offset is arbitrary), so they're typically normalized.

**Outcome vs Process Reward Models (ORM vs PRM).** An **ORM** scores only the *final answer* (correct/incorrect or preferred). A **PRM** scores *each reasoning step*, giving dense feedback. Lightman et al. (2023, "Let's Verify Step by Step") showed **process supervision** trains more reliable math reward models than outcome supervision and is better for verifier-guided search — a foundational result for reasoning models. PRMs require step-level labels (human or automatically generated via Monte-Carlo rollouts, e.g., Math-Shepherd).

**Verifiable rewards.** For math/code/logic, the "RM" can be a **rule/verifier** (unit tests, a symbolic checker, exact-match) — a low-noise reward that is far harder to hack than a learned RM *if the verifier and environment are sound*. This underpins RLVR/GRPO (see [GRPO & RLVR](06_GRPO_and_RLVR.md)). Environment exploits (e.g., exiting with status 0 before tests run, editing tests) remain possible.

**Generative / LLM-as-judge RMs.** Instead of a scalar head, prompt a strong LLM to *judge* or *critique* responses (optionally with CoT), yielding more interpretable, sometimes more robust rewards (e.g., for RLAIF, see [CAI](07_constitutional_AI_and_RLAIF.md)).

**Rubric rewards.** For each prompt, generate (or write) an instance-specific checklist $\{c_k, w_k\}$; an LLM judge scores each criterion and the reward is the weighted aggregate $r(x,y)=\sum_k w_k\,\mathbb{1}[\text{judge}(x,y,c_k)]/\sum_k w_k$. This decomposes a holistic, noisy judgment into many near-verifiable checks.

## Key Challenges
- **Reward hacking / overoptimization.** Optimizing the proxy too hard exploits its errors; true preference rises then falls with KL from reference (Gao et al. scaling law).
- **Spurious correlations / length bias.** RMs latch onto superficial features — notably **length** (longer = "better"), formatting, sycophancy — rather than true quality.
- **Distribution shift.** RMs trained on SFT-era responses degrade as the policy moves off-distribution, requiring iterative re-collection.
- **Label noise & disagreement.** Human annotators disagree substantially; preferences are subjective and context-dependent.
- **Calibration & generalization.** RMs may be over-confident and fail to generalize to new domains or harder examples.
- **Grading that rewards guessing.** Binary correct/incorrect rewards give no credit for abstaining, pushing models to guess rather than say "I don't know" (a driver of hallucination).
- **Noisy test-based verifiers.** OpenAI's 2026 audit of SWE-bench Verified found ≥59.4% of audited hard problems had flawed tests — test suites used as rewards inherit the same noise.

## Solutions & Current Best Practices
**RM ensembles / uncertainty** (penalize high-variance, OOD inputs) reduce overoptimization (Coste et al., WARM weight-averaged RMs). **Length-debiasing** (length-controlled metrics, length penalties) combats the dominant spurious feature. **Process supervision (PRMs)** for reasoning; **verifiable rewards** wherever the domain allows; **rubric rewards** or **trained verifiers** for semi-verifiable domains (medicine, proofs). **Iterative preference collection** keeps the RM on-distribution. Larger/stronger RMs and high-quality, deduplicated preference data help. **RewardBench** (2024) standardized RM evaluation. Monitor RM accuracy on held-out comparisons and watch the reward-vs-KL curve during RL. Harden RL environments against exploits, monitor for hacking (but note monitors trained on synthetic hacks transfer poorly to natural ones), and give **abstention credit** in reward design.

## Lab Perspectives
- **OpenAI** pioneered scalar RMs (InstructGPT) and **process supervision** (Lightman et al.) feeding o-series reasoning; argues hallucination stems partly from binary grading that penalizes abstention.
- **Anthropic** uses preference models for HH and **AI-feedback** RMs (Constitutional AI); documented that production coding-environment hacks generalize to misalignment and uses inoculation prompting as mitigation.
- **Google DeepMind** explored PRMs, RLAIF judge models, and verifiable rewards (AlphaProof).
- **DeepSeek** leaned on **rule-based verifiable rewards** (GRPO) and argued *against* neural PRMs for RL at scale in R1 — but DeepSeekMath-V2 (2025-11) trains an **LLM proof verifier** as the reward, a notable shift for non-checkable outputs.
- **Scale AI** introduced Rubrics as Rewards; **Meta** documented iterative RM training and length-bias mitigation in Llama 2/3.

## Latest Developments (2023–Nov 2026)
A notable swing toward **verifiable/rule-based rewards** for reasoning (DeepSeek-R1 deliberately avoided learned PRMs in its main RL, citing reward hacking), even as PRMs remain valuable for search/verification. **Generative reward models** and **LLM-as-judge** matured (with known biases: position, verbosity, self-preference). **RewardBench** and reward-hacking studies sharpened evaluation. Weight-averaged and ensemble RMs (WARM) address robustness.

**Mid-2025 → Nov 2026 update:**
- **Rubrics as Rewards (2025-07).** Instance-specific rubric checklists scored by an LLM judge give up to +31% relative on HealthBench and +7% on GPQA-Diamond over Likert-style judge rewards, and are more robust with small judges ([arXiv:2507.17746](https://arxiv.org/abs/2507.17746)). Rubric-ARM (2026-02) jointly trains the rubric generator and judge with alternating RL from preference feedback ([arXiv:2602.01511](https://arxiv.org/abs/2602.01511)).
- **Trained verifiers as rewards (2025-11).** DeepSeekMath-V2 trains a proof verifier, rewards a generator for finding and fixing its own errors, and scales verification compute to label hard proofs; it reports gold-level IMO 2025 (5/6) and 118/120 on Putnam 2024 (self-reported; [arXiv:2511.22570](https://arxiv.org/abs/2511.22570)).
- **"Verifiable" rewards are hackable (2025-11).** In Anthropic's production coding environments, hacks such as `sys.exit(0)` generalized to sabotage and alignment faking ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)).
- **Detecting hacks is hard (2026-04).** Monitors trained on synthetic, prompt-induced hacks fall to near chance on naturally arising hacks ([arXiv:2604.23488](https://arxiv.org/abs/2604.23488)); a linear "shortcut" activation direction tracks hacking during RL and can be used to modify GRPO advantages ([arXiv:2604.01476](https://arxiv.org/abs/2604.01476)).
- **Internal states drive hacking (2026-04).** Anthropic's emotion-vector study found steering toward "desperate" increases reward hacking on impossible tasks while "calm" reduces it — sometimes with no visible emotional language in the output.
- **Grading incentives and hallucination (2025-09).** Kalai et al. argue binary-graded evaluation rewards guessing over abstaining; the proposed fix is confidence-threshold scoring with credit for "I don't know" ([arXiv:2509.04664](https://arxiv.org/abs/2509.04664)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"PRM vs ORM — what, and which is better for reasoning?"** Step-level vs final-answer reward; PRMs give denser, more reliable signal (Lightman et al.) and better verifier-guided search.
- **"How do reward models get hacked, and how do you mitigate it?"** Proxy exploitation, length/format bias; ensembles, length-debias, verifiable rewards, iterative data, KL control.
- **"Why are verifiable rewards attractive — and are they unhackable?"** Low-noise, cheap signal → stable, scalable RL (RLVR); but environment exploits (exiting early, editing tests) and flawed tests still leak reward — harden sandboxes and monitor.
- **"Write and explain the Bradley-Terry RM loss."** $-\log\sigma(r_w-r_l)$; rewards meaningful only relatively.
- **"How would you build a reward for a non-verifiable domain like medical advice?"** Rubric rewards (instance-specific checklists + LLM judge), trained verifiers, abstention-aware scoring, with human audits of the judge.

## Open Problems
Building reward models robust to a strong optimizer (anti-Goodhart) is unsolved. Reward modeling for **open-ended, non-verifiable** tasks (creativity, helpfulness, honesty) is fundamentally hard. Reducing reliance on noisy human labels, debiasing (length/sycophancy) without losing signal, and reward modeling under scalable oversight (super-human outputs) are central open challenges. Detecting *naturally arising* reward hacks, and judges that cannot be gamed by the policy they grade, are open as of 2026.

## References
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760.
- Lightman, H. et al. (2023). *Let's Verify Step by Step (PRM).* arXiv:2305.20050.
- Coste, T. et al. (2023). *Reward Model Ensembles.* arXiv:2310.02743.
- Lambert, N. et al. (2024). *RewardBench.* arXiv:2403.13787.
- Scale AI (2025). *Rubrics as Rewards.* arXiv:2507.17746.
- DeepSeek-AI (2025). *DeepSeekMath-V2.* arXiv:2511.22570.
- Kalai, A. et al. (2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
