# Evaluation and Benchmarks

> **Last Updated:** 2026-06-18
> **Related Files:** [Data Contamination](../01_data/05_data_contamination_and_evaluation_leakage.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [Hallucination](01_hallucination_causes_and_mitigations.md)
> **Key Papers:** Hendrycks et al. 2020 MMLU ([arXiv:2009.03300](https://arxiv.org/abs/2009.03300)) · Chen et al. 2021 HumanEval ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)) · Rein et al. 2023 GPQA ([arXiv:2311.12022](https://arxiv.org/abs/2311.12022)) · Chiang et al. 2024 Chatbot Arena ([arXiv:2403.04132](https://arxiv.org/abs/2403.04132))

## Overview
Evaluation is how the field measures progress, compares models, and decides what to deploy — yet it is in a state of near-permanent crisis. Benchmarks **saturate** as soon as they become important, suffer **contamination** from web-scale training, and often fail to predict real-world usefulness. The result is a constant churn of new, harder, more contamination-resistant benchmarks and a shift toward **human-preference** and **agentic/real-task** evaluation. For interviews (especially eval-focused roles), knowing the major benchmarks, their failure modes, and modern methodology is essential.

## Core Concepts
**Knowledge & reasoning benchmarks.**
- **MMLU** (57 subjects, multiple choice) — the standard knowledge benchmark; now largely **saturated** (frontier >88%), prompting **MMLU-Pro** (harder, more options) and **MMLU-Redux** (fixing label errors).
- **GPQA** ("Google-proof" graduate science) — hard, expert-written; the "Diamond" set is a current reasoning yardstick.
- **MATH** / **AIME** / **GSM8K** — math reasoning; GSM8K saturated, AIME/competition math now central for reasoning models; **FrontierMath** (extremely hard, held-out) resists saturation.
- **BIG-Bench** / **BIG-Bench Hard** — diverse task suite; **HLE (Humanity's Last Exam)** targets frontier difficulty.

**Code & agentic benchmarks.**
- **HumanEval / MBPP** — function synthesis (pass@k); largely saturated, contamination-prone → **LiveCodeBench** (continuously refreshed, contamination-resistant).
- **SWE-bench (/Verified)** — resolve real GitHub issues in real repos; the flagship *agentic* coding benchmark; **τ-bench**, **WebArena**, **GAIA** for tool/agent tasks.

**Human-preference evaluation.**
- **Chatbot Arena (LMSYS/LMArena):** crowd-sourced blind pairwise battles, aggregated via **Elo/Bradley-Terry** into rankings. Captures real-world preference and is hard to contaminate, but reflects *style/likability* as much as correctness and can be gamed (verbosity, formatting).
- **LLM-as-judge** (MT-Bench, AlpacaEval, Arena-Hard): a strong model scores responses — scalable but biased (position, verbosity, self-preference); length-controlled variants mitigate this.

**Metrics.** pass@k (code), exact match/accuracy (QA/math), Elo (preference), win-rate vs a reference. Statistical rigor (confidence intervals, multiple seeds) is increasingly demanded.

## Key Challenges
- **Saturation.** Important benchmarks hit ceiling fast, losing discriminative power.
- **Contamination.** Test data leaks into training, inflating scores (see [Data Contamination](../01_data/05_data_contamination_and_evaluation_leakage.md)).
- **Construct validity.** Benchmarks often don't measure what we care about (real usefulness, safety, robustness); MCQA rewards test-taking tricks.
- **Gaming & overfitting.** Labs (consciously or not) optimize for leaderboard numbers; Arena susceptible to style/verbosity gaming and selective submission.
- **Reasoning-model artifacts.** Long-CoT models need eval setups that account for variable compute; pass@1 vs pass@k and inference budget matter.
- **Safety/alignment is hard to benchmark.** Refusal, honesty, robustness lack stable ground truth.

## Solutions & Current Best Practices
Use a **portfolio**: knowledge (MMLU-Pro, GPQA), reasoning (AIME, FrontierMath), code/agentic (SWE-bench Verified, LiveCodeBench), preference (Arena), and **private/held-out** internal evals created after the training cutoff. Prefer **contamination-resistant / live / dynamic** benchmarks and **perturbation tests** (GSM-Symbolic) to detect memorization. Report **decontamination methodology**, **confidence intervals**, and **inference settings**. For preference judging, use **length-controlled** metrics and human spot-checks. Treat any single benchmark skeptically; emphasize real-task and held-out performance.

## Lab Perspectives
- **OpenAI/Anthropic/Google:** report broad benchmark suites plus internal held-out evals; increasingly emphasize agentic (SWE-bench) and reasoning (AIME/GPQA/FrontierMath) results; publish dangerous-capability/safety evals.
- **LMArena (LMSYS):** the de-facto public preference leaderboard; influential but debated (style bias, gaming concerns).
- **Meta/DeepSeek/Qwen:** report standard suites; open weights enable independent re-evaluation (a check on cherry-picking).
- **AllenAI/EleutherAI:** open eval harnesses (lm-eval-harness, OLMES) standardizing methodology.

## Latest Developments (2023–2026)
The pivot to **agentic and real-task** evals (SWE-bench Verified as the headline metric), **contamination-resistant/live** benchmarks (LiveCodeBench, LiveBench, FrontierMath, HLE), and **harder knowledge sets** (MMLU-Pro, GPQA) as old ones saturated. Scrutiny of **Arena gaming** and judge biases grew. Reasoning models forced **inference-aware** evaluation. Rising interest in **evaluation science** itself (reliability, statistical rigor, construct validity) and third-party/independent evaluation organizations.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How would you evaluate a new reasoning model?"** Portfolio (AIME/GPQA/FrontierMath + SWE-bench + held-out), contamination checks, pass@k with controlled inference budget, human review.
- **"What's wrong with MMLU/HumanEval today?"** Saturation + contamination + construct validity; move to MMLU-Pro/GPQA, LiveCodeBench, SWE-bench.
- **"Strengths and weaknesses of Chatbot Arena?"** Real preference, hard to contaminate; but style/verbosity bias and gameable.
- **"How do you detect benchmark contamination?"** n-gram/embedding overlap, perturbation tests, order/exchangeability tests, canaries.
- **"Biases of LLM-as-judge?"** Position, verbosity, self-preference; mitigate with length control, swapping, human calibration.

## Open Problems
There is **no stable, contamination-proof, construct-valid** way to measure general capability — benchmarks decay and proxies mislead. Evaluating **safety, honesty, robustness, and agentic competence** reliably is unsolved, as is measuring **superhuman** performance where humans can't judge (links to scalable oversight). "Evaluation science" — making evals rigorous, predictive, and gaming-resistant — is itself an open research program.

## References
- Hendrycks, D. et al. (2020). *Measuring Massive Multitask Language Understanding (MMLU).* arXiv:2009.03300.
- Chen, M. et al. (2021). *Evaluating LLMs Trained on Code (HumanEval).* arXiv:2107.03374.
- Rein, D. et al. (2023). *GPQA.* arXiv:2311.12022.
- Chiang, W.-L. et al. (2024). *Chatbot Arena.* arXiv:2403.04132.
- Jimenez, C. et al. (2023). *SWE-bench.* arXiv:2310.06770.
