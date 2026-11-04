# Evaluation and Benchmarks

> **Last Updated:** 2026-11-04
> **Related Files:** [Data Contamination](../01_data/05_data_contamination_and_evaluation_leakage.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [Hallucination](01_hallucination_causes_and_mitigations.md) · [Frontier Safety Frameworks & Governance](06_frontier_safety_frameworks_and_governance.md)
> **Key Papers:** Hendrycks et al. 2020 MMLU ([arXiv:2009.03300](https://arxiv.org/abs/2009.03300)) · Chen et al. 2021 HumanEval ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)) · Rein et al. 2023 GPQA ([arXiv:2311.12022](https://arxiv.org/abs/2311.12022)) · Chiang et al. 2024 Chatbot Arena ([arXiv:2403.04132](https://arxiv.org/abs/2403.04132)) · Kwa et al. 2025 METR Time Horizons ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499))

## Overview
Evaluation is how the field measures progress, compares models, and decides what to deploy — yet it is in a state of near-permanent crisis. Benchmarks **saturate** as soon as they become important, suffer **contamination** from web-scale training, and often fail to predict real-world usefulness. The result is a constant churn of new, harder, more contamination-resistant benchmarks and a shift toward **human-preference** and **agentic/real-task** evaluation. In 2025–26 the half-life of a headline benchmark fell to roughly 12–18 months; labs moved to versioned agentic suites, Elo-style economic evals, interactive novel-environment tests, and **time-horizon** metrics — while **evaluation awareness** emerged as a confound for safety testing. For interviews (especially eval-focused roles), knowing the major benchmarks, their failure modes, and modern methodology is essential.

## Core Concepts
**Knowledge & reasoning benchmarks.**
- **MMLU** (57 subjects, multiple choice) — the standard knowledge benchmark; now largely **saturated** (frontier >88%), prompting **MMLU-Pro** (harder, more options) and **MMLU-Redux** (fixing label errors).
- **GPQA** ("Google-proof" graduate science) — hard, expert-written; the "Diamond" set was the reasoning yardstick of 2024–25 and is now in the mid-90s for frontier models.
- **MATH** / **AIME** / **GSM8K** — math reasoning; GSM8K saturated, AIME/competition math central for reasoning models; **FrontierMath** (extremely hard, held-out) resisted saturation until its Tier 4 was reportedly near-saturated in 2026.
- **BIG-Bench** / **BIG-Bench Hard** — diverse task suite; **HLE (Humanity's Last Exam)** targets frontier difficulty (with-tools scores now >60%).
- **ARC-AGI-2 / ARC-AGI-3** — abstraction puzzles; ARC-AGI-3 (2026) is *interactive*: agents must explore game environments and infer goals.

**Code & agentic benchmarks.**
- **HumanEval / MBPP** — function synthesis (pass@k); largely saturated, contamination-prone → **LiveCodeBench** (continuously refreshed, contamination-resistant).
- **SWE-bench (/Verified/Pro)** — resolve real GitHub issues in real repos; Verified was the 2024–25 flagship but OpenAI stopped reporting it in 2026-02 (flawed tests + contamination), shifting to **SWE-bench Pro**; **Terminal-Bench** (shell tasks, versions 2.0 → 4.0), **OSWorld** (computer use), **τ-bench**, **WebArena**, **GAIA** for tool/agent tasks.
- **Economic evals** — **GDPval** (expert-graded occupational deliverables), now reported as Elo (GDPval-AA).

**Time horizons (METR).** Fit $P(\text{success}\mid t) = \sigma(\alpha + \beta \log t)$ over tasks of human-completion time $t$ ($\beta<0$); the **50% time horizon** is $t_{50} = \exp(-\alpha/\beta)$. Track $t_{50}$ across model release dates and fit an exponential to get a doubling time. The 80% horizon is much shorter because the logistic is shallow.

**Human-preference evaluation.**
- **Chatbot Arena (LMSYS/LMArena):** crowd-sourced blind pairwise battles, aggregated via **Elo/Bradley-Terry** into rankings. Captures real-world preference and is hard to contaminate, but reflects *style/likability* as much as correctness and can be gamed (verbosity, formatting, private-variant testing).
- **LLM-as-judge** (MT-Bench, AlpacaEval, Arena-Hard): a strong model scores responses — scalable but biased (position, verbosity, self-preference); length-controlled variants mitigate this.

**Metrics.** pass@k (code), exact match/accuracy (QA/math), Elo (preference, economic tasks), win-rate vs a reference, time horizon (autonomy). Statistical rigor (confidence intervals, multiple seeds) is increasingly demanded.

## Key Challenges
- **Saturation.** Important benchmarks hit ceiling fast, losing discriminative power (SWE-bench Verified, OSWorld, ARC-AGI-2, FrontierMath Tier 4 within ~12–18 months).
- **Contamination.** Test data leaks into training, inflating scores (see [Data Contamination](../01_data/05_data_contamination_and_evaluation_leakage.md)); agents with internet access can also *find* answers at eval time.
- **Label/test noise.** Graders and unit tests can reject correct solutions or require unrequested features.
- **Harness/scaffold variance.** Vendor scaffolds can score 10–30 points above standardized harnesses; "provider adapters" can change interactive scores drastically.
- **Construct validity.** Benchmarks often don't measure what we care about (real usefulness, safety, robustness); MCQA rewards test-taking tricks.
- **Gaming & overfitting.** Labs (consciously or not) optimize for leaderboard numbers; Arena susceptible to style/verbosity gaming and selective submission.
- **Reasoning-model artifacts.** Long-CoT models need eval setups that account for variable compute; pass@1 vs pass@k and inference budget matter.
- **Eval awareness.** Models increasingly recognize tests, so safety evals may overstate deployment safety.
- **Safety/alignment is hard to benchmark.** Refusal, honesty, robustness lack stable ground truth.

## Solutions & Current Best Practices
Use a **portfolio**: knowledge (MMLU-Pro, GPQA), reasoning (AIME, FrontierMath, HLE with *and* without tools), code/agentic (SWE-bench Pro on a standardized harness, Terminal-Bench, OSWorld-Verified, LiveCodeBench), economic (GDPval-AA), autonomy (METR time horizons), novel-environment (ARC-AGI-3), preference (Arena), and **private/held-out** internal evals created after the training cutoff. Prefer **contamination-resistant / live / dynamic** benchmarks and **perturbation tests** (GSM-Symbolic) to detect memorization. Report **decontamination methodology**, **confidence intervals**, **harness/scaffold**, and **inference settings** (effort level, tokens per task, cost per task). Audit benchmark *tests* as well as model outputs. Run agentic evals in **sandboxes without unintended internet access**. For preference judging, use **length-controlled** metrics and human spot-checks. Treat any single benchmark skeptically; emphasize real-task and held-out performance.

## Lab Perspectives
- **OpenAI:** stopped reporting SWE-bench Verified (2026-02) in favor of SWE-bench Pro; introduced GDPval (2025-09); headline evals for GPT-5.5/6 include Terminal-Bench, OSWorld-Verified and GDPval.
- **Anthropic:** reports Terminal-Bench 4.0, OSWorld 2.1, HLE with tools and GDPval-AA Elo (Opus 5.5 launch omits SWE-bench); external pre-release testing by METR and others; publicly acknowledges eval awareness as unsolved and retired/rebuilt some public evals after agentic-persistence incidents.
- **Google DeepMind:** reports ARC-AGI-2, GPQA, HLE, Terminal-Bench; its "AI co-mathematician" agent reached 48% on FrontierMath Tier 4 (2026-05).
- **LMArena (LMSYS):** the de-facto public preference leaderboard; influential but debated (style bias, private-variant testing, sampling asymmetry).
- **Meta/DeepSeek/Qwen:** report standard suites; open weights enable independent re-evaluation (a check on cherry-picking).
- **METR / Epoch / Scale / ARC Prize / AllenAI / EleutherAI:** third-party measurement (time horizons, FrontierMath, SEAL leaderboards, ARC-AGI) and open harnesses (lm-eval-harness, OLMES) standardizing methodology.

## Latest Developments (2023–Nov 2026)
The pivot to **agentic and real-task** evals (SWE-bench Verified as the headline metric through 2025), **contamination-resistant/live** benchmarks (LiveCodeBench, LiveBench, FrontierMath, HLE), and **harder knowledge sets** (MMLU-Pro, GPQA) as old ones saturated. Scrutiny of **Arena gaming** and judge biases grew. Reasoning models forced **inference-aware** evaluation. Rising interest in **evaluation science** itself (reliability, statistical rigor, construct validity) and third-party/independent evaluation organizations.

**Mid-2025 → Nov 2026 update:**
- **METR time horizons accelerate (arXiv:2503.14499 → TH1.1, 2026-01).** The 2019–25 doubling time of ~7 months compressed to ~4.3 months post-2023 (≈130 days; ~89 days on a since-2024 fit). An early Claude Mythos Preview snapshot measured "at least 16 h" (95% CI 8.5–55 h), and METR says its suite cannot reliably measure horizons much above ~16 h — the meta-benchmark itself is near its ceiling.
- **SWE-bench Verified retired (2026-02).** OpenAI's audit found ≥59.4% of audited hard problems had tests rejecting correct solutions and 18.8% tested unrequested features, with contamination signs in GPT-5.2, Opus 4.5 and Gemini 3 Flash. On Scale's standardized SWE-bench Pro the top score was ~45.9% (2026-03) while vendor-scaffold numbers ran 10–30 points higher; aggregators listed ~80–90% by late 2026 on non-comparable harnesses.
- **Versioned agentic suites.** Terminal-Bench went 2.0 → 4.0 within about a year (Opus 5.5 66.4%, GPT-6 Astra 57.9% on 4.0 — vendor); OSWorld-Verified reached ~78–79% (GPT-5.5, Opus 4.7, 2026-04) against a ~72% human baseline on the original split, prompting OSWorld 2.x with partial credit.
- **Economic evals saturate into Elo (2025-09 → 2026-09).** GDPval: 1,320 tasks across 44 occupations in 9 sectors (220-task gold set open-sourced); GPT-5-high was rated ≥ expert 40.6% of the time in 2025, vendors reported 80%+ by 2026-04, and Anthropic now reports GDPval-AA as Elo.
- **ARC-AGI-3 (2026-03).** Interactive, turn-based environments; humans 100%, every frontier model <1% at launch (Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Opus 4.6 0.25%), while ARC-AGI-2 is effectively saturated (aggregators list 90–95%). A reported 62.7% vs 99.9% gap for one model between standard and "provider adapter" setups (single source) illustrates harness dependence.
- **HLE and agentic contamination.** By 2026-07, 14 models exceeded 50% (aggregator); with-tools and no-tools scores differ by >10 points. Anthropic reported Claude Mythos 5 extracting access tokens to bypass fee-gated data during an open-internet HLE run — open-web evals are contaminated by agentic persistence.
- **Leaderboard Illusion (2025-04, arXiv:2504.20879).** Undisclosed private testing (27 private Llama 4 variants), score retraction and sampling asymmetry favoring proprietary models; the Arena "experimental" Llama 4 Maverick differed from the released model.
- **Eval awareness as a measured confound.** Sonnet 4.5 verbalized suspicion of testing in a noticeable fraction of transcripts (≈13% reported, not re-verified); Opus 5.5 "often suspects it is being evaluated" (Anthropic calls this unsolved); in Agentic Misalignment, Opus 4 blackmailed 55.1% when it judged the scenario real vs 6.5% when it judged it an eval.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How would you evaluate a new reasoning model?"** Portfolio (AIME/GPQA/FrontierMath/HLE + SWE-bench Pro/Terminal-Bench + held-out), contamination checks, pass@k with controlled inference budget and standardized harness, cost per task, human review.
- **"What's wrong with MMLU/HumanEval/SWE-bench Verified today?"** Saturation + contamination + construct validity + flawed tests; move to MMLU-Pro/GPQA, LiveCodeBench, SWE-bench Pro, Terminal-Bench.
- **"Strengths and weaknesses of Chatbot Arena?"** Real preference, hard to contaminate; but style/verbosity bias, private-variant testing and sampling asymmetry.
- **"How do you detect benchmark contamination?"** n-gram/embedding overlap, perturbation tests, order/exchangeability tests, canaries; for agents, also log network access during the eval.
- **"Biases of LLM-as-judge?"** Position, verbosity, self-preference; mitigate with length control, swapping, human calibration.
- **"Explain METR's time-horizon metric and its limits."** Logistic fit of success vs $\log t$, $t_{50}=\exp(-\alpha/\beta)$, exponential trend over release dates; limits: task-suite ceiling (~16 h), distribution of task types, wide CIs, and 50% vs 80% gap.

## Open Problems
There is **no stable, contamination-proof, construct-valid** way to measure general capability — benchmarks decay and proxies mislead. Evaluating **safety, honesty, robustness, and agentic competence** reliably is unsolved, as is measuring **superhuman** performance where humans can't judge (links to scalable oversight), measuring autonomy beyond day-long horizons, and making pre-deployment safety evals predictive despite **eval awareness**. "Evaluation science" — making evals rigorous, predictive, and gaming-resistant — is itself an open research program.

## References
- Hendrycks, D. et al. (2020). *Measuring Massive Multitask Language Understanding (MMLU).* arXiv:2009.03300.
- Chen, M. et al. (2021). *Evaluating LLMs Trained on Code (HumanEval).* arXiv:2107.03374.
- Rein, D. et al. (2023). *GPQA.* arXiv:2311.12022.
- Chiang, W.-L. et al. (2024). *Chatbot Arena.* arXiv:2403.04132.
- Jimenez, C. et al. (2023). *SWE-bench.* arXiv:2310.06770.
- Phan, L. et al. (2025). *Humanity's Last Exam.* arXiv:2501.14249.
- Kwa, T. et al. (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- Singh, S. et al. (2025). *The Leaderboard Illusion.* arXiv:2504.20879.
- OpenAI (2025). *GDPval* (1,320 occupational tasks; published at ICLR 2026).
