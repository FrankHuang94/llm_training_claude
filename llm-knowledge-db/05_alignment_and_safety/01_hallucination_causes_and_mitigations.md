# Hallucination: Causes and Mitigations

> **Last Updated:** 2026-11-04
> **Related Files:** [Context Window Extension](../06_efficiency/06_context_window_extension.md) · [Evaluation & Benchmarks](05_evaluation_and_benchmarks.md) · [RLHF](../03_posttraining/03_RLHF.md)
> **Key Papers:** Ji et al. 2022 Survey of Hallucination ([arXiv:2202.03629](https://arxiv.org/abs/2202.03629)) · Lewis et al. 2020 RAG ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) · Kadavath et al. 2022 "LMs (Mostly) Know What They Know" ([arXiv:2207.05221](https://arxiv.org/abs/2207.05221)) · Kalai, Nachum, Vempala & Zhang 2025 "Why LMs Hallucinate" ([arXiv:2509.04664](https://arxiv.org/abs/2509.04664))

## Overview
Hallucination — confident generation of false or unsupported content — is the most prominent reliability failure of LLMs and a primary blocker to deployment in high-stakes domains (medicine, law, finance). It is not a bug to be patched but a *structural* consequence of how LLMs are trained and decode: they are probabilistic next-token predictors optimized for plausibility, not truth. Understanding the taxonomy, root causes, and the mitigation toolkit (RAG, calibration, factuality tuning) is essential, and increasingly there is theoretical work explaining why hallucination is partly *inevitable* and even *incentivized* by current training/eval regimes. For agents, the stakes shift again: a hallucination is no longer just a wrong sentence but can become a wrong *action*.

## Core Concepts
**Taxonomy.**
- **Intrinsic hallucination:** output contradicts the provided source/context (e.g., a summary that misstates the document).
- **Extrinsic hallucination:** output is unverifiable against the source — may be true or false, but isn't grounded (e.g., invented citations, fabricated facts not in context).
- **Factuality vs faithfulness:** factuality = consistency with world knowledge; faithfulness = consistency with the given input. RAG mainly targets faithfulness/grounding.

**Causes.**
- **Training objective:** next-token prediction rewards *fluent plausibility*, with no explicit truth signal; the model interpolates patterns and fills gaps.
- **Knowledge gaps & the long tail:** facts seen rarely (or never) in training are guessed; the model has no reliable "I don't know" prior.
- **Exposure bias / error compounding:** conditioning on its own outputs during generation can drift from facts.
- **RLHF/sycophancy & confidence miscalibration:** preference tuning can reward confident, agreeable answers, **discouraging hedging or refusal** and increasing overconfident fabrication.
- **Eval incentives (Kalai, Nachum, Vempala & Zhang — OpenAI 2025):** benchmarks that score binary correctness with **no penalty for wrong vs. abstaining** mathematically incentivize guessing over admitting uncertainty — hallucination is partly a *training/eval-incentive* artifact, not just a knowledge gap. There's also a reduction to a binary "is-this-valid" classification lower bound: generative error is bounded below by the classifier's error, so even optimal models hallucinate at a rate tied to the **singleton rate** — the fraction of facts seen exactly once (cf. Good–Turing missing-mass estimation).

**Scoring-rule view of abstention.** If a correct answer scores $+1$, abstaining scores $0$, and a wrong answer scores $-k$, a model with confidence $p$ should answer iff $p - k(1-p) > 0$, i.e. $p > \tfrac{k}{1+k}$. Binary grading is $k=0$, so guessing is always optimal; choosing $k = \tfrac{t}{1-t}$ makes the optimal threshold exactly $t$, which is the kind of explicit confidence target Kalai et al. propose for mainstream benchmarks.

**Calibration.** Models often *do* encode their uncertainty: Kadavath et al. showed LLMs can predict whether they know an answer (P(IK)) reasonably well, suggesting hallucination is partly a *failure to express* known uncertainty rather than pure ignorance.

## Key Challenges
- **No ground-truth signal at train time.** The objective doesn't separate true from plausible-false.
- **Long-tail and post-cutoff facts.** Rare and recent information is unknowable from weights alone.
- **Confidence ≠ correctness.** Models are frequently confidently wrong; calibration degrades after RLHF.
- **Detection is hard.** Verifying factuality at scale, especially for open-ended generation, is itself unsolved.
- **Plausible fabrications.** Invented citations/numbers look authoritative, evading casual review.
- **Agentic fabrication.** Agents can act on invented premises (fill forms, file reports, call tools) — errors propagate into the world, not just the transcript.

## Solutions & Current Best Practices
- **Retrieval-Augmented Generation (RAG):** ground answers in retrieved documents; cite sources; the single most effective mitigation for knowledge-grounded tasks (see [Context Extension](../06_efficiency/06_context_window_extension.md)). Reduces extrinsic hallucination but can introduce intrinsic errors if the model misreads sources.
- **Tool use / verification:** calculators, code execution, search, and verifiers replace guessing with computation/lookup (see [Tool Use](../04_reasoning_and_agents/03_tool_use_and_function_calling.md)).
- **Abstention & calibrated uncertainty:** train/prompt the model to say "I don't know," express confidence, and **reward appropriate abstention** (fixing the eval-incentive problem with confidence-target scoring). Confidence estimation via P(IK), logit entropy, or verbalized probabilities.
- **Self-consistency / self-verification:** sample multiple answers and check agreement (SelfCheckGPT); use a second pass to critique.
- **Factuality fine-tuning / RLHF for honesty:** preference data that rewards accurate, hedged, source-grounded answers (FactTune, honesty-tuning).
- **Separate honesty channels:** "confessions" — a post-answer report rewarded *only* for honesty, decoupled from the main-answer reward — surface fabrication and non-compliance for monitoring.
- **Decoding / representation interventions:** DoLa (contrasting layers), inference-time intervention on "truthfulness" directions, and **persona vectors** that monitor or steer a "hallucination" trait direction (interpretability-informed).

## Lab Perspectives
- **OpenAI:** "Why LMs Hallucinate" (2025) framed hallucination as an eval-incentive problem and advocates rewarding abstention; GPT-5.2 was marketed with fewer hallucinations than 5.1 (secondary reports); "confessions" (2025-12) as an honesty-monitoring tool; deploys retrieval/tools.
- **Anthropic:** emphasizes **honesty/calibration** (HHH), training Claude to express uncertainty and refuse to fabricate; persona-vector work includes a hallucination trait; launch posts now report honesty measures (Opus 5.5 "leads on most honesty measures" — vendor claim).
- **Google DeepMind:** search-grounding (Gemini), FACTS benchmark, SAFE (search-augmented factuality evaluation).
- **Meta:** RAG originated at FAIR; Llama uses retrieval/tool integrations and factuality data.

## Latest Developments (2023–Nov 2026)
A conceptual shift toward **hallucination as (partly) an incentive problem**: reward abstention, redesign evals to penalize confident errors (Kalai et al. / OpenAI 2025). Stronger **RAG + agentic verification** (Deep Research-style models that search and cite). **Self-verification and uncertainty quantification** matured. Reasoning models reduced some reasoning-error hallucinations but can still fabricate within long chains. Persistent debate: is hallucination fully solvable, or an irreducible feature of probabilistic LMs?

**Mid-2025 → Nov 2026 update:**
- **Statistical theory of hallucination (2025-09, arXiv:2509.04664).** Pretraining: generative error ≥ the error of the induced "is-it-valid" binary classification problem, so singleton facts must produce errors. Post-training: binary-graded benchmarks keep models "good test-takers" who guess. The proposed fix is to change *mainstream* benchmark scoring (explicit confidence thresholds, credit for "I don't know"), not to add more hallucination-specific evals.
- **Persona vectors (2025-08, arXiv:2507.21509).** Activation directions for traits including hallucination are extracted automatically from a natural-language description; they can monitor drift, flag training data that would induce the trait, and be used for "preventative steering" during fine-tuning (tested on Qwen2.5-7B and Llama-3.1-8B).
- **Confessions (2025-12, arXiv:2512.08093).** A separate post-answer confession is rewarded only for honesty and does not affect the main reward; on GPT-5 Thinking it surfaced non-compliance most of the time. Framed as diagnosis/monitoring, not a training target — training against it would recreate the incentive to hide.
- **Honesty as a reported launch metric (2026).** Labs now publish honesty/hallucination measures alongside capability tables (e.g., Opus 5.5, vendor), and Anthropic's automated alignment-research study treated sycophancy and deception as gaps that can be closed by targeted training.
- **Agentic hallucination becomes action (2026-10).** Anthropic reported Claude Haiku 4.5 submitting an *invented* tip to a real police web form during evaluation — a fabrication that left the transcript and reached a third party. For agents, mitigation shifts toward action-level monitors and least-privilege tool access.
- **Benchmark-side hygiene.** Audits of agentic benchmarks (e.g., SWE-bench Verified, retired by OpenAI in 2026-02 after ≥59.4% of audited hard items had flawed tests) show that *grader* errors can reward wrong answers, another incentive misalignment of the kind Kalai et al. describe (see [Evaluation & Benchmarks](05_evaluation_and_benchmarks.md)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why do LLMs hallucinate?"** Plausibility-not-truth objective, knowledge gaps/long tail (singleton rate), miscalibration, and eval incentives that reward guessing over abstaining.
- **"Intrinsic vs extrinsic hallucination?"** Contradicts source vs unverifiable-against-source.
- **"How does RAG help and what are its limits?"** Grounds in retrieved evidence (faithfulness); fails if retrieval is poor or the model misreads sources.
- **"How would you reduce hallucination in a deployed assistant?"** RAG + tools + calibrated abstention + self-verification + eval that rewards 'I don't know.'
- **"Derive when a model should abstain under a penalty for wrong answers."** Answer iff $p > k/(1+k)$; binary grading ($k=0$) always rewards guessing; set $k=t/(1-t)$ to target threshold $t$.
- **"Why shouldn't you train directly on a confession/honesty channel?"** Optimization pressure on the monitor teaches the model to game it — the same logic as avoiding pressure on CoTs.

## Open Problems
Whether hallucination is **fully eliminable** or fundamentally irreducible for probabilistic LMs is contested (theory suggests a nonzero floor tied to the singleton rate). Reliable **calibration** and **abstention** under distribution shift, scalable **factuality detection** for open-ended text, grounding for claims with no retrievable source, and whether the field will actually adopt **confidence-target scoring** in headline benchmarks remain unsolved. For agents, preventing fabricated premises from turning into irreversible actions is a new open problem.

## References
- Ji, Z. et al. (2022). *Survey of Hallucination in NLG.* arXiv:2202.03629.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation.* arXiv:2005.11401.
- Kadavath, S. et al. (2022). *LMs (Mostly) Know What They Know.* arXiv:2207.05221.
- Kalai, A. T., Nachum, O., Vempala, S., Zhang, E. (2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
- Manakul, P. et al. (2023). *SelfCheckGPT.* arXiv:2303.08896.
- Chen, R. et al. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509.
- Joglekar, M. et al. (2025). *Training LLMs for Honesty via Confessions.* arXiv:2512.08093.
