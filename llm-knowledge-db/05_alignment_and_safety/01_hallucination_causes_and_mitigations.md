# Hallucination: Causes and Mitigations

> **Last Updated:** 2026-06-18
> **Related Files:** [Context Window Extension](../06_efficiency/06_context_window_extension.md) · [Evaluation & Benchmarks](05_evaluation_and_benchmarks.md) · [RLHF](../03_posttraining/03_RLHF.md)
> **Key Papers:** Ji et al. 2022 Survey of Hallucination ([arXiv:2202.03629](https://arxiv.org/abs/2202.03629)) · Lewis et al. 2020 RAG ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) · Kadavath et al. 2022 "LMs (Mostly) Know What They Know" ([arXiv:2207.05221](https://arxiv.org/abs/2207.05221)) · Kalai & Vempala 2024 "Why LMs Hallucinate" ([arXiv:2509.04664](https://arxiv.org/abs/2509.04664))

## Overview
Hallucination — confident generation of false or unsupported content — is the most prominent reliability failure of LLMs and a primary blocker to deployment in high-stakes domains (medicine, law, finance). It is not a bug to be patched but a *structural* consequence of how LLMs are trained and decode: they are probabilistic next-token predictors optimized for plausibility, not truth. Understanding the taxonomy, root causes, and the mitigation toolkit (RAG, calibration, factuality tuning) is essential, and increasingly there is theoretical work explaining why hallucination is partly *inevitable* and even *incentivized* by current training/eval regimes.

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
- **Eval incentives (Kalai & Vempala 2024 / OpenAI 2025):** benchmarks that score binary correctness with **no penalty for wrong vs. abstaining** mathematically incentivize guessing over admitting uncertainty — hallucination is partly a *training/eval-incentive* artifact, not just a knowledge gap. There's also a reduction to a binary "is-this-valid" classification lower bound: even optimal models hallucinate at a rate tied to the fraction of facts seen once.

**Calibration.** Models often *do* encode their uncertainty: Kadavath et al. showed LLMs can predict whether they know an answer (P(IK)) reasonably well, suggesting hallucination is partly a *failure to express* known uncertainty rather than pure ignorance.

## Key Challenges
- **No ground-truth signal at train time.** The objective doesn't separate true from plausible-false.
- **Long-tail and post-cutoff facts.** Rare and recent information is unknowable from weights alone.
- **Confidence ≠ correctness.** Models are frequently confidently wrong; calibration degrades after RLHF.
- **Detection is hard.** Verifying factuality at scale, especially for open-ended generation, is itself unsolved.
- **Plausible fabrications.** Invented citations/numbers look authoritative, evading casual review.

## Solutions & Current Best Practices
- **Retrieval-Augmented Generation (RAG):** ground answers in retrieved documents; cite sources; the single most effective mitigation for knowledge-grounded tasks (see [Context Extension](../06_efficiency/06_context_window_extension.md)). Reduces extrinsic hallucination but can introduce intrinsic errors if the model misreads sources.
- **Tool use / verification:** calculators, code execution, search, and verifiers replace guessing with computation/lookup (see [Tool Use](../04_reasoning_and_agents/03_tool_use_and_function_calling.md)).
- **Abstention & calibrated uncertainty:** train/prompt the model to say "I don't know," express confidence, and **reward appropriate abstention** (fixing the eval-incentive problem). Confidence estimation via P(IK), logit entropy, or verbalized probabilities.
- **Self-consistency / self-verification:** sample multiple answers and check agreement (SelfCheckGPT); use a second pass to critique.
- **Factuality fine-tuning / RLHF for honesty:** preference data that rewards accurate, hedged, source-grounded answers (FactTune, honesty-tuning).
- **Decoding interventions:** DoLa (contrasting layers), inference-time intervention on "truthfulness" directions (interpretability-informed).

## Lab Perspectives
- **OpenAI:** "Why LMs Hallucinate" (2025) framed hallucination as an eval-incentive problem and advocates rewarding abstention; deploys retrieval/tools.
- **Anthropic:** emphasizes **honesty/calibration** (HHH), training Claude to express uncertainty and refuse to fabricate; interpretability for truthfulness.
- **Google DeepMind:** search-grounding (Gemini), FACTS benchmark, SAFE (search-augmented factuality evaluation).
- **Meta:** RAG originated at FAIR; Llama uses retrieval/tool integrations and factuality data.

## Latest Developments (2023–2026)
A conceptual shift toward **hallucination as (partly) an incentive problem**: reward abstention, redesign evals to penalize confident errors (Kalai & Vempala / OpenAI 2025). Stronger **RAG + agentic verification** (Deep Research-style models that search and cite). **Self-verification and uncertainty quantification** matured. Reasoning models reduced some reasoning-error hallucinations but can still fabricate within long chains. Persistent debate: is hallucination fully solvable, or an irreducible feature of probabilistic LMs?

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why do LLMs hallucinate?"** Plausibility-not-truth objective, knowledge gaps/long tail, miscalibration, and eval incentives that reward guessing over abstaining.
- **"Intrinsic vs extrinsic hallucination?"** Contradicts source vs unverifiable-against-source.
- **"How does RAG help and what are its limits?"** Grounds in retrieved evidence (faithfulness); fails if retrieval is poor or the model misreads sources.
- **"How would you reduce hallucination in a deployed assistant?"** RAG + tools + calibrated abstention + self-verification + eval that rewards 'I don't know.'

## Open Problems
Whether hallucination is **fully eliminable** or fundamentally irreducible for probabilistic LMs is contested (theory suggests a nonzero floor). Reliable **calibration** and **abstention** under distribution shift, scalable **factuality detection** for open-ended text, and grounding for claims with no retrievable source remain unsolved.

## References
- Ji, Z. et al. (2022). *Survey of Hallucination in NLG.* arXiv:2202.03629.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation.* arXiv:2005.11401.
- Kadavath, S. et al. (2022). *LMs (Mostly) Know What They Know.* arXiv:2207.05221.
- Kalai, A., Vempala, S. et al. (2024/2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
- Manakul, P. et al. (2023). *SelfCheckGPT.* arXiv:2303.08896.
