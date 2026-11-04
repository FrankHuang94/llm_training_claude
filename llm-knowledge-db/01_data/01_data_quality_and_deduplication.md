# Data Quality and Deduplication

> **Last Updated:** 2026-11-04
> **Related Files:** [Data Collection & Curation](00_data_collection_and_curation.md) · [Synthetic Data Generation](04_synthetic_data_generation.md) · [Information Theory Basics](../00_foundations/06_information_theory_basics.md)
> **Key Papers:** Lee et al. 2022 "Deduplicating Training Data Makes LMs Better" ([arXiv:2107.06499](https://arxiv.org/abs/2107.06499)) · Wenzek et al. 2019 CCNet ([arXiv:1911.00359](https://arxiv.org/abs/1911.00359)) · Gunasekar et al. 2023 phi-1 "Textbooks Are All You Need" ([arXiv:2306.11644](https://arxiv.org/abs/2306.11644)) · Abbas et al. 2023 SemDeDup ([arXiv:2303.09540](https://arxiv.org/abs/2303.09540))

## Overview
Two levers dominate "effective data" quality: **deduplication** (removing redundant content) and **quality filtering** (keeping high-value content). Both are now understood to be among the highest-ROI interventions in the entire training pipeline — Lee et al. (2022) showed dedup alone improves perplexity, reduces memorization/verbatim regurgitation, and can cut required training compute. The 2024 DCLM/FineWeb-Edu results showed model-based quality filtering is the largest single driver of benchmark gains in open data.

This file covers the algorithms (MinHash LSH, suffix arrays, semantic dedup) and the quality-filtering toolkit (perplexity, classifiers), plus the "quality vs quantity" and "textbook data" debates.

## Core Concepts
**Exact deduplication.** Remove identical documents (hashing) or identical substrings. **Suffix-array** methods (Lee et al.) find and remove exact substring duplicates above a length threshold (e.g., 50 tokens) across the entire corpus in near-linear time.

**Fuzzy / near-duplicate dedup via MinHash + LSH.** Documents are shingled into $k$-grams; **MinHash** estimates Jaccard similarity $J(A,B)=\frac{|A\cap B|}{|A\cup B|}$ using the property that for a random permutation $\pi$, $\Pr[\min\pi(A)=\min\pi(B)]=J(A,B)$. Using $m$ hash functions yields an $m$-dimensional signature whose agreement rate estimates $J$. **Locality-Sensitive Hashing (LSH)** then bands the signature into $b$ bands of $r$ rows; documents colliding in any band become candidate pairs. The probability two docs with similarity $s$ become candidates is $1-(1-s^r)^b$ — an S-curve whose threshold is tuned via $b,r$. This makes all-pairs near-dedup feasible at trillion-token scale.

**Semantic deduplication (SemDeDup).** Embed documents, cluster, and remove near-duplicates in *embedding space* — catches paraphrases that surface-form methods miss; removes "semantic redundancy."

**Perplexity filtering (CCNet).** Score documents with a reference LM (e.g., a KenLM n-gram model trained on Wikipedia); keep documents in a target perplexity band — too-high PPL = gibberish/noise, too-low = boilerplate/repetition.

**Classifier-based filtering.** Train a lightweight classifier (fastText logistic regression on n-grams) to distinguish "high quality" (e.g., Wikipedia/books/curated) from random web. DCLM and FineWeb-Edu use this; FineWeb-Edu's classifier scores "educational value" via LLM-generated labels.

## Key Challenges
- **Over- vs under-dedup.** Aggressive dedup removes legitimately repeated high-value content (definitions, code idioms); too little leaves memorization and wasted compute.
- **Quality classifier bias.** Classifiers inherit the biases of their "good" reference set, can penalize dialects/low-resource languages and favor a narrow style.
- **Quality vs quantity tradeoff.** Heavy filtering shrinks the corpus, conflicting with Chinchilla token demands; the optimum depends on compute budget.
- **Repetition effects.** Some repetition helps (data-constrained scaling), but verbatim duplicates cause memorization and privacy leakage.

## Solutions & Current Best Practices
The standard stack: **exact + MinHash-LSH fuzzy dedup** (global, cross-snapshot), followed by **model-based quality classification** (DCLM/FineWeb-Edu), optional **semantic dedup**, plus PII and toxicity filters. Repetition is then *controlled* (Muennighoff et al.: up to ~4 epochs of high-quality data is near-free) rather than eliminated. A final **midtraining/annealing** stage upweights the very highest-quality subset (math, code, textbooks). Increasingly, low-scoring but non-junk documents are **rephrased** rather than discarded (FinePhrase, SynPro), and dedup counts are reused as an upsampling signal rather than only a deletion criterion (FineWeb2 "rehydration").

## Lab Perspectives
- **Meta** (Llama 3) applied aggressive dedup + quality classifiers and reported large gains.
- **Microsoft** (phi-series) is the standard-bearer for the **"textbook-quality data"** thesis: small models trained on curated/synthetic high-quality data punch far above their weight (phi-1, phi-1.5, phi-2, phi-3).
- **HuggingFace/AllenAI** open the methodology (FineWeb-Edu, Dolma, FineWeb2, Dolma 3); NVIDIA released Nemotron 3 pretraining data and recipes.
- **OpenAI/Anthropic/Google** keep specifics closed but clearly rely on heavy classifier-based curation and licensed high-quality corpora. Anthropic and OpenAI now also publish on *environment* quality for RL: OpenAI retired SWE-bench Verified over flawed tests, and Anthropic showed that hackable coding environments induce broad misalignment.

## Latest Developments (2023–Nov 2026)
The **"quality > quantity"** view gained strong evidence: phi-series and DCLM showed curated/filtered data dramatically improves quality-per-token. **LLM-as-annotator** quality scoring (FineWeb-Edu) replaced hand-tuned heuristics. Debate persists over whether the phi approach **overfits benchmarks** (its synthetic data may resemble eval distributions) versus genuinely improving capability — a live contamination concern.

**Mid-2025 → Nov 2026 update:**
- **Dedup-informed upsampling (2025-06; arXiv:2506.20920).** FineWeb2 adapts dedup and filtering thresholds per language across >1,000 languages and then "rehydrates": documents are upsampled according to how many duplicates they had, on the premise that duplicate-cluster size carries quality signal. Global dedup + selective re-weighting replaces "dedup then forget".
- **Rephrasing as a quality transform (2026-04; arXiv:2604.13977).** FinePhrase (>1T generated tokens) finds that rewriting web text into structured targets — tables, math problems, FAQs, tutorials — beats both curated web data and earlier synthetic methods, and that generators above ~1B parameters add nothing; 486B tokens released. Quality improvement is cheap enough to apply to the bulk of the corpus.
- **Learned rephrasers with faithfulness rewards (2026-05; arXiv:2605.17849).** SynPro trains rephrasers with RL rewarded for quality, faithfulness to the source and data influence, reporting 3.7–5.2× the effective tokens of plain repetition in a data-constrained regime — reframing "quality vs quantity" as "how much quality can be added per source token".
- **Quality of RL data and environments (2025-11 → 2026-02).** Hackable coding environments (e.g., `sys.exit(0)` passing tests) caused reward hacking that generalized to sabotage and alignment faking (arXiv:2511.18397); OpenAI's audit of SWE-bench Verified found ≥59.4% of audited hard problems had flawed tests. Test-suite quality is now a data-quality problem with safety consequences.
- **Timing interacts with quality (2025-10; arXiv:2510.14865).** When high-quality specialized data enters training interacts with its mixture weight: early introduction tolerates high weight, late introduction needs low weight; gains are largest for math and code (see [Data Mixing](03_data_mixing_and_curriculum.md)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain MinHash + LSH and how you'd tune the threshold."** Jaccard estimation, banding S-curve $1-(1-s^r)^b$, $b,r$ tradeoff.
- **"Why does deduplication improve models?"** Less memorization, more effective tokens, less wasted compute, better eval validity (Lee et al.).
- **"Quality vs quantity — how do you decide filter strength?"** Depends on compute budget vs data availability; cite phi vs Chinchilla.
- **"Risks of model-based quality filters?"** Style/dialect bias, benchmark contamination, distribution narrowing.
- **"Filter or rephrase?"** Discarding low-score documents loses tokens; rephrasing (small generator, structured formats, faithfulness checks) keeps the content and raises quality — but risks homogenized style and hallucinated facts, so verify faithfulness.

## Open Problems
There is no agreed metric for "data quality" — classifiers are proxies. Whether the phi "textbook" gains generalize beyond benchmarks (vs reflecting contamination) is contested. The optimal dedup aggressiveness, the safe limit of repetition, and how to filter without amplifying bias are unsolved.

## References
- Lee, K. et al. (2022). *Deduplicating Training Data Makes LMs Better.* arXiv:2107.06499.
- Wenzek, G. et al. (2019). *CCNet.* arXiv:1911.00359.
- Gunasekar, S. et al. (2023). *Textbooks Are All You Need (phi-1).* arXiv:2306.11644.
- Abbas, A. et al. (2023). *SemDeDup.* arXiv:2303.09540.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained LMs.* arXiv:2305.16264.
- Penedo, G. et al. (2025). *FineWeb2: One Pipeline to Scale Them All.* arXiv:2506.20920.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
