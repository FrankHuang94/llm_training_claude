# Data Quality and Deduplication

> **Last Updated:** 2026-06-18
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
The standard stack: **exact + MinHash-LSH fuzzy dedup** (global, cross-snapshot), followed by **model-based quality classification** (DCLM/FineWeb-Edu), optional **semantic dedup**, plus PII and toxicity filters. Repetition is then *controlled* (Muennighoff et al.: up to ~4 epochs of high-quality data is near-free) rather than eliminated. A final **midtraining/annealing** stage upweights the very highest-quality subset (math, code, textbooks).

## Lab Perspectives
- **Meta** (Llama 3) applied aggressive dedup + quality classifiers and reported large gains.
- **Microsoft** (phi-series) is the standard-bearer for the **"textbook-quality data"** thesis: small models trained on curated/synthetic high-quality data punch far above their weight (phi-1, phi-1.5, phi-2, phi-3).
- **HuggingFace/AllenAI** open the methodology (FineWeb-Edu, Dolma).
- **OpenAI/Anthropic/Google** keep specifics closed but clearly rely on heavy classifier-based curation and licensed high-quality corpora.

## Latest Developments (2023–2026)
The **"quality > quantity"** view gained strong evidence: phi-series and DCLM showed curated/filtered data dramatically improves quality-per-token. **LLM-as-annotator** quality scoring (FineWeb-Edu) replaced hand-tuned heuristics. Debate persists over whether the phi approach **overfits benchmarks** (its synthetic data may resemble eval distributions) versus genuinely improving capability — a live contamination concern.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain MinHash + LSH and how you'd tune the threshold."** Jaccard estimation, banding S-curve $1-(1-s^r)^b$, $b,r$ tradeoff.
- **"Why does deduplication improve models?"** Less memorization, more effective tokens, less wasted compute, better eval validity (Lee et al.).
- **"Quality vs quantity — how do you decide filter strength?"** Depends on compute budget vs data availability; cite phi vs Chinchilla.
- **"Risks of model-based quality filters?"** Style/dialect bias, benchmark contamination, distribution narrowing.

## Open Problems
There is no agreed metric for "data quality" — classifiers are proxies. Whether the phi "textbook" gains generalize beyond benchmarks (vs reflecting contamination) is contested. The optimal dedup aggressiveness, the safe limit of repetition, and how to filter without amplifying bias are unsolved.

## References
- Lee, K. et al. (2022). *Deduplicating Training Data Makes LMs Better.* arXiv:2107.06499.
- Wenzek, G. et al. (2019). *CCNet.* arXiv:1911.00359.
- Gunasekar, S. et al. (2023). *Textbooks Are All You Need (phi-1).* arXiv:2306.11644.
- Abbas, A. et al. (2023). *SemDeDup.* arXiv:2303.09540.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained LMs.* arXiv:2305.16264.
