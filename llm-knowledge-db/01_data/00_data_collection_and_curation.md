# Data Collection and Curation

> **Last Updated:** 2026-06-18
> **Related Files:** [Data Quality & Deduplication](01_data_quality_and_deduplication.md) · [Data Mixing & Curriculum](03_data_mixing_and_curriculum.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md)
> **Key Papers:** Raffel et al. 2020 C4/T5 ([arXiv:1910.10683](https://arxiv.org/abs/1910.10683)) · Gao et al. 2020 The Pile ([arXiv:2101.00027](https://arxiv.org/abs/2101.00027)) · Penedo et al. 2024 FineWeb ([arXiv:2406.17557](https://arxiv.org/abs/2406.17557)) · Li et al. 2024 DCLM ([arXiv:2406.11794](https://arxiv.org/abs/2406.11794))

## Overview
Pretraining data is the single largest determinant of base-model quality — arguably more than architecture at fixed scale. Frontier models consume 10–30+ trillion tokens, the overwhelming majority sourced from web crawls. The pipeline that turns petabytes of raw HTML into a clean, deduplicated, quality-filtered token stream is a major engineering artifact, and the details (which filters, which mixture, how much dedup) are among the most closely guarded secrets at frontier labs. The open-source community (The Pile → RedPajama → DCLM → FineWeb) has progressively reverse-engineered competitive pipelines.

The central tension is **scale vs quality**: Chinchilla-optimal training needs enormous token counts, but naive web text is noisy, duplicated, and full of boilerplate. Curation is the art of extracting maximal "effective tokens."

## Core Concepts
**Common Crawl (CC).** The raw substrate: ~petabytes/month of web pages in WARC (raw) / WET (text) format. A typical pipeline: (1) **text extraction** from HTML (trafilatura, resiliparse — better than CC's default WET), (2) **language identification** (fastText/CLD3), (3) **quality filtering** (heuristics + classifiers), (4) **deduplication** (exact + fuzzy/MinHash), (5) **PII/toxicity filtering**, (6) **decontamination** against eval sets.

**Landmark corpora.**
- **C4** (2020): Colossal Clean Crawled Corpus; aggressive heuristic cleaning (e.g., remove pages without terminal punctuation, "bad words" list). ~750GB.
- **The Pile** (2020, EleutherAI): 825GB curated *mix* of 22 high-quality sources (PubMed, ArXiv, GitHub, Books, StackExchange) — pioneered deliberate domain composition.
- **RedPajama** (2023): open reproduction of the LLaMA-1 mixture; RedPajama-v2 shipped 30T tokens with quality signals attached.
- **RefinedWeb** (2023, Falcon): showed *web-only*, heavily filtered+deduped data can match curated mixes.
- **DCLM** (2024): a benchmark + dataset showing **model-based filtering** (a fastText classifier trained to recognize high-quality text) drives most of the gains; DCLM-Baseline beat prior open sets.
- **FineWeb / FineWeb-Edu** (2024, HuggingFace): 15T-token open web corpus; FineWeb-Edu filters by an "educational value" classifier and substantially boosts benchmark performance.

## Key Challenges
- **Boilerplate and noise.** Navigation, ads, SEO spam dominate raw HTML; extraction quality compounds through the whole pipeline.
- **The data wall.** High-quality public text is finite (~tens of trillions of tokens); frontier models are approaching exhaustion, forcing synthetic data and repetition.
- **Legal & ethical exposure.** Copyright (NYT v. OpenAI), robots.txt/ToS, GDPR/PII, and licensing are unresolved and litigated.
- **Benchmark contamination.** Crawls inevitably contain test sets; undetected leakage inflates reported scores (see [Data Contamination](05_data_contamination_and_evaluation_leakage.md)).

## Solutions & Current Best Practices
The converged recipe: better **text extraction** (trafilatura/resiliparse over WET), **model-based quality classifiers** (DCLM/FineWeb-Edu style) rather than only heuristics, **global fuzzy dedup** (MinHash LSH), strict **eval decontamination**, and a final **annealing/midtraining** phase on the highest-quality subset (textbooks, curated code/math). Document-level provenance and quality signals are stored so mixtures can be re-weighted cheaply.

## Lab Perspectives
- **OpenAI/Anthropic/Google** keep pipelines closed; all are believed to use heavy classifier-based filtering and licensed/curated data deals.
- **Meta** disclosed Llama 3 trained on ~15T tokens with extensive dedup and quality filtering, plus large code/math/multilingual fractions.
- **DeepSeek** emphasizes code- and math-heavy mixtures and efficient pipelines.
- **HuggingFace/EleutherAI/AllenAI** (FineWeb, The Pile, Dolma) drive open, reproducible curation that the field benchmarks against.

## Latest Developments (2023–2026)
The shift from heuristic to **model-based filtering** (FineWeb-Edu, DCLM) is the headline result of 2024. Licensed-data partnerships (news, forums like Reddit, Stack Overflow) proliferated. **Synthetic and rephrased web data** ("Web Rephrase Augmented Pre-training", phi-series) emerged to beat the data wall. Multimodal and multilingual crawls expanded. Provenance/watermarking and opt-out registries (e.g., robots.txt extensions) became active policy areas.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Walk me through a Common Crawl → pretraining-data pipeline."** Extraction → langID → quality filter → dedup → decontam → mix.
- **"What changed between The Pile and FineWeb-Edu?"** Heuristic curation → model-based educational-quality filtering at web scale.
- **"How do you handle the data wall?"** Repetition (up to ~4 epochs), synthetic/rephrased data, multimodal, licensed sources.
- **"How do you prevent benchmark contamination during data collection?"** n-gram/embedding decontam against held-out evals.

## Open Problems
There is no principled theory of *how much* and *which* filtering is optimal — it is empirical and dataset-specific. Quantifying "effective tokens" vs raw tokens, the long-term ceiling imposed by the data wall, the value and risks of synthetic data (model collapse), and the legal status of web-scale training all remain open and contested.

## References
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning (C4/T5).* arXiv:1910.10683.
- Gao, L. et al. (2020). *The Pile.* arXiv:2101.00027.
- Penedo, G. et al. (2023). *RefinedWeb.* arXiv:2306.01116.
- Penedo, G. et al. (2024). *FineWeb / FineWeb-Edu.* arXiv:2406.17557.
- Li, J. et al. (2024). *DataComp-LM (DCLM).* arXiv:2406.11794.
