# Data Mixing and Curriculum

> **Last Updated:** 2026-06-18
> **Related Files:** [Data Collection & Curation](00_data_collection_and_curation.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md) · [Pretraining Objectives](../02_pretraining/00_pretraining_objectives.md)
> **Key Papers:** Xie et al. 2023 DoReMi ([arXiv:2305.10429](https://arxiv.org/abs/2305.10429)) · Albalak et al. 2023 Online Data Mixing ([arXiv:2312.02406](https://arxiv.org/abs/2312.02406)) · Gao et al. 2020 The Pile ([arXiv:2101.00027](https://arxiv.org/abs/2101.00027)) · Blakeney et al. 2024 "Does your data spark joy?" curriculum ([arXiv:2406.03476](https://arxiv.org/abs/2406.03476))

## Overview
A pretraining corpus is a *mixture* of domains — web, code, books, academic papers, math, multilingual, Q&A — and the **mixture weights** (how much of each domain the model sees) are a first-class hyperparameter with large effects on capability. Upweighting code improves reasoning even on non-code tasks; too much of any narrow domain degrades general ability. Beyond static proportions, the *order* in which data is presented (**curriculum**) and end-of-training **annealing** on high-quality data are now standard levers. Because a full retraining is prohibitively expensive, labs invest heavily in cheaply predicting good mixtures from small proxy runs.

## Core Concepts
**Domain weights.** Each domain $i$ contributes a fraction $w_i$ of training tokens, $\sum_i w_i = 1$. Naive proportional-to-availability weighting is suboptimal because domains differ in quality and marginal value. Hand-tuned weights (The Pile, LLaMA-1) were the early norm; LLaMA-1's mixture (~67% CommonCrawl/C4, ~15% code+arxiv+books+wiki) became a community reference.

**Upsampling and capping.** High-value but small domains (Wikipedia, code, math, books) are **upsampled** (repeated) relative to their raw size; low-value or huge domains are **capped**. Repetition interacts with data-constrained scaling (a few epochs of good data ≈ fresh data).

**DoReMi (Domain Reweighting with Minimax Optimization).** Xie et al. (2023): train a small **proxy** model, use **Group Distributionally Robust Optimization (Group DRO)** to find domain weights that minimize worst-case excess loss across domains, then use those weights to train the large model. It learns weights *automatically* and transferred across scales, improving downstream accuracy and speeding training. The key insight: optimize weights on a cheap proxy, deploy on the expensive run.

**Online / adaptive mixing.** Rather than fixed weights, adjust the mixture *during* training based on per-domain loss or learning signals (Online Data Mixing, multi-armed-bandit formulations) — the model "pulls" more from domains where it is still improving.

**Curriculum & annealing.** Order matters: presenting easier/general data first and harder/specialized data later can help. The dominant modern practice is **annealing / midtraining**: in the final 10–20% of training (the LR decay phase), shift the mixture sharply toward the **highest-quality** data (curated math, code, textbooks, instruction-like data). Llama 3 and many others credit this phase with large benchmark gains.

## Key Challenges
- **Combinatorial search.** The space of mixtures is huge; full-scale ablation is infeasible, so proxy-based methods are essential but imperfect.
- **Proxy–target mismatch.** Weights tuned on a small proxy may not transfer to the target scale or architecture.
- **Interference and forgetting.** Heavy late-stage specialization risks forgetting general capability; sequencing must balance plasticity and retention.
- **Multilingual tradeoffs.** Adding languages can dilute English performance ("curse of multilinguality") unless capacity/weights are tuned.

## Solutions & Current Best Practices
**Proxy-model mixture search** (DoReMi / IsoFLOP sweeps over weights) to set static proportions, **upsample** high-quality small domains, then a **two-phase schedule**: a large "stable" phase on the broad mixture followed by an **annealing phase** on a high-quality, instruction/math/code-rich mixture during LR decay (a natural fit with WSD schedules — see [Optimizers](../02_pretraining/03_optimizer_choices.md)). Continuously monitor per-domain validation loss to catch over/under-weighting.

## Lab Perspectives
- **Google DeepMind** introduced DoReMi and uses systematic mixture optimization (Gemini/Gemma).
- **Meta** (Llama 3) disclosed careful domain weights (notably large code fraction) plus a high-quality annealing phase.
- **DeepSeek/Qwen** publicly emphasize large code+math fractions for reasoning, with staged curricula.
- **Microsoft phi** is the extreme of curation: the "mixture" is dominated by synthetic textbook-quality data.

## Latest Developments (2023–2026)
**Midtraining/annealing** became a standard, distinct phase. **Multi-stage curricula** (general → domain-dense → long-context → high-quality) are common. Research on **predictive mixing laws** (RegMix, data mixing laws) fits functional forms to predict downstream loss from mixture weights, enabling near-optimal mixtures from small experiments. The data wall pushes more **synthetic data** into the mixture, raising fresh weighting questions.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How would you choose data mixture weights without retraining many times?"** Proxy models + DoReMi/Group DRO or mixing-law regression.
- **"What is annealing/midtraining and why does it help?"** Late-phase shift to high-quality data during LR decay; large benchmark gains.
- **"Why upsample code if you care about reasoning?"** Code improves structured/logical reasoning broadly.
- **"What is the curse of multilinguality?"** Fixed capacity split across languages trades off per-language quality.

## Open Problems
Whether there is a *universal* optimal mixture or whether it is fundamentally model/scale/objective dependent is open. The theory of *why* code/math transfer to general reasoning is immature. Optimal curriculum ordering, the right amount of late-stage specialization without forgetting, and principled weighting of synthetic vs natural data are all unsolved.

## References
- Xie, S. M. et al. (2023). *DoReMi.* arXiv:2305.10429.
- Albalak, A. et al. (2023). *Online Data Mixing.* arXiv:2312.02406.
- Liu, Q. et al. (2024). *RegMix.* arXiv:2407.01492.
- Ye, J. et al. (2024). *Data Mixing Laws.* arXiv:2403.16952.
- Blakeney, C. et al. (2024). *Does Your Data Spark Joy? (curriculum/annealing).* arXiv:2406.03476.
