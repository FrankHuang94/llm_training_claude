# Data Mixing and Curriculum

> **Last Updated:** 2026-11-04
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
**Proxy-model mixture search** (DoReMi / IsoFLOP sweeps over weights) to set static proportions, **upsample** high-quality small domains, then a **two-phase schedule**: a large "stable" phase on the broad mixture followed by an **annealing phase** on a high-quality, instruction/math/code-rich mixture during LR decay (a natural fit with WSD schedules — see [Optimizers](../02_pretraining/03_optimizer_choices.md)). Continuously monitor per-domain validation loss to catch over/under-weighting. 2025–26 practice adds two more stages: a **length curriculum** for long context (short-window dense training before long-window or sparse-attention stages) and a **post-training environment mix** for RL whose composition is itself treated as a capability and safety decision.

## Lab Perspectives
- **Google DeepMind** introduced DoReMi and uses systematic mixture optimization (Gemini/Gemma).
- **Meta** (Llama 3) disclosed careful domain weights (notably large code fraction) plus a high-quality annealing phase.
- **DeepSeek/Qwen** publicly emphasize large code+math fractions for reasoning, with staged curricula; DeepSeek-V4 reportedly stages context (dense 4K→16K, sparse attention switched on at 64K, then 1M).
- **Anthropic** treats the mix as a capability-shaping tool: Opus 4.7's cyber capability was deliberately reduced during training and Opus 5 was not trained on cyber tasks.
- **Microsoft phi** is the extreme of curation: the "mixture" is dominated by synthetic textbook-quality data.

## Latest Developments (2023–Nov 2026)
**Midtraining/annealing** became a standard, distinct phase. **Multi-stage curricula** (general → domain-dense → long-context → high-quality) are common. Research on **predictive mixing laws** (RegMix, data mixing laws) fits functional forms to predict downstream loss from mixture weights, enabling near-optimal mixtures from small experiments. The data wall pushes more **synthetic data** into the mixture, raising fresh weighting questions.

**Mid-2025 → Nov 2026 update:**
- **Timing × weight interaction (2025-10; arXiv:2510.14865).** "Midtraining Bridges Pretraining and Posttraining Distributions" finds that introducing specialized data early tolerates a high mixture weight, whereas late introduction needs a low weight; gains are largest for math and code. Mixture weights and schedule cannot be tuned independently.
- **Midtraining sets the RL ceiling (2026-03; arXiv:2603.17074).** PRISM: adding science data at midtraining unlocks +17–28 points on GPQA-Diamond *after* RL, while changing the RL data mix moves results by <2 points; midtraining restructures >90% of weights, RL sparsely updates ~5% of parameters. The most consequential mixture decision is upstream of RL.
- **Structured synthetic data as a mixture component (2026-04; arXiv:2604.13977).** FinePhrase shows rephrased web text in structured formats (tables, math problems, FAQs, tutorials) outperforms curated web data, giving a new high-quality domain whose weight must be set against natural data (see [Synthetic Data](04_synthetic_data_generation.md)).
- **Dedup-informed upsampling across languages (2025-06; arXiv:2506.20920).** FineWeb2's "rehydration" turns duplicate counts into per-document upsampling weights across >1,000 languages — a data-driven alternative to hand-set per-language weights.
- **Length curricula for 1M context (2026-04).** DeepSeek-V4 reportedly trains dense attention at 4K→16K, enables sparse attention at 64K, and reaches a 1M default context; Kimi K3, MiniMax M3 and Nemotron 3 also ship 1M windows. Long-context data is now its own late-stage mixture with its own weight.
- **RL environment mixes and data volume (2026).** Agentic RL hinges on environment scaling — synthesized repository tasks (KAT-Coder-V2.5, arXiv:2607.05471) and environments mined from existing code (CodeMidas, arXiv:2609.22068). In data-constrained RL post-training, total data volume reportedly matters more than uniqueness (Tan et al., ACL 2026; secondary), echoing pretraining's repetition results.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How would you choose data mixture weights without retraining many times?"** Proxy models + DoReMi/Group DRO or mixing-law regression.
- **"What is annealing/midtraining and why does it help?"** Late-phase shift to high-quality data during LR decay; large benchmark gains.
- **"Why upsample code if you care about reasoning?"** Code improves structured/logical reasoning broadly.
- **"What is the curse of multilinguality?"** Fixed capacity split across languages trades off per-language quality.
- **"Your RL run plateaus on science QA. Change the RL mix or the midtraining mix?"** Evidence (PRISM) favors midtraining: RL mostly elicits what midtraining installed; also check timing/weight interaction before raising the domain's weight late.

## Open Problems
Whether there is a *universal* optimal mixture or whether it is fundamentally model/scale/objective dependent is open. The theory of *why* code/math transfer to general reasoning is immature. Optimal curriculum ordering, the right amount of late-stage specialization without forgetting, and principled weighting of synthetic vs natural data are all unsolved.

## References
- Xie, S. M. et al. (2023). *DoReMi.* arXiv:2305.10429.
- Albalak, A. et al. (2023). *Online Data Mixing.* arXiv:2312.02406.
- Liu, Q. et al. (2024). *RegMix.* arXiv:2407.01492.
- Ye, J. et al. (2024). *Data Mixing Laws.* arXiv:2403.16952.
- Blakeney, C. et al. (2024). *Does Your Data Spark Joy? (curriculum/annealing).* arXiv:2406.03476.
- CMU (2025). *Midtraining Bridges Pretraining and Posttraining Distributions.* arXiv:2510.14865.
- Penedo, G. et al. (2025). *FineWeb2: One Pipeline to Scale Them All.* arXiv:2506.20920.
