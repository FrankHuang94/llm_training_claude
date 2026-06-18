# Frontier Open Problems

> **Last Updated:** 2026-06-18
> **Related Files:** [Scaling Laws](../00_foundations/04_scaling_laws.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [Alignment Overview](../05_alignment_and_safety/00_alignment_overview.md)
> **Key Papers:** Schaeffer et al. 2023 Emergence Mirage ([arXiv:2304.15004](https://arxiv.org/abs/2304.15004)) · Dziri et al. 2023 Faith and Fate (compositionality) ([arXiv:2305.18654](https://arxiv.org/abs/2305.18654)) · Chollet 2019 On the Measure of Intelligence (ARC) ([arXiv:1911.01547](https://arxiv.org/abs/1911.01547)) · Bubeck et al. 2023 Sparks of AGI ([arXiv:2303.12712](https://arxiv.org/abs/2303.12712))

## Overview
Despite extraordinary progress, LLMs remain limited in ways that define the field's research agenda and the gap to robust, general intelligence. This file consolidates the **major open problems** — reliable reasoning, compositional generalization, sample efficiency, continual learning, formal verification, and AGI-level planning — that recur across the knowledge base. These are the questions frontier labs are organized around and the ones interviewers use to probe whether a candidate thinks critically about *limitations*, not just capabilities. Treat each as a live debate, not a solved checkbox.

## Core Open Problems
**1. Reliable reasoning.** Reasoning models (o1/R1) dramatically improved math/code, but reasoning remains **brittle**: sensitive to surface perturbations (GSM-Symbolic), prone to confident errors, and unfaithful (the stated chain may not reflect the true computation). Whether RL on verifiable rewards yields *robust, generalizable* reasoning or **sharper pattern-matching** is unresolved (see [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)).

**2. Compositional generalization.** Models struggle to **systematically combine** known primitives into novel structures. Dziri et al. ("Faith and Fate") show transformers solve multi-step compositional tasks (e.g., multi-digit multiplication) via memorized sub-patterns that break down as complexity grows, rather than learning the underlying algorithm. **ARC-AGI** (Chollet) targets exactly this abstraction-and-composition ability where LLMs long lagged humans.

**3. Sample efficiency.** LLMs need *trillions* of tokens; a human learns language from millions of words. This ~$10^3$–$10^4\times$ gap suggests current learning is far from optimal — likely missing strong inductive biases, grounding, or active/interactive learning. The **data wall** makes sample efficiency economically, not just scientifically, urgent.

**4. Continual / lifelong learning.** Models are **frozen** after training; updating them causes catastrophic forgetting (see [Continual Learning](01_continual_learning.md)). True lifelong learning — accumulating knowledge over a deployment lifetime without forgetting or plasticity loss — is unsolved, and the field largely sidesteps it with retrieval + retraining.

**5. Formal verification of behavior.** We cannot **guarantee** model properties (safety, honesty, correctness) — only test them empirically. Formal verification of neural network behavior at LLM scale is far beyond current methods, yet may be necessary for high-stakes deployment and for trusting superhuman systems (links to interpretability and [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)).

**6. AGI-level planning & agency.** Long-horizon, open-ended planning with reliable execution remains the key agentic bottleneck: errors compound, models lose coherence over many steps, and robust recovery is hard (see [Agentic Frameworks](../04_reasoning_and_agents/04_agentic_frameworks.md)). Reliable autonomy is the gating capability for transformative agents.

**7. Grounding & world models.** Whether language-trained models build robust **causal world models** (vs surface statistics) is contested (see [World Models](02_world_models.md)) — central to physical reasoning, planning, and genuine understanding.

**8. Alignment of superhuman systems.** As capability approaches/exceeds human level, **scalable oversight**, deception detection, and value specification become existential rather than usability concerns — the deepest open problem (see [Alignment Overview](../05_alignment_and_safety/00_alignment_overview.md)).

## Cross-Cutting Debates
- **Emergence: real or mirage?** Wei et al. (abrupt capability jumps) vs Schaeffer et al. (artifacts of discontinuous metrics) — matters for forecasting and safety (see [Scaling Laws](../00_foundations/04_scaling_laws.md)).
- **Is scaling enough?** "Scale is all you need" (continued pretraining + RL + inference compute) vs "new ideas required" (LeCun's JEPA, neuro-symbolic, world models). The data wall and reasoning brittleness fuel skepticism; reasoning-RL gains fuel optimism.
- **Understanding vs interpolation.** Do LLMs *understand* or perform sophisticated retrieval/interpolation? "Sparks of AGI" optimism vs "stochastic parrots" / compositionality-failure critiques.
- **Timeline to AGI.** Wildly divergent estimates; depends on whether the above problems yield to scale or require breakthroughs.

## Why These Matter (for research & interviews)
Labs are literally organized around these problems: reasoning (OpenAI o-series, DeepSeek), alignment/oversight (Anthropic), sample efficiency/architecture (SSM and JEPA researchers), agents/planning (everyone). A strong candidate can (a) name the problem precisely, (b) cite the key evidence on both sides, and (c) propose a credible research direction.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What are the biggest unsolved problems in LLMs?"** Reasoning robustness, compositionality, sample efficiency, continual learning, agentic planning, alignment of superhuman systems — with evidence.
- **"Do you think scaling is enough for AGI?"** Take a *nuanced* position citing data wall, reasoning brittleness, compositionality failures vs reasoning-RL and inference-scaling gains.
- **"Is emergence real?"** Engage Wei vs Schaeffer (metric artifacts).
- **"Why are LLMs so sample-inefficient vs humans?"** Missing inductive biases/grounding/active learning; trillions vs millions of tokens.
- **"What research direction excites you most and why?"** (Expect a specific, well-justified answer tied to a lab's priorities.)

## Open Problems (meta)
The field lacks consensus on **what intelligence even is** to measure (Chollet's "measure of intelligence"), making "are we close to AGI?" partly ill-posed. Whether progress will be **continuous** (scaling + incremental method gains) or require **paradigm shifts** (new architectures, grounding, neuro-symbolic, lifelong learning) is the overarching uncertainty that all the specific problems feed into.

## References
- Schaeffer, R. et al. (2023). *Are Emergent Abilities a Mirage?* arXiv:2304.15004.
- Dziri, N. et al. (2023). *Faith and Fate: Limits of Transformers on Compositionality.* arXiv:2305.18654.
- Chollet, F. (2019). *On the Measure of Intelligence (ARC).* arXiv:1911.01547.
- Bubeck, S. et al. (2023). *Sparks of Artificial General Intelligence.* arXiv:2303.12712.
- Mirzadeh, I. et al. (2024). *GSM-Symbolic.* arXiv:2410.05229.
