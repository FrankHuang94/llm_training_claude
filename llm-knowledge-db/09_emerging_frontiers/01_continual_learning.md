# Continual Learning

> **Last Updated:** 2026-11-04
> **Related Files:** [Long Context & Memory](00_long_context_and_memory.md) · [LoRA & Variants](../06_efficiency/01_LoRA_and_variants.md) · [Test-Time Training](03_test_time_training.md) · [Knowledge Distillation](../06_efficiency/03_knowledge_distillation.md)
> **Key Papers:** Kirkpatrick et al. 2017 EWC ([arXiv:1612.00796](https://arxiv.org/abs/1612.00796)) · French 1999 Catastrophic Forgetting · Ke et al. 2023 Continual Pretraining ([arXiv:2302.03241](https://arxiv.org/abs/2302.03241)) · Ibrahim et al. 2024 Simple Continual Pretraining ([arXiv:2403.08763](https://arxiv.org/abs/2403.08763)) · Lin et al. 2025 Sparse Memory Finetuning ([arXiv:2510.15103](https://arxiv.org/abs/2510.15103))

## Overview
Continual (lifelong) learning is the ability to acquire new knowledge and skills *over time* without forgetting what was learned before — a capability humans have and current LLMs largely lack. Today's models are **static after training**: their knowledge is frozen at the pretraining cutoff, and updating them (new facts, new domains, new preferences) risks **catastrophic forgetting** of prior capabilities. As the world changes and deployment lifetimes lengthen, the inability to update efficiently is a major limitation — driving interest in continual pretraining, parameter-efficient updates, and the deeper "plasticity vs stability" problem. This is a key open frontier and a frequent research-direction interview topic.

## Core Concepts
**Catastrophic forgetting.** When a neural network trained on task A is then trained on task B, gradient updates for B overwrite the weights encoding A, sharply degrading A performance. The root cause is **shared, distributed representations** updated by SGD without regard to their importance for old tasks.

**The stability–plasticity dilemma.** A learner must be **plastic** enough to absorb new information yet **stable** enough to retain old knowledge — these pull in opposite directions. Too stable → can't learn new things; too plastic → forgets.

**Major method families.**
- **Regularization-based:** penalize changes to weights important for old tasks. **EWC (Elastic Weight Consolidation)** uses the **Fisher information** to estimate per-parameter importance and adds a quadratic penalty $\sum_i \frac{\lambda}{2} F_i (\theta_i-\theta_i^*)^2$ anchoring important weights near their old values.
- **Rehearsal / replay:** mix in (or generate) examples from old data/tasks while learning new ones — the most reliably effective approach; for LLMs, **replaying a fraction of pretraining data** during continual pretraining largely prevents forgetting.
- **Architecture/parameter isolation:** allocate new capacity per task — **LoRA adapters per domain/skill** (freeze base, add adapters), progressive networks, MoE-style routing. Avoids interference by not touching shared weights (see [LoRA & Variants](../06_efficiency/01_LoRA_and_variants.md)).
- **Continual pretraining recipes:** Ibrahim et al. (2024) show **LR re-warming + re-decaying + replaying a small % of old data** lets you continually pretrain on new data (new domain/language/time period) with minimal forgetting — a practical, simple recipe now widely used.

**Sparse / memory-layer updates.** A 2025 direction: give the model a large **memory layer** (key–value slots) and, when learning new facts, update only the few slots that are *specific* to the new data (accessed often in this batch, rarely during pretraining — a TF-IDF-like score). Isolating updates to rarely shared parameters attacks the root cause (shared, distributed representations) rather than penalizing drift after the fact.

**Knowledge editing.** Targeted fact updates (ROME, MEMIT) directly edit specific weights to change a fact (e.g., a leader's name) without retraining — a complementary, surgical alternative to full continual learning, though it can have side effects and scaling limits.

## Key Challenges
- **Forgetting at scale.** Even with replay, long sequences of updates accumulate drift; truly *unbounded* lifelong learning isn't solved.
- **No access to original data.** Privacy/cost may forbid replaying pretraining data; generative replay is imperfect.
- **Plasticity loss.** Repeatedly-trained networks can lose the *ability to learn* (dormant neurons, rank collapse) — "loss of plasticity," distinct from forgetting.
- **Evaluation.** Measuring retention vs acquisition across many tasks/time is hard; backward/forward transfer metrics are imperfect.
- **Interference vs transfer.** New learning can either hurt (interfere) or help (positive transfer) old tasks; controlling this is unsolved.
- **Post-training as a continual-learning problem.** Domain mid-training or SFT routinely degrades instruction-following and safety behaviors learned earlier; every new post-training stage is an instance of forgetting control.

## Solutions & Current Best Practices
For **continual pretraining** (updating a base model with new data): **LR re-warm/re-decay + replay ~1–5% of original data** (Ibrahim et al.) — simple and effective. For **multi-skill/domain adaptation**: **per-task LoRA adapters** (isolation) or rehearsal-augmented fine-tuning. For **recovering behaviors lost after domain training**: **on-policy distillation** from the earlier checkpoint (student samples, teacher gives per-token reverse-KL) restores instruction-following cheaply. For **new facts with minimal forgetting**: **sparse memory-layer finetuning**. For **targeted facts**: **knowledge editing** (MEMIT) or, more robustly, **RAG** (externalize changing knowledge rather than updating weights — often the pragmatic answer). In practice, the field largely **sidesteps** continual learning via **retrieval** (update the database, not the weights) and **periodic full retraining**.

## Lab Perspectives
- **Frontier labs** mostly handle "new knowledge" via **retrieval + periodic retraining** rather than online continual learning (RAG updates the knowledge store; new model generations refresh weights — now at a 4–8-week cadence per lab, which itself is a coarse form of continual learning). Agents get "memory" via context engineering and compaction rather than weight updates.
- **Google Research:** **Nested Learning / HOPE** reframes architecture and optimizer as nested learners at different update frequencies, explicitly targeting continual learning.
- **Meta FAIR:** **sparse memory finetuning** on memory-layer models as a forgetting-resistant update path.
- **Thinking Machines:** LoRA-without-regret and on-policy distillation results reframe cheap adapter updates and behavior recovery after domain training.
- **Academia/DeepMind:** plasticity-loss research (continual RL, "dormant neuron" phenomena), EWC lineage, and replay studies.
- **Open community:** continual pretraining recipes (domain/language adaptation of Llama/Qwen) and per-task adapters are widely used.
- **Test-time-training researchers** treat continual/online adaptation as part of inference (see [Test-Time Training](03_test_time_training.md)).

## Latest Developments (2023–Nov 2026)
Practical **continual pretraining recipes** (re-warm/re-decay + replay) became standard for cheaply updating base models with new data/time periods without full retraining. **Plasticity loss** gained attention as a distinct failure mode. **LoRA-based continual learning** and **model merging** (averaging task-specialized models) emerged as cheap composition tools. The dominant *production* answer to "stale knowledge," however, remains **RAG + scheduled retraining**, leaving true online lifelong learning an open research goal.

**Mid-2025 → Nov 2026 update:**
- **Sparse memory finetuning (2025-10, Meta FAIR).** Memory-layer models update only the top-$t$ slots ranked by a TF-IDF-like ratio (frequent in the batch, rare in pretraining). On NaturalQuestions, forgetting is 11% vs 89% for full fine-tuning and 71% for LoRA at equal new-knowledge acquisition ([arXiv:2510.15103](https://arxiv.org/abs/2510.15103)) — parameter isolation done *per update*, not per task.
- **Nested Learning / HOPE (2025-11, Google, NeurIPS 2025).** Treats a model as nested optimization problems with different update frequencies (architecture and optimizer as the same concept at different levels); HOPE is a self-modifying Titans variant with a "continuum memory system" of MLP blocks updated at multiple frequencies — a principled multi-timescale answer to stability–plasticity.
- **Long context as continual learning (2025-12).** TTT-E2E updates some MLP weights with next-token gradient steps on the context at test time, from a meta-learned initialization ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)); blurs in-context learning, memory, and weight updates (see [Test-Time Training](03_test_time_training.md)).
- **On-policy distillation as a forgetting repair (2025-10, Thinking Machines).** Student samples its own trajectories; teacher supplies dense per-token reverse-KL. Reported to recover instruction-following after domain mid-training at a fraction of RL cost (blog; secondary coverage verified). By 2026 multi-teacher OPD is used to *merge* domain experts (Qwen3, KAT-Coder-V2.5).
- **LoRA Without Regret (2025-09).** LoRA on all layers (esp. MLP/MoE) matches full FT with ~10× the LR; for RL even rank-1 suffices since policy gradient absorbs ~1 bit/episode (blog; secondary coverage) — supports adapter-based, low-interference updates.
- **Where updates land (2026-03).** PRISM reports midtraining restructures >90% of weights while RL makes sparse updates to ~5% of parameters ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)); with CMU's finding that the *timing* of specialized data interacts with its mixture weight ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865)), this reframes forgetting as a property of *which stage* edits the weights.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What is catastrophic forgetting and why does it happen?"** New-task gradients overwrite shared weights encoding old tasks.
- **"Explain EWC."** Fisher-weighted quadratic penalty anchoring important weights near old values.
- **"How would you add a new domain/language to a base model without forgetting?"** Continual pretraining: LR re-warm/re-decay + replay a small fraction of old data; or per-domain LoRA.
- **"Why do labs often use RAG instead of updating weights?"** Externalizing changing knowledge avoids forgetting/retraining cost — update the database, not the model.
- **"Stability–plasticity dilemma?"** Retain old vs absorb new; the core tension.
- **"Your domain fine-tune broke instruction-following — how do you fix it without redoing post-training?"** On-policy distillation from the pre-domain checkpoint (dense reverse-KL on the student's own samples), replay of general SFT data, or restrict updates (all-layer LoRA, sparse memory slots).
- **"Why might RL forget less than SFT?"** RL updates are sparse/low-information (~1 bit per episode; ~5% of parameters move), whereas SFT/midtraining rewrites most weights.

## Open Problems
Truly **unbounded lifelong learning** without forgetting or plasticity loss is unsolved. Whether multi-timescale architectures (HOPE), sparse memory layers, or test-time-trained weights scale to frontier models — and how to make *deployed* updates safe (a weight update is also an unreviewed behavior change) — are new open questions. Updating **parametric knowledge** reliably and surgically (editing without side effects), learning **online** during deployment, and the deep question of *why* SGD-trained nets forget (and how brains avoid it) remain central open problems linking ML, neuroscience, and systems.

## References
- Kirkpatrick, J. et al. (2017). *Overcoming Catastrophic Forgetting (EWC).* arXiv:1612.00796.
- Ke, Z. et al. (2023). *Continual Pre-training of LMs.* arXiv:2302.03241.
- Ibrahim, A. et al. (2024). *Simple and Scalable Strategies to Continually Pre-train LLMs.* arXiv:2403.08763.
- Meng, K. et al. (2022). *Mass-Editing Memory (MEMIT).* arXiv:2210.07229.
- Dohare, S. et al. (2024). *Loss of Plasticity in Deep Continual Learning.* Nature.
- Lin, J. et al. (2025). *Continual Learning via Sparse Memory Finetuning.* arXiv:2510.15103.
- Stanford/NVIDIA et al. (2025). *End-to-End Test-Time Training for Long Context (TTT-E2E).* arXiv:2512.23675.
- CMU et al. (2025). *Midtraining Bridges Pretraining and Posttraining Distributions.* arXiv:2510.14865.
- Google Research (2025). *Nested Learning / HOPE.* NeurIPS 2025; Google Research blog.
