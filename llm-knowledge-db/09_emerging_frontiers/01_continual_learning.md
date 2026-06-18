# Continual Learning

> **Last Updated:** 2026-06-18
> **Related Files:** [Long Context & Memory](00_long_context_and_memory.md) · [LoRA & Variants](../06_efficiency/01_LoRA_and_variants.md) · [Test-Time Training](03_test_time_training.md)
> **Key Papers:** Kirkpatrick et al. 2017 EWC ([arXiv:1612.00796](https://arxiv.org/abs/1612.00796)) · French 1999 Catastrophic Forgetting · Ke et al. 2023 Continual Pretraining ([arXiv:2302.03241](https://arxiv.org/abs/2302.03241)) · Ibrahim et al. 2024 Simple Continual Pretraining ([arXiv:2403.08763](https://arxiv.org/abs/2403.08763))

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

**Knowledge editing.** Targeted fact updates (ROME, MEMIT) directly edit specific weights to change a fact (e.g., a leader's name) without retraining — a complementary, surgical alternative to full continual learning, though it can have side effects and scaling limits.

## Key Challenges
- **Forgetting at scale.** Even with replay, long sequences of updates accumulate drift; truly *unbounded* lifelong learning isn't solved.
- **No access to original data.** Privacy/cost may forbid replaying pretraining data; generative replay is imperfect.
- **Plasticity loss.** Repeatedly-trained networks can lose the *ability to learn* (dormant neurons, rank collapse) — "loss of plasticity," distinct from forgetting.
- **Evaluation.** Measuring retention vs acquisition across many tasks/time is hard; backward/forward transfer metrics are imperfect.
- **Interference vs transfer.** New learning can either hurt (interfere) or help (positive transfer) old tasks; controlling this is unsolved.

## Solutions & Current Best Practices
For **continual pretraining** (updating a base model with new data): **LR re-warm/re-decay + replay ~1–5% of original data** (Ibrahim et al.) — simple and effective. For **multi-skill/domain adaptation**: **per-task LoRA adapters** (isolation) or rehearsal-augmented fine-tuning. For **targeted facts**: **knowledge editing** (MEMIT) or, more robustly, **RAG** (externalize changing knowledge rather than updating weights — often the pragmatic answer). In practice, the field largely **sidesteps** continual learning via **retrieval** (update the database, not the weights) and **periodic full retraining**.

## Lab Perspectives
- **Frontier labs** mostly handle "new knowledge" via **retrieval + periodic retraining** rather than online continual learning (RAG updates the knowledge store; new model generations refresh weights).
- **Academia/DeepMind:** plasticity-loss research (continual RL, "dormant neuron" phenomena), EWC lineage, and replay studies.
- **Open community:** continual pretraining recipes (domain/language adaptation of Llama/Qwen) and per-task adapters are widely used.
- **Test-time-training researchers** treat continual/online adaptation as part of inference (see [Test-Time Training](03_test_time_training.md)).

## Latest Developments (2023–2026)
Practical **continual pretraining recipes** (re-warm/re-decay + replay) became standard for cheaply updating base models with new data/time periods without full retraining. **Plasticity loss** gained attention as a distinct failure mode. **LoRA-based continual learning** and **model merging** (averaging task-specialized models) emerged as cheap composition tools. The dominant *production* answer to "stale knowledge," however, remains **RAG + scheduled retraining**, leaving true online lifelong learning an open research goal.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What is catastrophic forgetting and why does it happen?"** New-task gradients overwrite shared weights encoding old tasks.
- **"Explain EWC."** Fisher-weighted quadratic penalty anchoring important weights near old values.
- **"How would you add a new domain/language to a base model without forgetting?"** Continual pretraining: LR re-warm/re-decay + replay a small fraction of old data; or per-domain LoRA.
- **"Why do labs often use RAG instead of updating weights?"** Externalizing changing knowledge avoids forgetting/retraining cost — update the database, not the model.
- **"Stability–plasticity dilemma?"** Retain old vs absorb new; the core tension.

## Open Problems
Truly **unbounded lifelong learning** without forgetting or plasticity loss is unsolved. Updating **parametric knowledge** reliably and surgically (editing without side effects), learning **online** during deployment, and the deep question of *why* SGD-trained nets forget (and how brains avoid it) remain central open problems linking ML, neuroscience, and systems.

## References
- Kirkpatrick, J. et al. (2017). *Overcoming Catastrophic Forgetting (EWC).* arXiv:1612.00796.
- Ke, Z. et al. (2023). *Continual Pre-training of LMs.* arXiv:2302.03241.
- Ibrahim, A. et al. (2024). *Simple and Scalable Strategies to Continually Pre-train LLMs.* arXiv:2403.08763.
- Meng, K. et al. (2022). *Mass-Editing Memory (MEMIT).* arXiv:2210.07229.
- Dohare, S. et al. (2024). *Loss of Plasticity in Deep Continual Learning.* Nature.
