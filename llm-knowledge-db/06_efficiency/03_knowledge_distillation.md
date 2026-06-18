# Knowledge Distillation

> **Last Updated:** 2026-06-18
> **Related Files:** [Synthetic Data Generation](../01_data/04_synthetic_data_generation.md) · [Speculative Decoding](04_speculative_decoding.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)
> **Key Papers:** Hinton et al. 2015 Distilling the Knowledge in a NN ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)) · Sanh et al. 2019 DistilBERT ([arXiv:1910.01108](https://arxiv.org/abs/1910.01108)) · Kim & Rush 2016 Sequence-Level KD ([arXiv:1606.07947](https://arxiv.org/abs/1606.07947)) · Agarwal et al. 2023 GKD (on-policy) ([arXiv:2306.13649](https://arxiv.org/abs/2306.13649))

## Overview
Knowledge distillation (KD) transfers the capability of a large, expensive **teacher** model into a smaller, cheaper **student** that approximates the teacher's behavior. KD is how the field produces small models that punch above their weight (DistilBERT, TinyLlama, Gemma, the DeepSeek-R1-Distill series) and is increasingly central to the post-reasoning era, where reasoning traces from a strong teacher (o1/R1) are distilled into small models that inherit much of the reasoning ability at a fraction of the inference cost. KD also underlies speculative-decoding draft models and on-device deployment.

## Core Concepts
**Classic (logit) distillation.** Hinton et al.: train the student to match the teacher's **soft probability distribution** (not just the hard label). The teacher's softened logits (via temperature $T$) carry "**dark knowledge**" — the relative probabilities of wrong classes encode similarity structure the hard label lacks. Loss:
$$\mathcal{L}=\alpha\, T^2\, D_{KL}\!\big(p_T^{(T)}\,\|\,p_S^{(T)}\big) + (1-\alpha)\,\mathcal{L}_{CE}(y, p_S)$$
where $p^{(T)}=\text{softmax}(z/T)$ and the $T^2$ rescales gradients. For LLMs, the analogue is matching the teacher's full **next-token distribution** at each position (forward-KL).

**Sequence-level KD (Kim & Rush).** For generation, distill on the *sequences the teacher produces*: generate teacher outputs and train the student with standard CLM loss on them. This is effectively **training on teacher-generated data** — simple, scalable, and the basis of most modern LLM distillation (it's also "distillation = synthetic data from a teacher," see [Synthetic Data](../01_data/04_synthetic_data_generation.md)).

**White-box vs black-box.**
- **White-box** (teacher logits available): match full distributions per token — richer signal, needs teacher access and shared tokenizer.
- **Black-box** (API only): train on teacher *outputs* (text), optionally with CoT/explanations (Orca-style). Most open distillation of frontier models is black-box.

**On-policy / generative KD (GKD).** A key failure of offline sequence-KD: **distribution mismatch** — the student trains on teacher trajectories but at inference sees its *own* (exposure bias). **On-policy KD** (GKD) has the student generate sequences and the teacher *score/correct* them (minimizing divergence on the student's own distribution), fixing the mismatch and improving results. Choice of divergence (forward vs reverse KL, JSD) matters: reverse KL is mode-seeking (more focused student).

**Reasoning distillation.** Distill *chain-of-thought traces* from a reasoning teacher. DeepSeek showed **distilling R1's reasoning into small dense models** (1.5B–70B) often **beats running RL on the small model directly** — a major, somewhat surprising result reshaping how small reasoning models are built.

## Key Challenges
- **Exposure bias / distribution mismatch.** Offline KD trains on teacher outputs but the student decodes its own — degrades quality (fixed by on-policy KD).
- **Capacity gap.** A very small student can't represent a much larger teacher's function; too-large a gap hurts transfer (sometimes a mid-size "teacher assistant" helps).
- **Tokenizer/vocabulary mismatch.** White-box logit matching requires shared tokenizers; cross-tokenizer distillation is hard.
- **Inheriting flaws.** The student copies the teacher's biases, hallucinations, and errors; ceiling is the teacher.
- **Legal/ToS.** Distilling closed-API models may violate terms of service and raises IP questions.

## Solutions & Current Best Practices
For LLMs, **sequence-level / data distillation** (train on high-quality teacher generations, filtered/verified) is the workhorse; add **on-policy KD (GKD)** to fix exposure bias when teacher access allows. For **reasoning**, distill **verified CoT traces** (keep correct ones — STaR/rejection sampling) from a strong teacher; this is now standard for small reasoning models. Use **temperature/soft-label** matching when logits are available and tokenizers match. Combine KD with **quantization** (small + quantized student) for deployment. Validate the student doesn't just memorize.

## Lab Perspectives
- **Google:** DistilBERT lineage, GKD (on-policy KD), Gemini **Flash** models distilled from larger Gemini; Gemma distillation.
- **DeepSeek:** **R1-Distill** series — distilling reasoning into Qwen/Llama bases; argued distillation > small-model RL for cost-efficiency.
- **OpenAI/Anthropic:** ship distilled "mini/haiku/flash"-tier models (smaller, cheaper variants of flagships); offer **distillation APIs** (store teacher outputs to fine-tune students).
- **Meta/open community:** TinyLlama, MiniLM, and countless distilled open models.

## Latest Developments (2023–2026)
**Reasoning distillation** became the dominant way to make small reasoning models (R1-Distill, Qwen-distill), often outperforming direct small-model RL. **On-policy KD** (GKD, MiniLLM with reverse-KL) addressed exposure bias. **Distillation scaling laws** (2024–2025) characterized when distillation beats from-scratch training (it helps most when teacher is strong and student-compute is limited). Frontier labs productized **distillation pipelines** and small/fast model tiers as core offerings.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why use soft labels / what is 'dark knowledge'?"** Teacher's full distribution encodes inter-class similarity the hard label omits; richer gradient signal.
- **"Sequence-level vs logit distillation for LLMs?"** Train on teacher generations (scalable, black-box) vs match per-token distributions (richer, needs logits + shared vocab).
- **"What is the distribution-mismatch problem and how does on-policy KD fix it?"** Student trains on teacher trajectories but decodes its own; GKD trains on student-generated sequences scored by the teacher.
- **"Why distill R1 into a small model instead of RL-ing it directly?"** Empirically cheaper and better — the small model inherits reasoning the teacher discovered via expensive RL.

## Open Problems
The **capacity-gap** limit (how small a student can capture a given teacher) lacks clean theory. **Cross-tokenizer** white-box distillation, distilling **agentic/tool-use** competence, and whether students can ever **exceed** their teacher (vs being capped) are open. The interaction of distillation with RL (distill-then-RL vs RL-then-distill) is actively studied.

## References
- Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
- Sanh, V. et al. (2019). *DistilBERT.* arXiv:1910.01108.
- Kim, Y., Rush, A. (2016). *Sequence-Level Knowledge Distillation.* arXiv:1606.07947.
- Agarwal, R. et al. (2023). *GKD (On-Policy Distillation).* arXiv:2306.13649.
- DeepSeek-AI (2025). *DeepSeek-R1 (distillation).* arXiv:2501.12948.
