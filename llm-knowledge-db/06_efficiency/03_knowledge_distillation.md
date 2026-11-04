# Knowledge Distillation

> **Last Updated:** 2026-11-04
> **Related Files:** [Synthetic Data Generation](../01_data/04_synthetic_data_generation.md) · [Speculative Decoding](04_speculative_decoding.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)
> **Key Papers:** Hinton et al. 2015 Distilling the Knowledge in a NN ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)) · Sanh et al. 2019 DistilBERT ([arXiv:1910.01108](https://arxiv.org/abs/1910.01108)) · Kim & Rush 2016 Sequence-Level KD ([arXiv:1606.07947](https://arxiv.org/abs/1606.07947)) · Agarwal et al. 2023 GKD (on-policy) ([arXiv:2306.13649](https://arxiv.org/abs/2306.13649)) · Thinking Machines 2025 On-Policy Distillation (blog)

## Overview
Knowledge distillation (KD) transfers the capability of a large, expensive **teacher** model into a smaller, cheaper **student** that approximates the teacher's behavior. KD is how the field produces small models that punch above their weight (DistilBERT, TinyLlama, Gemma, the DeepSeek-R1-Distill series) and is increasingly central to the post-reasoning era, where reasoning traces from a strong teacher (o1/R1 and successors) are distilled into small models that inherit much of the reasoning ability at a fraction of the inference cost. Since late 2025, **on-policy distillation** has become a standard post-training stage in its own right — a dense-reward middle ground between SFT and RL — and is used to merge domain experts into one model. KD also underlies speculative-decoding draft models and on-device deployment.

## Core Concepts
**Classic (logit) distillation.** Hinton et al.: train the student to match the teacher's **soft probability distribution** (not just the hard label). The teacher's softened logits (via temperature $T$) carry "**dark knowledge**" — the relative probabilities of wrong classes encode similarity structure the hard label lacks. Loss:
$$\mathcal{L}=\alpha\, T^2\, D_{KL}\!\big(p_T^{(T)}\,\|\,p_S^{(T)}\big) + (1-\alpha)\,\mathcal{L}_{CE}(y, p_S)$$
where $p^{(T)}=\text{softmax}(z/T)$ and the $T^2$ rescales gradients. For LLMs, the analogue is matching the teacher's full **next-token distribution** at each position (forward-KL).

**Sequence-level KD (Kim & Rush).** For generation, distill on the *sequences the teacher produces*: generate teacher outputs and train the student with standard CLM loss on them. This is effectively **training on teacher-generated data** — simple, scalable, and the basis of most modern LLM distillation (it's also "distillation = synthetic data from a teacher," see [Synthetic Data](../01_data/04_synthetic_data_generation.md)).

**White-box vs black-box.**
- **White-box** (teacher logits available): match full distributions per token — richer signal, needs teacher access and shared tokenizer.
- **Black-box** (API only): train on teacher *outputs* (text), optionally with CoT/explanations (Orca-style). Most open distillation of frontier models is black-box.

**On-policy / generative KD (GKD).** A key failure of offline sequence-KD: **distribution mismatch** — the student trains on teacher trajectories but at inference sees its *own* (exposure bias). **On-policy KD** (GKD) has the student generate sequences and the teacher *score/correct* them (minimizing divergence on the student's own distribution), fixing the mismatch and improving results. Choice of divergence (forward vs reverse KL, JSD) matters: reverse KL is mode-seeking (more focused student). In the RL framing, the student samples $y\sim\pi_\theta$ and receives a dense per-token reward
$$r_t = \log \pi_T(y_t\mid y_{<t}) - \log \pi_\theta(y_t\mid y_{<t}),$$
whose expectation is the negative per-token reverse KL — RL-like on-policyness with SFT-like dense supervision.

**Reasoning distillation.** Distill *chain-of-thought traces* from a reasoning teacher. DeepSeek showed **distilling R1's reasoning into small dense models** (1.5B–70B) often **beats running RL on the small model directly** — a major, somewhat surprising result reshaping how small reasoning models are built.

## Key Challenges
- **Exposure bias / distribution mismatch.** Offline KD trains on teacher outputs but the student decodes its own — degrades quality (fixed by on-policy KD).
- **Capacity gap.** A very small student can't represent a much larger teacher's function; too-large a gap hurts transfer (sometimes a mid-size "teacher assistant" helps).
- **Tokenizer/vocabulary mismatch.** White-box logit matching requires shared tokenizers; cross-tokenizer distillation is hard.
- **Inheriting flaws.** The student copies the teacher's biases, hallucinations, and errors; ceiling is the teacher.
- **Reverse-KL failure modes.** Mode-seeking can collapse diversity, and the teacher's signal weakens on long or off-distribution student prefixes.
- **Legal/ToS and access.** Distilling closed-API models may violate terms of service and raises IP questions; frontier labs now actively defend against it.

## Solutions & Current Best Practices
For LLMs, **sequence-level / data distillation** (train on high-quality teacher generations, filtered/verified) is the workhorse for cold-start; follow with **on-policy distillation** (student samples, teacher gives per-token reverse-KL reward) when teacher logits are available — it is far cheaper than RL for reaching a teacher's level and repairs capabilities lost during domain mid-training. Use **multi-teacher OPD** to merge domain-specialist checkpoints. For **reasoning**, distill **verified CoT traces** (keep correct ones — STaR/rejection sampling) from a strong teacher; this is now standard for small reasoning models. Use **temperature/soft-label** matching when logits are available and tokenizers match. Combine KD with **pruning** and **quantization** (small + quantized student) for deployment. Validate the student doesn't just memorize.

## Lab Perspectives
- **Google:** DistilBERT lineage, GKD (on-policy KD), Gemini **Flash** models distilled from larger Gemini; Gemma distillation.
- **DeepSeek:** **R1-Distill** series — distilling reasoning into Qwen/Llama bases; argued distillation > small-model RL for cost-efficiency.
- **Qwen / KAT-Coder team:** use multi-teacher on-policy distillation to merge domain experts.
- **NVIDIA:** Minitron-style **prune-then-distill** (Nemotron Nano 2: 12B → 9B).
- **Thinking Machines Lab:** popularized on-policy distillation as a cheap alternative to RL.
- **OpenAI/Anthropic:** ship distilled "mini/haiku/flash"-tier models (smaller, cheaper variants of flagships); OpenAI offers **distillation APIs**; Anthropic now routes detected distillation attempts on its top tier to a weaker model and restricts reasoning-trace exposure.
- **Meta/open community:** TinyLlama, MiniLM, and countless distilled open models.

## Latest Developments (2023–Nov 2026)
**Reasoning distillation** became the dominant way to make small reasoning models (R1-Distill, Qwen-distill), often outperforming direct small-model RL. **On-policy KD** (GKD, MiniLLM with reverse-KL) addressed exposure bias. **Distillation scaling laws** (2024–2025) characterized when distillation beats from-scratch training (it helps most when teacher is strong and student-compute is limited). Frontier labs productized **distillation pipelines** and small/fast model tiers as core offerings.

**Mid-2025 → Nov 2026 update:**
- **On-Policy Distillation (2025-10, Thinking Machines).** Student-sampled trajectories with a per-token reverse-KL teacher reward reach ~70% on AIME24 at roughly a tenth of the GPU-hours of RL, and recover instruction-following lost during domain mid-training.
- **OPD as a standard stage (2026).** Qwen3 and KAT-Coder-V2.5 merge domain experts via multi-teacher OPD ([arXiv:2607.05471](https://arxiv.org/abs/2607.05471)); OPD sits between SFT cold-start and (optional) RL.
- **Fixing reverse-KL failure modes (2026-03 → 2026-05).** Entropy-gated switching between forward and reverse KL (EOPD, [arXiv:2603.07079](https://arxiv.org/abs/2603.07079)), truncated top-$K$ reverse KL ([arXiv:2603.25562](https://arxiv.org/abs/2603.25562)), and corrections for teacher-signal decay on long prefixes ([arXiv:2605.30833](https://arxiv.org/abs/2605.30833)).
- **Distilling into faster decoders (2026-02).** Online self-distillation converts a pretrained NTP model into a standalone multi-token-prediction decoder, reporting >3× faster decoding on GSM8K at <5% accuracy loss ([arXiv:2602.06019](https://arxiv.org/abs/2602.06019)).
- **Prune + distill for hybrids (2025-08).** Nemotron Nano 2 (Mamba-2/Transformer hybrid) is compressed from 12B to 9B with Minitron pruning and distillation ([arXiv:2508.14444](https://arxiv.org/abs/2508.14444)).
- **Distillation defenses (2026).** Anthropic's Fable 5 classifiers detect distillation requests and route them to Opus 4.8; "preserved thinking" prevents API users from editing context to extract reasoning; Anthropic published on defending against distillation attacks (2026-10). Black-box distillation of frontier reasoning traces is getting harder.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why use soft labels / what is 'dark knowledge'?"** Teacher's full distribution encodes inter-class similarity the hard label omits; richer gradient signal.
- **"Sequence-level vs logit distillation for LLMs?"** Train on teacher generations (scalable, black-box) vs match per-token distributions (richer, needs logits + shared vocab).
- **"What is the distribution-mismatch problem and how does on-policy KD fix it?"** Student trains on teacher trajectories but decodes its own; GKD trains on student-generated sequences scored by the teacher.
- **"Why distill R1 into a small model instead of RL-ing it directly?"** Empirically cheaper and better — the small model inherits reasoning the teacher discovered via expensive RL.
- **"Why is on-policy distillation ~10× cheaper than RL to reach a teacher's level?"** RL gets ~1 scalar reward per episode; OPD gets a dense per-token signal on the student's own states — same on-policyness, far more bits per sample. Failure modes: mode-seeking collapse and weak teacher signal on off-distribution prefixes.

## Open Problems
The **capacity-gap** limit (how small a student can capture a given teacher) lacks clean theory. **Cross-tokenizer** white-box distillation, distilling **agentic/tool-use** competence, and whether students can ever **exceed** their teacher (vs being capped) are open. The interaction of distillation with RL (distill-then-RL vs RL-then-distill vs OPD replacing RL), multi-teacher merging without interference, and the long-term effect of anti-distillation defenses on the open ecosystem are actively studied.

## References
- Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
- Sanh, V. et al. (2019). *DistilBERT.* arXiv:1910.01108.
- Kim, Y., Rush, A. (2016). *Sequence-Level Knowledge Distillation.* arXiv:1606.07947.
- Agarwal, R. et al. (2023). *GKD (On-Policy Distillation).* arXiv:2306.13649.
- DeepSeek-AI (2025). *DeepSeek-R1 (distillation).* arXiv:2501.12948.
- Thinking Machines Lab (2025). *On-Policy Distillation.* Blog post.
- NVIDIA (2025). *NVIDIA Nemotron Nano 2.* arXiv:2508.14444.
