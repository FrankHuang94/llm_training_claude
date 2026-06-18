# DeepSeek — Lab Roadmap

> **Last Updated:** 2026-06-18
> **Related Files:** [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md) · [Mixed Precision & Low-Precision Training](../02_pretraining/04_mixed_precision_and_quantization.md)
> **Key Papers:** DeepSeek-AI 2024 DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · DeepSeek-AI 2024 DeepSeek-V2 (MLA) ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) · Shao et al. 2024 DeepSeekMath/GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))

## Overview
DeepSeek, a Chinese lab (spun out of the quant fund High-Flyer), became the field's biggest disruptor in late 2024–2025 by releasing **open-weights** models that matched or approached frontier (GPT-4o / o1) quality at a **fraction of the training cost** — and publishing detailed methods. **DeepSeek-V3** (671B MoE, ~$5.6M reported training cost) and **DeepSeek-R1** (open reasoning model rivaling o1) triggered a market shock (the "DeepSeek moment," Jan 2025) and demonstrated that algorithmic and systems efficiency — not just raw compute — can reach the frontier, even under **export-control hardware constraints** (H800 GPUs with limited interconnect).

## Model Lineage & Key Innovations
- **DeepSeek LLM / Coder / Math (2023–2024):** strong open base/code/math models; **DeepSeekMath** introduced **GRPO** (see [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)).
- **DeepSeek-V2 (2024):** introduced **Multi-head Latent Attention (MLA)** — low-rank KV compression for big KV-cache savings — plus **DeepSeekMoE** (fine-grained + shared experts).
- **DeepSeek-V3 (2024):** 671B-param **MoE** (37B active), trained on ~14.8T tokens in **FP8** with **DualPipe** (near-zero-bubble pipeline) and **auxiliary-loss-free load balancing** and **multi-token prediction (MTP)** — a landmark in efficient large-scale training.
- **DeepSeek-R1 / R1-Zero (2025):** open **reasoning** models via **pure RL (GRPO) with verifiable rewards**; R1-Zero showed reasoning *emerges from RL with no SFT*; R1 distilled into smaller dense models (R1-Distill).
- **Subsequent (V3.1/R1-updates, 2025):** continued efficiency and reasoning improvements; NSA (native sparse attention) research.

## Technical Differentiation
- **MLA** — a distinctive KV-efficient attention (vs GQA), enabling cheap long context.
- **DeepSeekMoE** — fine-grained + shared experts, **auxiliary-loss-free** balancing.
- **GRPO + RLVR** — critic-free reasoning RL; the open blueprint for o1-class reasoning.
- **Extreme training efficiency** — FP8 training, DualPipe, communication-aware MoE on constrained (H800) hardware.
- **Radical openness** — open weights + unusually detailed technical reports.

## Published Research Focus
Efficient architectures (MLA, MoE), low-precision training (FP8), systems (DualPipe, communication overlap), reasoning RL (GRPO, RLVR, R1), distillation of reasoning, multi-token prediction, and native sparse attention. The reports are detailed enough to be widely reproduced — a deliberate contrast to closed labs.

## Compute & Infrastructure
Operates under **US export controls**, using **NVIDIA H800** (and earlier A100) GPUs with reduced interconnect bandwidth — which *forced* the systems innovations (FP8, DualPipe, MLA, comm-overlap) that became its hallmark. Backed by High-Flyer's substantial GPU stockpile; emphasizes squeezing maximal efficiency from constrained hardware.

## Stated Strategic Direction
Push **open**, **efficient** frontier models; pursue AGI via algorithmic/systems efficiency rather than brute-force compute; share methods openly. Positioned as proof that the frontier is not solely a function of compute budget — a narrative with major geopolitical and economic implications for the "compute moat" thesis.

## Known Strengths & Weaknesses
- **Strengths:** efficiency leadership (cost/quality), open reasoning models, architectural innovation (MLA, MoE, GRPO), reproducible transparency, strong math/code.
- **Weaknesses:** hardware access constrained by export controls (scaling ceiling risk), less multimodal/product breadth, smaller safety-research public profile, geopolitical/regulatory headwinds (bans/scrutiny in some jurisdictions), and English-ecosystem trust questions.

## Stated Hiring Focus
(Inferred from output and reports) systems/efficiency engineering (low-precision, parallelism, kernels), RL for reasoning (GRPO/RLVR), MoE and attention architecture, and data/math/code curation. Emphasis on doing more with constrained compute — a distinctive engineering culture.

## Interview Angles
> 💡 **What labs actually ask (DeepSeek-relevant):**
- **"Explain GRPO and how it differs from PPO."** Critic-free, group-relative normalized advantage (see [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)).
- **"What is MLA and why does it help?"** Low-rank latent KV compression → small KV cache with MHA-like quality.
- **"How did DeepSeek-V3 train so cheaply?"** FP8 + DualPipe + aux-loss-free MoE + MTP + comm-overlap on H800s.
- **"What did R1-Zero demonstrate?"** Reasoning emerges from pure RL with verifiable rewards, no SFT.

## Open Problems / Watch Items
Whether DeepSeek can keep pace at the next scale under tightening export controls, extending efficiency wins to multimodal/agents, building a safety-research profile, and whether its "efficiency beats compute" thesis holds as frontier compute keeps growing.

## References
- DeepSeek-AI (2024). *DeepSeek-V3.* arXiv:2412.19437.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- DeepSeek-AI (2024). *DeepSeek-V2 (MLA).* arXiv:2405.04434.
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- Yuan, J. et al. (2025). *Native Sparse Attention.* arXiv:2502.11089.
