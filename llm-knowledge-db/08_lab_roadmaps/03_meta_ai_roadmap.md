# Meta AI — Lab Roadmap

> **Last Updated:** 2026-06-18
> **Related Files:** [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md) · [Lab Comparison Matrix](06_lab_comparison_matrix.md)
> **Key Papers:** Touvron et al. 2023 LLaMA ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) · Touvron et al. 2023 Llama 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · Grattafiori et al. 2024 Llama 3 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) · Zhang et al. 2022 OPT ([arXiv:2205.01068](https://arxiv.org/abs/2205.01068))

## Overview
Meta AI (FAIR + the GenAI/applied org) is the dominant force in **open-weights** frontier models. The **LLaMA** series democratized access to strong base models and effectively created the open LLM ecosystem — its architecture choices (RMSNorm + RoPE + SwiGLU) became the community standard, and its open releases seeded thousands of derivatives. Meta's strategy bets that **openness** drives ecosystem dominance, talent attraction, safety-via-scrutiny, and commoditization of the model layer (benefiting Meta's product/ads business). Meta also stewards **PyTorch**, the field's primary framework.

## Model Lineage & Key Innovations
- **OPT (2022):** an early open replication of GPT-3 with a candid training **logbook** (instabilities) — a transparency landmark.
- **LLaMA 1 (2023):** efficient, **overtrained** smaller models (the "Chinchilla-plus" inference-optimal paradigm); established the open recipe (RMSNorm/RoPE/SwiGLU). Leaked/released weights ignited the open ecosystem (Alpaca, Vicuna, etc.).
- **Llama 2 (2023):** open **with a commercial license**; documented **RLHF + rejection sampling** pipeline, Ghost Attention, safety (Llama Guard).
- **Llama 3 / 3.1 / 3.2 / 3.3 (2024):** scaled to **405B**, ~15T tokens, 128K context, strong multilingual/code/math; 3.2 added **vision** (cross-attention adapters) and small on-device models; detailed, openly-published technical report.
- **Llama 4 (2025): Scout / Maverick / Behemoth** — Meta's move to **MoE** with very long context (Scout 10M tokens) and native multimodality.
- **Other:** **Code Llama**, **SAM** (segmentation), **ImageBind**, **Chameleon** (early-fusion), **Movie Gen**, **Massively Multilingual Speech**, and extensive FAIR research.

## Technical Differentiation
- **Open weights at the frontier** — the defining strategy; sets the community baseline architecture.
- **Overtraining for inference efficiency** — small, strong, cheap-to-serve models.
- **PyTorch + FSDP/TorchTitan** ecosystem leadership.
- **Open, detailed technical reports** (Llama 3 report is a reference document) and open safety tooling (**Purple Llama / Llama Guard**).

## Published Research Focus
Open foundation models and detailed reports, efficient architectures, multilingual and speech, multimodal (Chameleon, ImageBind, SAM), data curation/dedup at scale, training stability (OPT), tokenizer-free (Byte Latent Transformer), and a broad FAIR basic-research portfolio. Strong culture of open publication and reproducible artifacts.

## Compute & Infrastructure
Very large GPU fleets (Llama 3 trained on **16k H100s**, with detailed fault-tolerance disclosures; building toward **~350k+ H100-equivalent** capacity). Drives **PyTorch**, FSDP, and open training stacks used field-wide. Vertical integration with Meta's products (Instagram/WhatsApp/Facebook) for distribution and on-device deployment (Llama on devices, Ray-Ban Meta).

## Stated Strategic Direction
"**Open source AI is the path forward**" (Zuckerberg) — release strong open models to commoditize the layer, build an ecosystem, and integrate AI across Meta's apps and hardware (smart glasses, assistants). Pursuit of "superintelligence" (a reorganized **Superintelligence Labs**, aggressive 2025 hiring) alongside open releases. Some signals of selective openness for the most capable models.

## Known Strengths & Weaknesses
- **Strengths:** open-ecosystem gravity, architecture standard-setting, PyTorch stewardship, inference-efficient models, massive distribution and compute, transparency.
- **Weaknesses:** historically behind on **reasoning** (no o1/R1-class open reasoner at launch), Llama 4 reception mixed vs DeepSeek/Qwen, talent churn and reorg turbulence (2025), and tension between openness and frontier-safety/competitive concerns.

## Hiring Focus
Large-scale pretraining and distributed systems (PyTorch/FSDP), data curation, multimodal and speech, efficient architectures (MoE, long context), reasoning/post-training (catching up), and on-device/efficiency. JDs emphasize PyTorch, scaling, open research, and systems engineering.

## Interview Angles
> 💡 **What labs actually ask (Meta-flavored):**
- **"Why overtrain small models past Chinchilla-optimal?"** Inference cost dominates lifetime cost (see [Scaling Laws](../00_foundations/04_scaling_laws.md)).
- **"Describe the Llama 2/3 post-training pipeline."** SFT + rejection sampling + RLHF (PPO)/DPO (see [Post-Training](../03_posttraining/00_posttraining_overview.md)).
- **"How does Llama handle 128K context and faults at 16k GPUs?"** RoPE scaling + checkpointing/fault tolerance (see [Checkpointing](../02_pretraining/06_checkpoint_and_resumption.md)).
- **"Trade-offs of open-weights releases?"** Ecosystem/safety-scrutiny vs misuse/competitive concerns.

## Open Problems / Watch Items
Closing the **reasoning** gap with an open o1/R1-class model, the competitiveness of Llama 4 MoE vs DeepSeek/Qwen, whether openness remains viable at the very frontier, and integrating the Superintelligence Labs reorganization into shipped models.

## References
- Touvron, H. et al. (2023). *LLaMA.* arXiv:2302.13971.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Zhang, S. et al. (2022). *OPT.* arXiv:2205.01068.
- Pagnoni, A. et al. (2024). *Byte Latent Transformer.* arXiv:2412.09871.
