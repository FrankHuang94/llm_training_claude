# Meta AI — Lab Roadmap

> **Last Updated:** 2026-11-04
> **Related Files:** [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md) · [Lab Comparison Matrix](06_lab_comparison_matrix.md)
> **Key Papers:** Touvron et al. 2023 LLaMA ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) · Touvron et al. 2023 Llama 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) · Grattafiori et al. 2024 Llama 3 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) · Zhang et al. 2022 OPT ([arXiv:2205.01068](https://arxiv.org/abs/2205.01068))

## Overview
Meta AI (FAIR + the applied GenAI org, reorganized in mid-2025 into **Meta Superintelligence Labs, MSL**) was, through Llama 4, the dominant force in **open-weights** frontier models. The **LLaMA** series democratized access to strong base models and effectively created the open LLM ecosystem — its architecture choices (RMSNorm + RoPE + SwiGLU) became the community standard and seeded thousands of derivatives. Meta's original bet was that **openness** drives ecosystem dominance, talent attraction, safety-via-scrutiny, and commoditization of the model layer. In **April 2026 that bet changed at the frontier**: MSL's first model, **Muse Spark**, shipped with **closed weights**, followed by the closed **Muse Wonder** frontier models (Oct 2026). Meta says future Muse models *may* be opened, but the open-weight frontier is now led by Chinese labs, gpt-oss, Gemma and Mistral. Meta still stewards **PyTorch**, the field's primary framework.

## Model Lineage & Key Innovations
- **OPT (2022):** an early open replication of GPT-3 with a candid training **logbook** (instabilities) — a transparency landmark.
- **LLaMA 1 (2023):** efficient, **overtrained** smaller models (the inference-optimal paradigm); established the open recipe (RMSNorm/RoPE/SwiGLU). Leaked/released weights ignited the open ecosystem (Alpaca, Vicuna, etc.).
- **Llama 2 (2023):** open **with a commercial license**; documented **RLHF + rejection sampling** pipeline, Ghost Attention, safety (Llama Guard).
- **Llama 3 / 3.1 / 3.2 / 3.3 (2024):** scaled to **405B**, ~15T tokens, 128K context, strong multilingual/code/math; 3.2 added **vision** (cross-attention adapters) and small on-device models; detailed, openly published technical report.
- **Llama 4 (2025-04): Scout / Maverick / Behemoth** — Meta's move to **MoE** (17B active; Scout 16 experts with a 10M-token context, Maverick 128 experts) with native early-fusion multimodality; Behemoth was previewed but not released at launch. Reception was mixed relative to DeepSeek/Qwen, and the LMArena entry was an "experimental" variant different from the released model after ~27 private variants were tested.
- **Muse Spark (2026-04-08):** first MSL model and first of the **Muse** family; **closed weights** with private-preview API access for partners. Meta claims Llama-4-Maverick-level reasoning or better at far less compute, strongest in multimodal understanding and health.
- **Muse Wonder / Muse Wonder Full (2026-10-12):** closed frontier models for agentic work, with "**Active Memory**" and "**Active Focus**" mechanisms for error recovery; reported 1M context and $4/$20 pricing (secondary). Meta claims they beat OpenAI and Anthropic models; independent arenas place them near but below Claude Opus 5.5.
- **Other:** **Code Llama**, **SAM** (segmentation), **ImageBind**, **Chameleon** (early-fusion), **V-JEPA** world models, **Movie Gen**, **Massively Multilingual Speech**, and extensive FAIR research. Smaller open Muse variants have been reported but are **unverified**.

## Technical Differentiation
- **Open weights at the (former) frontier** — Llama set the community baseline architecture; the Muse line marks a deliberate split between closed frontier models and (possibly) open smaller ones.
- **Overtraining for inference efficiency** — small, strong, cheap-to-serve models; Muse Spark's pitch is again compute efficiency.
- **PyTorch + FSDP/TorchTitan** ecosystem leadership.
- **Open, detailed technical reports** through Llama 3, and open safety tooling (**Purple Llama / Llama Guard**); Muse releases have so far not come with comparable reports.

## Published Research Focus
Open foundation models and detailed reports, efficient architectures, multilingual and speech, multimodal (Chameleon, ImageBind, SAM), self-supervised world models (JEPA line), data curation/dedup at scale, training stability (OPT), tokenizer-free modeling (Byte Latent Transformer), and a broad FAIR basic-research portfolio. Meta researchers co-signed the 2025 cross-lab position paper on **CoT monitorability** (arXiv:2507.11473). The center of gravity has shifted from FAIR's open publication culture to MSL's product-focused, less public frontier work.

## Compute & Infrastructure
Very large GPU fleets (Llama 3 trained on **16k H100s** with detailed fault-tolerance disclosures; ~350k+ H100-equivalents targeted by end-2024), now contracted at **gigawatt** scale: an expanded NVIDIA deal and, days later, **up to 6 GW of AMD GPUs** (2026-02-24). Meta's Open Rack Wide spec underpins AMD's **Helios** MI450 rack design. Gigawatt campuses (reportedly *Prometheus* and *Hyperion*) are planned but their status is not verified here. Drives **PyTorch**, FSDP, and open training stacks used field-wide; vertical integration with Instagram/WhatsApp/Facebook and devices (Ray-Ban Meta).

## Stated Strategic Direction
The 2024 line was "**open source AI is the path forward**" (Zuckerberg). Since mid-2025 the stated goal is **"superintelligence"** via MSL, backed by aggressive hiring and gigawatt compute, with openness now **selective**: frontier Muse models closed, possibly open smaller models later. Meta also declined to sign the EU GPAI Code of Practice (2025), consistent with its skepticism of heavy frontier regulation. AI is integrated across Meta's apps, ads and hardware (smart glasses, assistants).

## Known Strengths & Weaknesses
- **Strengths:** massive distribution and compute, PyTorch stewardship, architecture standard-setting legacy, multimodal and health strengths (Muse Spark), talent density after 2025 hiring, and a fast-improving closed frontier line (Muse Wonder near the top in arenas).
- **Weaknesses:** Llama 4's mixed reception and benchmark-presentation controversy, repeated reorganizations, loss of open-ecosystem leadership to Qwen/DeepSeek/Kimi, reduced research transparency after the closed pivot, and a still-unproven position against OpenAI/Anthropic/Google on agentic coding.

## Hiring Focus
Large-scale pretraining and distributed systems (PyTorch/FSDP), RL/post-training for reasoning and agents, multimodal and speech, memory/long-context architecture (Active Memory), efficient architectures (MoE), and on-device/efficiency. JDs emphasize PyTorch, scaling, and systems engineering; MSL roles emphasize frontier-model experience.

## Latest Developments (2023–Nov 2026)
2023–mid-2025 took Meta from Llama 2 to Llama 3's openly documented 405B model and Llama 4's MoE pivot, followed by the formation of MSL. Since then Meta has restarted its frontier line as closed models.

**Mid-2025 → Nov 2026 update:**
- **2025 — Llama 4 aftermath:** MoE + 10M context did not translate into leadership, and the arena-variant episode made "which checkpoint was evaluated?" a standard eval-hygiene question (see [Evaluation](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)).
- **2025 (mid) — MSL reorganization:** frontier work consolidated under Meta Superintelligence Labs with large-scale hiring; FAIR's open-publication model de-emphasized.
- **2026-02 — Gigawatt multi-vendor compute:** up to 6 GW of AMD GPUs on top of an expanded NVIDIA deal — Meta joins OpenAI/Anthropic in contracting compute in gigawatts across vendors.
- **2026-04 — Muse Spark (closed):** the strategic break with open frontier weights; Meta emphasizes compute efficiency (Maverick-level reasoning with far less compute) and multimodal/health strengths.
- **2026-10 — Muse Wonder:** agent-oriented "Active Memory/Active Focus" for error recovery and a reported 1M context; arena placement near but below Opus 5.5 suggests Meta is back near the frontier but not leading it.

## Interview Angles
> 💡 **What labs actually ask (Meta-flavored):**
- **"Why overtrain small models past Chinchilla-optimal?"** Inference cost dominates lifetime cost (see [Scaling Laws](../00_foundations/04_scaling_laws.md)).
- **"Describe the Llama 2/3 post-training pipeline."** SFT + rejection sampling + RLHF (PPO)/DPO (see [Post-Training](../03_posttraining/00_posttraining_overview.md)).
- **"How does Llama handle 128K context and faults at 16k GPUs?"** RoPE scaling + checkpointing/fault tolerance (see [Checkpointing](../02_pretraining/06_checkpoint_and_resumption.md)).
- **"Trade-offs of open-weights releases — and why might a lab stop releasing its frontier?"** Ecosystem/safety-scrutiny vs misuse, distillation by competitors, and loss of differentiation.
- **"How would you design agent memory for error recovery?"** External/learned memory, retrieval vs compaction, credit assignment over long horizons (cf. Muse Wonder's Active Memory).

## Open Problems / Watch Items
Whether Muse Wonder/Full can reach the very top of agentic benchmarks, whether Meta opens any Muse weights (and at what scale), the durability of MSL after repeated reorganizations, returns on gigawatt-scale compute, regulatory friction in the EU, and whether the community standard architecture now follows Chinese open models rather than Llama.

## References
- Touvron, H. et al. (2023). *LLaMA.* arXiv:2302.13971.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Zhang, S. et al. (2022). *OPT.* arXiv:2205.01068.
- Pagnoni, A. et al. (2024). *Byte Latent Transformer.* arXiv:2412.09871.
- Korbak, T. et al. (2025). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* arXiv:2507.11473.
