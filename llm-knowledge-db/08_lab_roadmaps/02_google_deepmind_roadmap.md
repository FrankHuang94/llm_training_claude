# Google DeepMind — Lab Roadmap

> **Last Updated:** 2026-11-04
> **Related Files:** [Scaling Laws](../00_foundations/04_scaling_laws.md) · [Multimodal Pretraining](../07_multimodal/01_multimodal_pretraining.md) · [World Models](../09_emerging_frontiers/02_world_models.md) · [Lab Comparison Matrix](06_lab_comparison_matrix.md)
> **Key Papers:** Hoffmann et al. 2022 Chinchilla ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) · Chowdhery et al. 2022 PaLM ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) · Gemini Team 2023 ([arXiv:2312.11805](https://arxiv.org/abs/2312.11805)) · Gemini 1.5 2024 (long context) ([arXiv:2403.05530](https://arxiv.org/abs/2403.05530))

## Overview
Google DeepMind (GDM) — formed by merging Google Brain and DeepMind in 2023 — combines the inventors of the **Transformer** (Vaswani et al. were at Google) with DeepMind's deep-RL and scientific-AI heritage (AlphaGo, AlphaFold). It is arguably the broadest AI research organization, spanning frontier LLMs (**Gemini**), open models (**Gemma**), formal reasoning (**AlphaProof/AlphaGeometry**), world models (**Genie**), robotics (**Gemini Robotics**), science (AlphaFold), and foundational research (Chinchilla scaling laws). Its differentiators are **native multimodality**, **TPU vertical integration** (now also sold externally at gigawatt scale), aggressive **price-performance in the Flash tier**, and a strong scientific-reasoning pedigree. In Aug 2026 Demis Hassabis became Chair of GDM and Alphabet Chief Scientist, with **Koray Kavukcuoglu** running DeepMind as SVP reporting to Sundar Pichai.

## Model Lineage & Key Innovations
- **Foundational:** the **Transformer** (2017); **Chinchilla** (2022) — compute-optimal scaling that corrected Kaplan; **PaLM** (540B, parallel layers, MQA).
- **Gemini 1.0 (2023):** **natively multimodal** (text/image/audio/video from pretraining) flagship; Ultra/Pro/Nano tiers.
- **Gemini 1.5 (2024):** **MoE** + breakthrough **long context (1M→2M tokens)** with strong needle-in-haystack recall; Flash (distilled, fast).
- **Gemini 2.0/2.5 (2024–2025):** agentic (Project Mariner), **native tool use**, thinking models (2.5 Pro/Flash) with strong math/coding/science.
- **Gemini 3 (2025-11-18):** Gemini 3 Pro (HLE no-tools 37.5%, SWE-bench Verified 76.2%) plus **Gemini 3 Deep Think** (HLE 41%, ARC-AGI-2 45.1%) and **Google Antigravity**, an agent-first IDE that also hosts Claude and gpt-oss agents. Its launch triggered OpenAI's "code red."
- **Gemini 3.1 Pro (2026-02-19):** ARC-AGI-2 77.1% verified (3 Pro: 31.1%), GPQA Diamond 94.3%.
- **Gemini 3.5 / 3.6 (2026-05 → 07):** at I/O, **3.5 Flash** became the default in the Gemini app and AI Mode, claimed to beat 3.1 Pro on Terminal-Bench 2.1, GDPval-AA and MCP Atlas; **3.5 Pro was announced but delayed**. July added **3.6 Flash** (~17% fewer output tokens than 3.5 Flash, secondary), **3.5 Flash-Lite**, and the restricted **3.5 Flash Cyber** (vulnerability-fixing inside the CodeMender agent, governments/trusted partners only).
- **Gemini 4:** Pichai said in Jul 2026 "we are now training Gemini 4"; Kavukcuoglu said in late Sep it was **in post-training**, targeted well before end-2026 (no specs disclosed).
- **Gemma (open, 2024+):** Gemma 1–3, Gemma Scope (open SAEs); **Gemma 4** (2026-04-02) — first under **Apache 2.0**: E2B/E4B edge models (per-layer embeddings; text/image/audio), a **26B-A4B MoE** and a **31B dense** model (256K context), built from Gemini 3 research.
- **Specialized reasoning & science:** **AlphaProof/AlphaGeometry** (IMO 2024 silver-level formal math), **AlphaCode**, AlphaFold; an "AI co-mathematician" agent reportedly reached 48% on FrontierMath Tier 4 (2026-05, secondary).
- **World models, robotics & generation:** **Genie 3** (2025-08: real-time 720p/24fps interactive worlds, consistent for minutes, pitched as an agent training ground); **Gemini Robotics 2 / ER 2** (2026-07: whole-body humanoid VLA plus an embodied-reasoning planner with multi-robot task splitting); **Veo**, **Imagen**, **Lyria**.

## Technical Differentiation
- **Native multimodality** from pretraining (vs bolt-on encoders) — text/image/audio/video unified, now extended to "Omni" and robotics.
- **Long context** — the original 1–2M-token leader; by 2026 1M context is industry-standard (Opus 4.6, GPT-5.4, DeepSeek V4), so the edge shifted to *usable* recall and cost.
- **TPU vertical integration** — Ironwood (TPU v7: 192 GB HBM, ~4.6 PF FP8 per chip; GA 2026-03-31) and the 8th generation split into **TPU 8t (training)** and **TPU 8i (inference)**; JAX/Pathways software. TPUs are now a merchant product (Anthropic: up to 1M chips).
- **Flash-tier price-performance** — distillation-heavy Flash models repeatedly matching the previous Pro (3.5 Flash vs 3.1 Pro).
- **Scientific & formal reasoning** — AlphaProof/AlphaGeometry/AlphaFold; deep RL + search heritage; Deep Think parallel-reasoning mode.

## Published Research Focus
Scaling laws (Chinchilla, DoReMi data mixing), long context, multimodal architecture, RLHF/RLAIF, reasoning and search (AlphaProof), world models (Genie), robotics, interpretability, safety (Frontier Safety Framework), and a vast science portfolio. Notably, the GDM interpretability team published **negative results for SAEs** on downstream tasks (linear probes beat SAE features on OOD harmful-intent detection) and deprioritized fundamental SAE work in favor of pragmatic and model-diffing approaches. Gemini technical details are published less fully than Gemma's.

## Compute & Infrastructure
Massive **TPU** fleets with **JAX/Pathways/GSPMD** for planet-scale training — full hardware-software vertical integration and a structural cost advantage over GPU-renting competitors. Since 2025 Google also **sells TPU capacity externally at gigawatt scale** (Anthropic: up to 1M TPUs, >1 GW in 2026), turning the TPU into a strategic cloud product. Generation cadence: Trillium (v6) → Ironwood (v7, GA Mar 2026) → TPU 8t/8i (announced Apr 2026; availability late 2026 or 2027, sources conflict). Deep integration with Search (AI Mode), Workspace, Android and Cloud.

## Stated Strategic Direction
AGI via broad capability + scientific impact; **multimodal + agentic** Gemini as the unifying platform (Gemini Spark persistent agent, Antigravity, CodeMender); **world models** (Genie) and **robotics** as paths to embodied agents; **open Gemma** under Apache 2.0 to seed an ecosystem; AI for science; and safety via **Frontier Safety Framework v3** (harmful-manipulation CCL, shutdown-resistance protocols) and restricted release of cyber-capable models. Leadership was consolidated under Kavukcuoglu (Aug 2026) to tighten the research-to-product loop.

## Known Strengths & Weaknesses
- **Strengths:** native multimodality, TPU cost/scale advantage (and TPU revenue), unmatched breadth (LLMs + science + RL + world models + robotics + generation), enormous distribution (Search/Android/Cloud), strong reasoning benchmarks (3.1 Pro ARC-AGI-2), and Flash-tier price-performance.
- **Weaknesses:** uneven shipping (3.5 Pro announced but delayed), org complexity and a 2026 leadership reshuffle, perceived lag behind Anthropic/OpenAI on agentic coding at times, and less complete technical disclosure for Gemini than for Gemma.

## Hiring Focus
Large-scale training on TPU/JAX, multimodal and long-context architecture, RL and search (reasoning, agents), world models and robotics, data/scaling research, interpretability and safety (Frontier Safety), and AI-for-science. JDs emphasize JAX/TPU, RL, research depth, and systems at planetary scale.

## Latest Developments (2023–Nov 2026)
2023–mid-2025 delivered Gemini 1.0 → 2.5 (native multimodality, 1–2M context, thinking models), Gemma 3, and AlphaProof. Since then GDM has moved to a fast Pro/Flash cadence, opened Gemma fully, and pushed world models and robotics.

**Mid-2025 → Nov 2026 update:**
- **2025-08 — Genie 3:** real-time, promptable interactive world model (720p/24fps, minutes of consistency) explicitly positioned as a scalable environment generator for agent RL (see [World Models](../09_emerging_frontiers/02_world_models.md)).
- **2025-09 — FSF v3.0:** adds a harmful-manipulation CCL and misalignment protocols for shutdown/modification resistance; safety-case reviews extend to large internal deployments. A reported **v3.1** (2026-04; version label disputed) adds early-warning **Tracked Capability Levels** below CCLs and raises security for CBRN/cyber/manipulation CCLs.
- **2025-11 → 2026-02 — Gemini 3 → 3.1 Pro:** ARC-AGI-2 from 31.1% to 77.1% in one point release; a jump this size within a version line suggests post-training/RL rather than a new pretraining run (GDM has not disclosed the recipe), and illustrates how fast reasoning benchmarks saturate.
- **2026-03 → 04 — Ironwood GA and TPU 8t/8i:** first split of the TPU line into training- and inference-specialized chips (claimed 2.7× training and up to 80% better inference price-performance vs Ironwood), mirroring industry prefill/decode disaggregation.
- **2026-04 — Gemma 4 (Apache 2.0):** a 26B-A4B MoE and per-layer-embedding edge models bring Gemini-3-era recipes to permissively licensed open weights.
- **2026-05 → 07 — Flash-first releases and restricted cyber model:** 3.5 Flash becomes the default consumer model; 3.5 Flash Cyber is restricted to vetted users — GDM joins Anthropic/OpenAI in tiered access for dual-use capability.
- **2026-07 → 09 — Gemini 4 in training/post-training; Gemini Robotics 2:** embodied VLA + reasoning planner with multi-robot coordination.

## Interview Angles
> 💡 **What labs actually ask (GDM-flavored):**
- **"Explain Chinchilla scaling and why it corrected Kaplan."** (See [Scaling Laws](../00_foundations/04_scaling_laws.md).)
- **"How would you build a 1M-token context model — and make it cheap?"** Positional extension + sparse/hybrid attention + KV management (see [Context Extension](../06_efficiency/06_context_window_extension.md)).
- **"Native vs late-fusion multimodality — tradeoffs?"** (See [Multimodal Pretraining](../07_multimodal/01_multimodal_pretraining.md).)
- **"How does AlphaProof achieve formal reasoning?"** RL + search over formal (Lean) proofs with verifiable rewards.
- **"Why split accelerators into training and inference chips?"** Different compute/bandwidth/memory ratios for training, prefill and decode; cost per token.
- **"How would you use a world model like Genie 3 to train agents?"** Environment generation, sim-to-real gaps, consistency horizon, reward specification.

## Open Problems / Watch Items
Shipping Gemini 4 and the delayed 3.5 Pro, converting research breadth into agentic-coding leadership, the effect of the 2026 leadership change, scaling long context to *usable* recall, turning Genie and Gemini Robotics into real-world agent gains, and leveraging TPU and scientific advantages (including external TPU sales) into a durable frontier lead.

## References
- Hoffmann, J. et al. (2022). *Chinchilla.* arXiv:2203.15556.
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Gemini Team (2023). *Gemini.* arXiv:2312.11805.
- Gemini Team (2024). *Gemini 1.5.* arXiv:2403.05530.
- AlphaGeometry: Trinh, T. et al. (2024). *Nature.*
- Google DeepMind (2025). *Frontier Safety Framework v3.0.*
- Google DeepMind (2025). *Genie 3: A New Frontier for World Models.*
