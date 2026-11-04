# What's New: Mid-2025 → November 2026 Digest

> **Last Updated:** 2026-11-04
> **Purpose:** A one-page briefing on what changed in LLM training and the frontier-lab landscape between mid-2025 and November 2026. Read this first if you studied the database before; every theme links to the topic file that covers it in depth.
> **Sourcing note:** Facts were checked against lab announcements, system cards, arXiv papers, and reputable press as of 2026-11-04. Vendor-reported benchmark numbers are labelled as such; items with weak sourcing are hedged ("reportedly") or omitted. Verify exact dates and numbers against primary sources before quoting them in an interview.

---

## 1. The frontier model landscape (as of Nov 2026)

| Lab | Current flagship(s) | What's notable |
|---|---|---|
| **OpenAI** | GPT-6 Astra (2026-09), GPT-6 Sol / Luna (2026-09-22), GPT-6.1 Sol (DevDay, 2026-09-29) | GPT-5 (2025-08) → 5.1 → 5.2 → 5.3-Codex → 5.4 (native computer use, ~1M context) → 5.5 → 5.6 Sol/Terra/Luna → GPT-6. Astra is the first OpenAI model rated **"Critical" for cyber** under the Preparedness Framework, first served via the **Daybreak** trusted-access program. gpt-oss (2025-08) was OpenAI's first open-weight LLM since GPT-2. |
| **Anthropic** | Claude Opus 5.5 / Sonnet 5.5 / Haiku 5.5 (Sep–Oct 2026); Claude Fable 5.1 / Mythos 5.1 (Sep 2026) | Claude 4.x through early 2026 (Opus 4.6: 1M context, adaptive thinking). **Claude Mythos Preview** (2026-04) was withheld from general release for its cyber-offense capability (Project Glasswing). From June 2026, the **same weights ship under two safeguard levels**: Fable (general availability, classifier routing) and Mythos (verified-access). RSP v3 (2026-02) added periodic Risk Reports and a public Frontier Safety Roadmap. |
| **Google DeepMind** | Gemini 3.1 Pro (2026-02), Gemini 3.5 Flash (I/O 2026), Gemini 3.6 Flash (2026-07) | Gemini 3 Pro + Deep Think (2025-11). Gemini 3.5 Pro was announced but delayed; Gemini 4 reported in post-training (late Sep 2026). Genie 3 world model (2025-08). Restricted **Gemini 3.5 Flash Cyber**. FSF v3.x added Tracked Capability Levels. |
| **Meta (MSL)** | Muse Spark (2026-04), Muse Wonder (2026-10) | Meta's frontier line moved to **closed weights** after Llama 4. |
| **DeepSeek** | V4-Pro (1.6T total / 49B active) and V4-Flash (284B / 13B), 2026-04-24; V4.1-Flash (2026-09) | V3.1 (UE8M0 FP8) → V3.2 with **DeepSeek Sparse Attention** → V4 with compressed sparse attention, Muon, FP4 QAT for experts, 1M context. MIT license. |
| **Qwen (Alibaba)** | Qwen3.8 (open weights, 2.4T total / ~95B active, Aug 2026) | Qwen3-Next introduced a Gated-DeltaNet hybrid; Qwen3.5/3.6 continued it. |
| **Moonshot (Kimi)** | Kimi K3 (2.8T total / ~104B active, Jul 2026); K3.1 Pro (Oct 2026) | K2 (1T, 2025-07) trained with **MuonClip** with no loss spikes; K3 mixes Kimi Delta Attention with MLA at 3:1. |
| **Mistral** | Mistral Large 4 (~1.05T / ~52B active, open weights 2026-10) | Large 3 (675B/41B, Apache 2.0, 2025-12); Medium 3.5 (128B dense, open). |

See [Lab Roadmaps](08_lab_roadmaps/06_lab_comparison_matrix.md) for the per-lab detail.

## 2. Ten themes that matter for training

1. **Hybrid and sparse attention went to production.** There are two recipes. One interleaves linear/delta-rule layers with full attention at about 3:1 (Qwen3-Next, Kimi Linear/K3). The other uses learned top-k sparse attention over a compressed KV cache (DeepSeek DSA → V4, adopted by GLM-5). MiniMax M2's return to full attention, citing multi-hop failures, is the standard counter-example. → [Attention](00_foundations/01_attention_mechanisms.md), [Diffusion & Hybrid LMs](09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)
2. **Open MoEs reached the trillions.** In about 15 months, open models went from 1T (Kimi K2) to 2.8T (Kimi K3) total parameters. Active parameters stayed at about 13–104B, so sparsity keeps rising (e.g., 16 of 896 experts). → [MoE](06_efficiency/05_mixture_of_experts.md)
3. **Precision moved to 4 bits.** FP8 is the baseline. NVFP4 pretraining matched FP8 over 10T+ tokens (NVIDIA), Nemotron 3 was pretrained in NVFP4, and DeepSeek V4 used FP4 QAT for its experts. Hardware is now marketed in FP4 FLOPs. → [Mixed Precision](02_pretraining/04_mixed_precision_and_quantization.md), [Quantization](06_efficiency/02_quantization_techniques.md)
4. **Muon-family optimizers went mainstream, with honest caveats.** Muon/MuonClip (QK-clip) is used in trillion-scale runs. Fairly-tuned benchmarks cap speedups over AdamW at about 1.4× and find they shrink with scale. → [Optimizers](02_pretraining/03_optimizer_choices.md)
5. **RL scaling became a science.** Several advances:
   - RL compute curves are sigmoidal and predictable from early runs (ScaleRL).
   - Objectives converged: token-level normalization, decoupled clipping (DAPO), sequence-level ratios (GSPO), and IS-weight clipping (CISPO).
   - Training–inference numerical mismatch is now treated as an algorithmic problem.
   - Async RL infrastructure is standard.

   → [GRPO & RLVR](03_posttraining/06_GRPO_and_RLVR.md)
6. **On-policy distillation sits between SFT and RL.** The student samples and the teacher scores every token, giving dense reward at a fraction of RL cost. LoRA works for RL even at very low rank. → [Distillation](06_efficiency/03_knowledge_distillation.md), [LoRA](06_efficiency/01_LoRA_and_variants.md)
7. **Agents are the main capability axis.** Signs of this shift:
   - Long-horizon coding and computer-use agents (hours-long autonomous sessions, parallel subagents, context compaction).
   - Effort levels and adaptive thinking exposed as product controls.
   - METR 50%-time-horizons doubling roughly every ~4 months since 2023.
   - Interop standards: MCP, A2A, Agent Skills.

   → [Agentic Coding & Computer Use](04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)
8. **Benchmarks saturate faster than ever.** SWE-bench Verified was retired by OpenAI (flawed tests plus contamination) in favor of SWE-bench Pro. HLE passed 60% with tools. ARC-AGI-3, an interactive benchmark, launched with frontier models below 1%. Economic Elo-style evals (GDPval-AA) and **eval awareness** (models noticing they are being tested) are now central. → [Evaluation](05_alignment_and_safety/05_evaluation_and_benchmarks.md)
9. **Safety moved into the training and release pipeline.** Several changes:
   - Reward hacking in production RL can generalize to broad misalignment; **inoculation prompting** is a cheap mitigation.
   - CoT monitorability is treated as a fragile safety property.
   - **Capability shaping** is now deliberate: Anthropic did not train Opus 5 on cyber tasks.
   - **Tiered release** of one set of weights with classifier routing (Fable/Mythos, Daybreak, Flash Cyber).
   - Safety frameworks shifted from thresholds to recurring public reporting.

   → [Frontier Safety Frameworks & Governance](05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md)
10. **Compute is bought in gigawatts across vendors, and governments intervene.** Examples:
    - OpenAI: about 26 GW across NVIDIA, AMD and Broadcom.
    - Anthropic: TPU (up to 1M chips), Trainium and NVIDIA.
    - Custom silicon is splitting into training and inference parts (TPU 8t/8i, Rubin CPX).
    - US export controls briefly pulled a frontier model from service (June 2026).
    - The EU AI Act's GPAI obligations apply, and California SB 53 is in force.

    → [Distributed Training](02_pretraining/01_distributed_training.md), [Lab Comparison](08_lab_roadmaps/06_lab_comparison_matrix.md)

## 3. New files in this update
- [`04_reasoning_and_agents/06_agentic_coding_and_computer_use.md`](04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)
- [`05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md`](05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md)
- [`09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md`](09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)

## 4. Fastest interview refresh (one hour)
1. Derive the gated delta-rule update and explain why hybrids beat pure linear attention ([Attention](00_foundations/01_attention_mechanisms.md)).
2. Compare GRPO → DAPO → GSPO → CISPO and explain how training–inference mismatch breaks "on-policy" ([GRPO & RLVR](03_posttraining/06_GRPO_and_RLVR.md)).
3. Explain NVFP4 vs MXFP4 and the tricks that make FP4 pretraining stable ([Mixed Precision](02_pretraining/04_mixed_precision_and_quantization.md)).
4. Explain why reward hacking → emergent misalignment, and why inoculation prompting works ([RLHF](03_posttraining/03_RLHF.md), [Alignment](05_alignment_and_safety/00_alignment_overview.md)).
5. Explain METR time horizons, eval awareness, and why SWE-bench Verified was retired ([Evaluation](05_alignment_and_safety/05_evaluation_and_benchmarks.md)).
6. Know your target lab's 2026 position ([Lab Comparison Matrix](08_lab_roadmaps/06_lab_comparison_matrix.md)).
