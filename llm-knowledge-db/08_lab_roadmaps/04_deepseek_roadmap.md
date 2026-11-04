# DeepSeek — Lab Roadmap

> **Last Updated:** 2026-11-04
> **Related Files:** [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md) · [Mixed Precision & Low-Precision Training](../02_pretraining/04_mixed_precision_and_quantization.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md)
> **Key Papers:** DeepSeek-AI 2024 DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · DeepSeek-AI 2024 DeepSeek-V2 (MLA) ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) · Shao et al. 2024 DeepSeekMath/GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) · DeepSeek-AI 2025 DeepSeek-V3.2 (DSA) ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556))

## Overview
DeepSeek, a Chinese lab (spun out of the quant fund High-Flyer), became the field's biggest disruptor in late 2024–2025 by releasing **open-weights** models that matched or approached frontier (GPT-4o / o1) quality at a **fraction of the training cost** — and publishing detailed methods. **DeepSeek-V3** (671B MoE, ~$5.6M reported final-run cost) and **DeepSeek-R1** (open reasoning model rivaling o1) triggered the "DeepSeek moment" (Jan 2025). Since then DeepSeek has focused less on headline launches and more on **attention efficiency for million-token context**: hybrid thinking (V3.1), learned sparse attention (**DSA**, V3.2), and compressed sparse attention in **DeepSeek-V4** (Apr 2026; 1.6T total / 49B active, 1M context, MIT). It remains a reference point for "efficiency beats compute" under **export-control hardware constraints**, though by 2026 it shares the Chinese open-weight frontier with Qwen, Kimi and GLM.

## Model Lineage & Key Innovations
- **DeepSeek LLM / Coder / Math (2023–2024):** strong open base/code/math models; **DeepSeekMath** introduced **GRPO** (see [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)).
- **DeepSeek-V2 (2024):** **Multi-head Latent Attention (MLA)** — low-rank KV compression — plus **DeepSeekMoE** (fine-grained + shared experts).
- **DeepSeek-V3 (2024):** 671B-param **MoE** (37B active), ~14.8T tokens in **FP8**, **DualPipe**, **auxiliary-loss-free load balancing**, **multi-token prediction (MTP)**.
- **DeepSeek-R1 / R1-Zero (2025-01):** open **reasoning** via **pure RL (GRPO) with verifiable rewards**; R1-Zero showed reasoning *emerges from RL with no SFT*; distilled into smaller dense models.
- **NSA (2025-02):** Native Sparse Attention — hardware-aligned, trainable sparse attention with compressed/selected/sliding branches (ACL 2025 Best Paper).
- **DeepSeek-V3.1 (2025-08-21, MIT):** one **hybrid model with Think / Non-Think modes** (replacing the separate R1 line); **UE8M0 FP8** scale format (exponent-only, power-of-two block scales) "tailored for next-generation domestic chips."
- **DeepSeek-V3.2-Exp (2025-09-29) → V3.2 / V3.2-Speciale (2025-12-01):** **DeepSeek Sparse Attention (DSA)** — an FP8 "lightning indexer" scores past tokens and MLA attends only to the top-k; training config matched to V3.1-Terminus to isolate DSA's effect (quality roughly on par), API prices cut >50%. Speciale claimed gold-level results at IMO, CMO, ICPC World Finals and IOI 2025.
- **DeepSeekMath-V2 (2025-11):** a trained LLM proof verifier used as the reward for a generator that finds and fixes its own errors; gold-level IMO 2025 (5/6) and 118/120 Putnam 2024 (self-reported).
- **mHC (2025-12):** Manifold-Constrained Hyper-Connections — widened residual streams with mixing matrices projected onto doubly-stochastic matrices (Sinkhorn-Knopp), restoring stable signal propagation; reportedly used in V4.
- **DeepSeek-V4 Preview (2026-04-24, MIT):** **V4-Pro 1.6T total / 49B active** and **V4-Flash 284B / 13B active**, both 1M context with Thinking/Non-Thinking modes. V4-Flash went GA 2026-07-31 and V4-Pro ~mid-Aug.
- **DeepSeek-V4.1-Flash (2026-09-10, MIT):** **552B** MoE with **native vision** (text+image in), 1M context; replaces V4-Flash and the vision experiment, while V4-Pro is being phased out. **No R2** has been released.

## Technical Differentiation
- **MLA → DSA → CSA/HCA** — a continuous line of KV- and FLOP-reducing attention designs; DSA has spread beyond DeepSeek (GLM-5 adopted it).
- **DeepSeekMoE** — fine-grained + shared experts, **auxiliary-loss-free** balancing.
- **GRPO + RLVR** — critic-free reasoning RL; the open blueprint for o1-class reasoning, extended to self-verification (DeepSeekMath-V2).
- **Extreme training efficiency and low precision** — FP8 (V3), UE8M0 FP8 for domestic chips (V3.1), **FP4 quantization-aware training** of routed experts (V4), and **Muon** as the main optimizer (V4).
- **Radical openness** — MIT-licensed weights + unusually detailed technical reports and open kernels (FlashMLA, DeepGEMM).

## Published Research Focus
Efficient attention (MLA, NSA, DSA, CSA/HCA), MoE, low-precision training (FP8, FP4 QAT), systems (DualPipe, kernels), reasoning RL (GRPO, RLVR, R1), self-verifying math (DeepSeekMath-V2), residual-stream architecture (mHC), multi-token prediction, and distillation. Reports are detailed enough to be widely reproduced — MTP modules and DSA-style indexers are now standard in other Chinese open models.

## Compute & Infrastructure
Operates under **US export controls**, historically on **NVIDIA H800** (and earlier A100) GPUs with reduced interconnect — the constraint that *forced* FP8, DualPipe, MLA and comm-overlap. Since 2025 it has explicitly **co-designed number formats for domestic accelerators** (UE8M0 FP8). V4's training hardware is **disputed** (claims range from Huawei Ascend, to NVIDIA for training plus Ascend for inference, to a US-official allegation of smuggled Blackwells); DeepSeek has not confirmed any. Backed by High-Flyer's GPU stockpile.

## Stated Strategic Direction
Push **open**, **efficient** frontier models; pursue AGI via algorithmic/systems efficiency rather than brute-force compute; make **million-token context affordable** (V4 reportedly needs ~27% of V3.2's per-token FLOPs and ~10% of its KV cache at 1M); and share methods openly under MIT licenses. Positioned as proof that the frontier is not solely a function of compute budget — a narrative with major geopolitical implications for the "compute moat" thesis.

## Known Strengths & Weaknesses
- **Strengths:** attention-efficiency leadership, cost/quality, open reasoning and math (Speciale, Math-V2), architectural innovation (MLA, DSA, CSA/HCA, mHC), permissive MIT licensing, reproducible transparency.
- **Weaknesses:** hardware access constrained by export controls (and uncertain provenance), slower cadence than Qwen/Kimi in 2026 and loss of sole "open frontier" status, late multimodality (native vision only with V4.1), smaller safety-research profile, and geopolitical/regulatory headwinds (bans/scrutiny in some jurisdictions).

## Stated Hiring Focus
(Inferred from output and reports) systems/efficiency engineering (low-precision, parallelism, kernels), attention and MoE architecture, RL for reasoning (GRPO/RLVR, verifiers), and data/math/code curation. Emphasis on doing more with constrained compute — a distinctive engineering culture.

## Latest Developments (2023–Nov 2026)
2023–mid-2025 produced GRPO, MLA, DeepSeekMoE, V3's FP8 recipe, R1, and NSA. Since then DeepSeek has concentrated on making long context cheap and on designing for domestic hardware.

**Mid-2025 → Nov 2026 update:**
- **2025-08 — V3.1 hybrid thinking + UE8M0 FP8:** one checkpoint serves both modes (API `deepseek-chat` vs `deepseek-reasoner`); exponent-only block scales trade mantissa precision in scales for simpler domestic-chip hardware.
- **2025-09/12 — DSA:** lightning indexer (few heads, ReLU, FP8) selects top-k tokens for MLA; trained in two stages (dense warm-up aligning the indexer to the main attention via KL, then sparse continued training). The indexer is still $O(L^2)$ but cheap, so main attention cost becomes $O(Lk)$. (arXiv:2512.02556)
- **2025-11 — DeepSeekMath-V2:** verifier-as-reward with scaled verification compute for hard proofs — RLVR extended to domains without a cheap ground-truth checker. (arXiv:2511.22570)
- **2025-12 — mHC:** projecting hyper-connection mixing onto the Birkhoff polytope restores the identity-like mapping that plain hyper-connections break; stable at 27B. (arXiv:2512.24880)
- **2026-04 — V4 (CSA + HCA):** CSA compresses each block of KV into one entry and the indexer selects top-k compressed entries; HCA runs dense attention over much more heavily compressed KV. Sparse attention reportedly switched on at a 64K stage after dense 4K→16K training; Muon optimizer (AdamW for embeddings/heads/norms); FP4 QAT for routed experts; ~32T tokens (secondary). Tech report: arXiv:2606.19348 (as listed on its Hugging Face card).
- **2026-09 — V4.1-Flash:** a single 552B model with native vision replacing both V4-Flash and V4-Pro — consolidation toward one efficient multimodal workhorse rather than a flagship ladder.

## Interview Angles
> 💡 **What labs actually ask (DeepSeek-relevant):**
- **"Explain GRPO and how it differs from PPO."** Critic-free, group-relative normalized advantage (see [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)).
- **"What is MLA and why does it help?"** Low-rank latent KV compression → small KV cache with MHA-like quality.
- **"How did DeepSeek-V3 train so cheaply?"** FP8 + DualPipe + aux-loss-free MoE + MTP + comm-overlap on H800s.
- **"What did R1-Zero demonstrate?"** Reasoning emerges from pure RL with verifiable rewards, no SFT.
- **"Walk through DSA and V4's CSA/HCA. Where does the cost go at 1M tokens?"** Indexer is quadratic but cheap; main attention over top-k (compressed) entries; KV-cache size and indexer K cache dominate serving design.
- **"Why would a lab choose UE8M0 (power-of-two) scales?"** Cheap scale application in hardware, compatibility with domestic accelerators, at the cost of coarser scaling.

## Open Problems / Watch Items
Whether DeepSeek can keep pace at the next scale under export controls (and clarify its hardware), whether CSA/HCA-style compression preserves multi-hop reasoning at 1M context, an R2 or next reasoning flagship, extending efficiency wins to multimodal agents (V4.1), building a safety-research profile, and whether "efficiency beats compute" holds as US labs contract tens of gigawatts.

## References
- DeepSeek-AI (2024). *DeepSeek-V3.* arXiv:2412.19437.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- DeepSeek-AI (2024). *DeepSeek-V2 (MLA).* arXiv:2405.04434.
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- Yuan, J. et al. (2025). *Native Sparse Attention.* arXiv:2502.11089.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- DeepSeek-AI (2025). *DeepSeekMath-V2.* arXiv:2511.22570.
- DeepSeek-AI (2025). *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880.
- DeepSeek-AI (2026). *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* arXiv:2606.19348 (ID as listed on its Hugging Face card).
