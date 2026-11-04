# Mixture of Experts (MoE)

> **Last Updated:** 2026-11-04
> **Related Files:** [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md) · [Distributed Training](../02_pretraining/01_distributed_training.md) · [DeepSeek Roadmap](../08_lab_roadmaps/04_deepseek_roadmap.md)
> **Key Papers:** Shazeer et al. 2017 Sparsely-Gated MoE ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) · Fedus et al. 2021 Switch Transformer ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)) · Jiang et al. 2024 Mixtral ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)) · DeepSeek-AI 2024 DeepSeekMoE/V3 ([arXiv:2401.06066](https://arxiv.org/abs/2401.06066))

## Overview
Mixture of Experts decouples a model's **total parameters** (capacity/knowledge) from its **active parameters per token** (compute cost). Instead of one dense FFN, an MoE layer has many expert FFNs and a **router** that sends each token to only a few. This lets a model have, say, 10–30× the parameters of a dense model while doing roughly the same FLOPs per token — a strong lever on the scaling laws (more capacity at fixed compute). MoE went from research curiosity (2017) to the frontier default: Mixtral, DeepSeek-V3/V4, Llama 4, Qwen3/3.5/3.8, Kimi K2/K3, GLM-5, MiniMax-M2, Mistral Large 3/4 and OpenAI's gpt-oss are MoEs (GPT-4 is widely believed to be). By 2026 open MoEs reach **1–2.8T total** parameters with only ~3–5% active per token. Understanding routing, load balancing, and the systems/training challenges is now core knowledge.

## Core Concepts
**The MoE layer.** Replace the FFN with $E$ expert FFNs $\{f_1,\dots,f_E\}$ and a gating network. The router computes scores $g(x)=\text{softmax}(W_r x)$, selects the **top-$k$** experts, and combines their outputs weighted by the (renormalized) gate values:
$$y=\sum_{i\in \text{TopK}(g(x))} g_i(x)\, f_i(x)$$
With $k=2$ of $E=8$ experts (Mixtral), only 2 FFNs run per token. **Active params** ≈ embeddings + attention + $k$ experts; **total params** includes all $E$. Mixtral 8×7B: ~47B total, ~13B active. DeepSeek-V3: 671B total, **37B active** (8 of 256 routed experts + 1 shared).

**Routing.** Usually **token-choice top-k** (each token picks its experts). The router is trained jointly; it must learn useful specialization. Alternatives: **expert-choice** (each expert picks its top tokens — guarantees balance), hashing (fixed), and **soft MoE** (weighted combination of all, no hard routing).

**Load balancing.** Without intervention, the router **collapses** — a few experts get all tokens (rich-get-richer), wasting capacity and overflowing **expert capacity** (the max tokens an expert can process; excess tokens are *dropped*). The fix: an **auxiliary load-balancing loss** that encourages uniform expert utilization:
$$\mathcal{L}_{aux}=\alpha\, E \sum_{i=1}^{E} f_i\, P_i$$
where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ the mean router probability for $i$. DeepSeek introduced an **auxiliary-loss-free** balancing (per-expert bias adjusted dynamically) to avoid the aux-loss's quality cost.

**Fine-grained + shared experts (DeepSeekMoE).** Split experts into *more, smaller* ones (finer specialization) and add **shared experts** always activated (capturing common knowledge), improving the specialization/redundancy tradeoff. This design now dominates:

| Model (release) | Total / active | Routed experts (active / total) + shared |
|---|---|---|
| gpt-oss-120b (2025-08) | 116.8B / 5.1B | 4 / 128 |
| Kimi K2 (2025-07) | ~1T / 32B | 8 / 384 |
| Qwen3-Next-80B-A3B (2025-09) | 80B / ~3B | 10 / 512 + 1 |
| MiniMax-M2 (2025-10) | ~230B / 10B | 256 experts (full attention) |
| GLM-5 (2026-02) | 744B / 40B | — (adopts DSA) |
| DeepSeek-V4-Pro (2026-04) | 1.6T / 49B | — |
| Kimi K3 (2026-07) | 2.8T / 104B | 16 / 896 + 2 |
| Qwen3.8 (2026-08) | 2.4T / 95B | 10 / 512 + 1 |
| Mistral Large 4 (2026-10) | ~1.05T / 52B | — |

**Systems: expert parallelism.** Experts are sharded across devices; routing requires **all-to-all** communication (tokens → their experts' devices and back) — the dominant systems cost, demanding communication/computation overlap (DeepSeek-V3's DualPipe) and, increasingly, rack-scale NVLink/scale-up domains.

## Key Challenges
- **Training instability.** Routers are discrete/noisy; MoE runs suffer more instability, **z-loss**/router tuning needed; load-balancing loss trades quality for balance.
- **Load imbalance & token dropping.** Collapse and capacity overflow waste compute and hurt quality; capacity factor must be tuned.
- **Communication overhead.** All-to-all dominates at scale; sensitive to interconnect bandwidth.
- **Memory footprint.** All experts must be *stored* (and often loaded) even though few are active — large memory/VRAM needs for serving (a 2.8T MoE ships at ~1.56 TB even in MXFP4).
- **Inference batching.** Different tokens hit different experts; efficient batched serving and expert placement are nontrivial.
- **RL on MoE.** Routing can differ between rollout engine and trainer, and token-level importance ratios become noisy, collapsing large-MoE RL without sequence-level ratios or routing replay.
- **Fine-tuning/generalization.** MoEs can be trickier to fine-tune and may overfit; expert specialization is often less interpretable than hoped.

## Solutions & Current Best Practices
**Fine-grained routing (top-8 to top-16 of hundreds of experts) + shared experts, auxiliary-loss-free (or low-coeff) load balancing, z-loss for stability**, capacity factor tuned to minimize drops. **Expert + tensor + pipeline parallelism** with **all-to-all overlap** for training; for serving, **expert-parallel** placement across a rack-scale domain, **4-bit expert weights** (MXFP4/INT4/FP4 QAT), and prefill/decode or attention/FFN disaggregation. DeepSeek-V3's recipe (fine-grained experts, shared experts, aux-loss-free balancing, MTP, fp8, DualPipe) remains the open reference, extended by sparse attention (V3.2/V4) and Muon training. For RL, use **sequence-level importance ratios (GSPO)** or routing replay. Choose MoE when you want **capacity at fixed inference FLOPs** and can afford the memory; choose dense when memory-constrained or for simpler fine-tuning.

## Lab Perspectives
- **Google** pioneered modern MoE at scale (Shazeer 2017, GShard, Switch, GLaM, ST-MoE) — much of the foundational work; Gemini uses MoE; Gemma 4 includes a 26B-A4B MoE.
- **Mistral** mainstreamed open MoE with **Mixtral 8×7B/8×22B**, then "granular" **Mistral Large 3** (675B / 41B) and **Large 4** (~1.05T / 52B, open weights 2026-10).
- **DeepSeek** advanced the architecture most visibly: **DeepSeekMoE** (fine-grained + shared experts), aux-loss-free balancing, V3's 671B/37B-active efficiency under hardware constraints, and V4-Pro at 1.6T / 49B.
- **Moonshot / Qwen** pushed extreme sparsity: Kimi K3 activates 16 of 896 experts; Qwen3-Next/3.8 activate 10 of 512.
- **Meta** moved to MoE with **Llama 4** (Scout/Maverick/Behemoth); its 2026 Muse models are closed and their architecture undisclosed.
- **OpenAI:** gpt-oss confirmed MoE publicly (128 experts, top-4); GPT-4 widely believed to be MoE (unconfirmed).
- **NVIDIA:** Nemotron 3 interleaves Mamba-2 and MoE layers (Nano 30B-A3B: 128 experts + 1 shared).

## Latest Developments (2023–Nov 2026)
MoE became the **frontier default** (DeepSeek-V3, Llama 4, Qwen-MoE, Mixtral, Grok). Key advances: **fine-grained + shared experts**, **auxiliary-loss-free load balancing**, **fp8 MoE training**, and communication-optimal schedules. **MoE scaling laws** (relating optimal expert count/granularity to compute) emerged. Research on **upcycling** (converting a dense model into an MoE) and on serving efficiency (expert offloading, prediction) for memory-bound deployment.

**Mid-2025 → Nov 2026 update:**
- **Trillion-scale open MoEs (2025-07 → 2026-10).** Largest open weights went from 1T (Kimi K2) to 1.6T (DeepSeek-V4-Pro), 2.4T (Qwen3.8) and 2.8T (Kimi K3), with active parameters held at ~13–104B — sparsity keeps rising while active compute stays serving-friendly. Mistral Large 4 (~1.05T) is the first Western open model near 1T.
- **Compute allocation laws (2026-03).** The optimal expert:attention compute ratio grows as a power law in total compute and depends on sparsity ([arXiv:2603.10379](https://arxiv.org/abs/2603.10379)); a related ICLR 2026 result reportedly finds optimal sparsity differs for memorization vs reasoning, with reasoning sometimes preferring denser models.
- **MoE RL stability (2025-07).** GSPO's sequence-level, length-normalized importance ratio with sequence-level clipping fixes the token-level IS noise that collapsed large-MoE RL and removes the need for routing replay; used for Qwen3 ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)).
- **Balancing at extreme sparsity (2026-07).** Kimi K3's report reportedly credits "Quantile Balancing" (and SiTU-GLU) for training stability with 896 experts.
- **4-bit experts by default (2025-08 → 2026-07).** gpt-oss ships MXFP4 MoE weights; Kimi K2 Thinking uses INT4 QAT on MoE components; DeepSeek-V4 uses FP4 QAT on routed experts; Kimi K3 ships in MXFP4. Experts hold most parameters but few FLOPs, so they are the natural quantization target.
- **Rack-scale serving and disaggregation (2026).** NVIDIA's guide serves Qwen3.8 on a GB300 NVL72; Trn3 UltraServers (144 chips) target similar workloads. An attention–FFN disaggregation study finds aggregated serving wins on throughput while AFD helps interactivity, and models with cheap attention can give ≥90% of GPUs to FFN/experts ([arXiv:2605.28302](https://arxiv.org/abs/2605.28302)).
- **MoE + efficient attention.** Most new MoEs pair sparse experts with sub-quadratic attention (DSA in GLM-5, Gated DeltaNet hybrids in Qwen3-Next/3.5, KDA + MLA in Kimi Linear/K3, Mamba-2 in Nemotron 3); MiniMax-M2 is a notable full-attention exception.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why MoE — what does it buy you?"** Decouples total capacity from active FLOPs/token → more parameters/knowledge at ~fixed compute.
- **"What is the load-balancing problem and how is it solved?"** Router collapse → aux load-balancing loss ($\sum f_i P_i$) or aux-loss-free bias balancing; capacity factor + token dropping.
- **"Mixtral vs DeepSeek-V3 architecture differences?"** Few large experts (top-2 of 8) vs fine-grained many experts + shared experts + aux-loss-free balancing.
- **"What's the main systems bottleneck in MoE training?"** All-to-all communication for expert routing; needs overlap and high interconnect bandwidth.
- **"MoE vs dense at inference — the memory tradeoff?"** Fewer active FLOPs but must store all experts → high memory footprint.
- **"Why does RL on large MoEs collapse, and how do GSPO/routing replay fix it?"** Token-level IS ratios are noisy when routing differs between sampler and trainer; sequence-level ratios/clipping (GSPO) or replaying the sampler's routing decisions stabilize updates.
- **"Why has sparsity kept rising (e.g., 16 of 896 experts)?"** Capacity scales with total params while serving cost tracks active params and memory bandwidth; rack-scale domains and 4-bit experts make storing huge expert pools affordable.

## Open Problems
Optimal **expert granularity, count, and routing** lack settled theory (and may differ for knowledge vs reasoning); **training stability** at extreme sparsity and **interpretability** of expert specialization are imperfect. **Memory-efficient serving** of multi-trillion-parameter MoEs, stable **RL on MoE policies**, robust **fine-tuning**, and reliable **dense→MoE upcycling** are active research. Whether MoE or dense (or hybrids) is ultimately superior at the very largest scales remains debated.

## References
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks (Sparsely-Gated MoE).* arXiv:1701.06538.
- Fedus, W. et al. (2021). *Switch Transformer.* arXiv:2101.03961.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Dai, D. et al. (2024). *DeepSeekMoE.* arXiv:2401.06066.
- Zoph, B. et al. (2022). *ST-MoE.* arXiv:2202.08906.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
