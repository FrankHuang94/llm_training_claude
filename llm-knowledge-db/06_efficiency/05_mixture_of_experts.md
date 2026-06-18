# Mixture of Experts (MoE)

> **Last Updated:** 2026-06-18
> **Related Files:** [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md) · [Distributed Training](../02_pretraining/01_distributed_training.md) · [DeepSeek Roadmap](../08_lab_roadmaps/04_deepseek_roadmap.md)
> **Key Papers:** Shazeer et al. 2017 Sparsely-Gated MoE ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) · Fedus et al. 2021 Switch Transformer ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)) · Jiang et al. 2024 Mixtral ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)) · DeepSeek-AI 2024 DeepSeekMoE/V3 ([arXiv:2401.06066](https://arxiv.org/abs/2401.06066))

## Overview
Mixture of Experts decouples a model's **total parameters** (capacity/knowledge) from its **active parameters per token** (compute cost). Instead of one dense FFN, an MoE layer has many expert FFNs and a **router** that sends each token to only a few (typically top-1 or top-2). This lets a model have, say, 10× the parameters of a dense model while doing roughly the same FLOPs per token — a strong lever on the scaling laws (more capacity at fixed compute). MoE went from research curiosity (2017) to the frontier default: Mixtral, DeepSeek-V3, Llama 4, Qwen-MoE, and (widely believed) GPT-4 are MoEs. Understanding routing, load balancing, and the systems/training challenges is now core knowledge.

## Core Concepts
**The MoE layer.** Replace the FFN with $E$ expert FFNs $\{f_1,\dots,f_E\}$ and a gating network. The router computes scores $g(x)=\text{softmax}(W_r x)$, selects the **top-$k$** experts, and combines their outputs weighted by the (renormalized) gate values:
$$y=\sum_{i\in \text{TopK}(g(x))} g_i(x)\, f_i(x)$$
With $k=2$ of $E=8$ experts (Mixtral), only 2 FFNs run per token. **Active params** ≈ embeddings + attention + $k$ experts; **total params** includes all $E$. Mixtral 8×7B: ~47B total, ~13B active. DeepSeek-V3: 671B total, **37B active**.

**Routing.** Usually **token-choice top-k** (each token picks its experts). The router is trained jointly; it must learn useful specialization. Alternatives: **expert-choice** (each expert picks its top tokens — guarantees balance), hashing (fixed), and **soft MoE** (weighted combination of all, no hard routing).

**Load balancing.** Without intervention, the router **collapses** — a few experts get all tokens (rich-get-richer), wasting capacity and overflowing **expert capacity** (the max tokens an expert can process; excess tokens are *dropped*). The fix: an **auxiliary load-balancing loss** that encourages uniform expert utilization:
$$\mathcal{L}_{aux}=\alpha\, E \sum_{i=1}^{E} f_i\, P_i$$
where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ the mean router probability for $i$. DeepSeek introduced an **auxiliary-loss-free** balancing (per-expert bias adjusted dynamically) to avoid the aux-loss's quality cost.

**Fine-grained + shared experts (DeepSeekMoE).** Split experts into *more, smaller* ones (finer specialization) and add **shared experts** always activated (capturing common knowledge), improving the specialization/redundancy tradeoff.

**Systems: expert parallelism.** Experts are sharded across devices; routing requires **all-to-all** communication (tokens → their experts' devices and back) — the dominant systems cost, demanding communication/computation overlap (DeepSeek-V3's DualPipe).

## Key Challenges
- **Training instability.** Routers are discrete/noisy; MoE runs suffer more instability, **z-loss**/router tuning needed; load-balancing loss trades quality for balance.
- **Load imbalance & token dropping.** Collapse and capacity overflow waste compute and hurt quality; capacity factor must be tuned.
- **Communication overhead.** All-to-all dominates at scale; sensitive to interconnect bandwidth.
- **Memory footprint.** All experts must be *stored* (and often loaded) even though few are active — large memory/VRAM needs for serving.
- **Inference batching.** Different tokens hit different experts; efficient batched serving and expert placement are nontrivial.
- **Fine-tuning/generalization.** MoEs can be trickier to fine-tune and may overfit; expert specialization is often less interpretable than hoped.

## Solutions & Current Best Practices
**Top-2 routing, fine-grained + shared experts, auxiliary-loss-free (or low-coeff) load balancing, z-loss for stability**, capacity factor tuned to minimize drops. **Expert + tensor + pipeline parallelism** with **all-to-all overlap** for training; for serving, **expert-parallel** placement and quantization. DeepSeek-V3's recipe (fine-grained experts, shared experts, aux-loss-free balancing, MTP, fp8, DualPipe) is the current open reference. Choose MoE when you want **capacity at fixed inference FLOPs** and can afford the memory; choose dense when memory-constrained or for simpler fine-tuning.

## Lab Perspectives
- **Google** pioneered modern MoE at scale (Shazeer 2017, GShard, Switch, GLaM, ST-MoE) — much of the foundational work; Gemini uses MoE.
- **Mistral** mainstreamed open MoE with **Mixtral 8×7B/8×22B**.
- **DeepSeek** advanced the architecture most visibly: **DeepSeekMoE** (fine-grained + shared experts), aux-loss-free balancing, and V3's 671B/37B-active efficiency under hardware constraints.
- **Meta** moved to MoE with **Llama 4** (Scout/Maverick/Behemoth).
- **OpenAI:** GPT-4 widely believed to be MoE (unconfirmed).

## Latest Developments (2023–2026)
MoE became the **frontier default** (DeepSeek-V3, Llama 4, Qwen-MoE, Mixtral, Grok). Key advances: **fine-grained + shared experts**, **auxiliary-loss-free load balancing**, **fp8 MoE training**, and communication-optimal schedules. **MoE scaling laws** (relating optimal expert count/granularity to compute) emerged. Research on **upcycling** (converting a dense model into an MoE) and on serving efficiency (expert offloading, prediction) for memory-bound deployment.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why MoE — what does it buy you?"** Decouples total capacity from active FLOPs/token → more parameters/knowledge at ~fixed compute.
- **"What is the load-balancing problem and how is it solved?"** Router collapse → aux load-balancing loss ($\sum f_i P_i$) or aux-loss-free bias balancing; capacity factor + token dropping.
- **"Mixtral vs DeepSeek-V3 architecture differences?"** Few large experts (top-2 of 8) vs fine-grained many experts + shared experts + aux-loss-free balancing.
- **"What's the main systems bottleneck in MoE training?"** All-to-all communication for expert routing; needs overlap and high interconnect bandwidth.
- **"MoE vs dense at inference — the memory tradeoff?"** Fewer active FLOPs but must store all experts → high memory footprint.

## Open Problems
Optimal **expert granularity, count, and routing** lack settled theory; **training stability** and **interpretability** of expert specialization are imperfect. **Memory-efficient serving** of huge-total-parameter MoEs, robust **fine-tuning**, and reliable **dense→MoE upcycling** are active research. Whether MoE or dense (or hybrids) is ultimately superior at the very largest scales remains debated.

## References
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks (Sparsely-Gated MoE).* arXiv:1701.06538.
- Fedus, W. et al. (2021). *Switch Transformer.* arXiv:2101.03961.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Dai, D. et al. (2024). *DeepSeekMoE.* arXiv:2401.06066.
- Zoph, B. et al. (2022). *ST-MoE.* arXiv:2202.08906.
