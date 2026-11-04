# Compute and Memory Efficiency

> **Last Updated:** 2026-11-04
> **Related Files:** [Distributed Training](01_distributed_training.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Mixed Precision & Quantization](04_mixed_precision_and_quantization.md)
> **Key Papers:** Dao et al. 2022 FlashAttention ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) · Chen et al. 2016 Gradient Checkpointing ([arXiv:1604.06174](https://arxiv.org/abs/1604.06174)) · Korthikanti et al. 2022 Activation Recomputation / Sequence Parallel ([arXiv:2205.05198](https://arxiv.org/abs/2205.05198)) · Chowdhery et al. 2022 PaLM (MFU) ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311))

## Overview
Beyond parallelism (how to *split* work across devices) lies the orthogonal problem of using each device efficiently: fitting the model and its activations in memory, and keeping the expensive tensor cores busy rather than stalled on memory traffic or communication. The headline metric is **MFU (Model FLOPs Utilization)** — the fraction of the hardware's peak FLOPs your training actually achieves for *useful* model math. Frontier runs target 35–55% MFU; the gap to 100% comes from memory-boundedness, communication, pipeline bubbles, and non-matmul overheads. Since 2025 the biggest efficiency levers have increasingly been *architectural* — sub-quadratic attention, hybrid SSM layers, extreme MoE sparsity and 4-bit arithmetic — rather than kernel tuning alone.

## Core Concepts
**The 6N FLOPs rule.** Training compute per token is $\approx 6N$ FLOPs for an $N$-parameter dense model: $2N$ forward + $4N$ backward (the backward pass costs ~2× the forward). Total training FLOPs $\approx 6ND$ for $D$ tokens — the basis of compute budgeting and scaling laws (see [Scaling Laws](../00_foundations/04_scaling_laws.md)). (Attention adds a sequence-length-dependent term; MoE uses *active* params; linear-attention/SSM layers replace the quadratic term with a constant-state recurrence.)

**MFU.** $\text{MFU}=\frac{6ND/T}{\text{peak FLOP/s}\times \text{#devices}}$ where $T$ is wall-clock training time. It measures real efficiency end-to-end (unlike "hardware FLOPs utilization" which counts recomputation). PaLM reported ~46% MFU at 540B — a benchmark figure. Note the denominator depends on the precision you quote peak at (bf16 vs fp8 vs NVFP4), so cross-run MFU comparisons must fix the precision.

**Activation memory.** Forward activations stored for the backward pass scale as $O(\text{batch}\times \text{seq}\times d\times \text{layers})$ and frequently *exceed* parameter memory at long context — the binding constraint for long-context training.

**Gradient checkpointing (activation recomputation).** Don't store all activations; store a subset ("checkpoints") and **recompute** the rest during backward. Trades ~33% extra compute for $O(\sqrt{L})$ (or better) activation memory. **Selective recomputation** (Korthikanti et al.) recomputes only cheap, memory-heavy ops (e.g., attention) and keeps expensive matmul activations — a better Pareto point.

**FlashAttention.** Removes the $O(n^2)$ attention activation memory entirely via tiling + online softmax, making long context feasible and speeding up training (see [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md)). It does not remove the $O(n^2)$ *compute*, which is what sparse and linear attention attack.

**Offloading.** Move optimizer states / params / activations to CPU RAM or NVMe (ZeRO-Offload/Infinity) when GPU memory is exhausted — trades PCIe/NVMe bandwidth for capacity; useful for memory-bound or budget-constrained setups.

## Key Challenges
- **Memory-bound operations.** Attention, LayerNorm, softmax, and elementwise ops are limited by HBM bandwidth, not FLOPs — leaving tensor cores idle.
- **Activation explosion at long context.** Long sequences blow up activation memory; checkpointing helps but adds compute.
- **Kernel launch & non-matmul overhead.** Many small ops underutilize the GPU; fusion is needed.
- **Recompute vs memory tradeoff.** Over-checkpointing wastes FLOPs; under-checkpointing OOMs — the sweet spot is workload-specific.
- **Quadratic attention at 1M context.** Even with FlashAttention, attention FLOPs dominate at very long sequence lengths, which now matter for both long-context pretraining stages and long-horizon RL rollouts.

## Solutions & Current Best Practices
**FlashAttention(-2/3) + selective gradient checkpointing + bf16/fp8 (NVFP4 on Blackwell-class) + kernel fusion** (via `torch.compile`, Triton, CUDA graphs, or fused kernels) is the standard efficiency stack. **Sequence parallelism** distributes activation memory for long context. Tune **micro-batch size** to balance memory and pipeline bubbles. Profile to find memory-bound bottlenecks and fuse them. Report and optimize **MFU** as the north-star metric. Use **CPU/NVMe offload** only when necessary (bandwidth-limited). At the architecture level, choose **sparse or hybrid linear attention** and **high-sparsity MoE** to cut FLOPs and KV per token before tuning kernels.

## Lab Perspectives
- **Google** popularized **MFU** as the standard efficiency metric (PaLM) and optimizes heavily for TPU systolic arrays; its 8th-gen TPUs split into training (8t) and inference (8i) chips.
- **NVIDIA/Meta** drive FlashAttention, selective recomputation, and `torch.compile`-based fusion in the GPU ecosystem; NVIDIA's Nemotron line pushes hybrid Mamba-Transformer MoEs trained in NVFP4.
- **DeepSeek** achieved remarkable MFU/cost efficiency on H800s (limited interconnect) through custom kernels, fp8, and communication-overlapping schedules, then cut long-context FLOPs with DSA (V3.2) and CSA/HCA (V4).
- **Moonshot / Qwen** made hybrid linear attention (KDA, Gated DeltaNet at a 3:1 ratio) production-grade.
- **OpenAI/Anthropic** keep specifics closed but clearly optimize MFU and memory aggressively to control run cost.

## Latest Developments (2023–Nov 2026)
**FlashAttention-3** exploits Hopper async + fp8. `torch.compile`/Triton fusion and **CUDA graphs** reduce launch overhead. **fp8** raises achievable throughput (and complicates MFU accounting). **MoE** shifts the calculus toward *active*-parameter efficiency and communication overlap. Cross-cluster and power-constrained training pushes interest in compute/communication co-optimization and energy-per-token as an emerging metric.

**Mid-2025 → Nov 2026 update:**
- **Learned sparse attention (2025-09/12).** DeepSeek Sparse Attention adds an FP8 "lightning indexer" (few heads, ReLU) that scores past tokens so main MLA attention runs only over the top-$k$; the indexer is still $O(L^2)$ but cheap, trained by a KL-aligned dense warm-up then sparse continued training ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556)). GLM-5 (744B) adopted it ([arXiv:2602.15763](https://arxiv.org/abs/2602.15763)).
- **Compressed sparse attention (2026-04).** DeepSeek-V4 compresses KV blocks (CSA) and adds heavily compressed dense attention (HCA); at 1M context V4-Pro reportedly needs ~27% of V3.2's per-token FLOPs and ~10% of its KV cache (DeepSeek-V4 tech report: arXiv:2606.19348, as listed on its Hugging Face card).
- **Hybrid linear attention (2025-09/10).** Qwen3-Next interleaves Gated DeltaNet and gated attention 3:1 in an 80B-A3B MoE, reporting >10× Qwen3-32B prefill throughput beyond 32K; Kimi Linear (KDA + MLA, 3:1) reports up to 75% less KV cache and up to 6× decode throughput at 1M ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)).
- **SSM-heavy hybrids (2025-08 → 2026-03).** Nemotron Nano 2 keeps only 6 attention layers out of 62 (28 Mamba-2, 28 FFN), reporting up to 6.3× Qwen3-8B throughput on reasoning workloads ([arXiv:2508.14444](https://arxiv.org/abs/2508.14444)); Mamba-3's MIMO formulation raises decode arithmetic intensity and matches Mamba-2 perplexity with half the state ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)).
- **Compute allocation inside MoE (2026-03).** The optimal expert:attention compute ratio grows as a power law in total compute and depends on sparsity ([arXiv:2603.10379](https://arxiv.org/abs/2603.10379)).
- **Hardware specialization (2025-09 → 2026-04).** NVIDIA's Rubin CPX (GDDR7, ~30 PF NVFP4) targets compute-bound prefill separately from bandwidth-bound decode; Rubin GPUs are quoted at 50 PF NVFP4 with 288 GB HBM4; Trainium3 at 2.52 PF FP8 per chip. Peak-FLOP figures are now routinely quoted in 4-bit, inflating apparent MFU denominators.
- **Counterpoint (2025-10).** MiniMax reverted M2 to full attention: hybrids matched full attention on standard benchmarks but lost on multi-hop reasoning, and linear-attention state was sensitive to low precision — efficiency architectures must be validated on hard reasoning, not perplexity alone.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Derive training FLOPs for an N-parameter model."** $6ND$: $2N$ fwd, $4N$ bwd per token; attention term separately.
- **"What is MFU and why is it usually <60%?"** Useful-FLOPs / peak; memory-boundedness, comm, bubbles, non-matmul ops.
- **"How does gradient checkpointing trade compute for memory?"** Store fewer activations, recompute in backward (~33% extra FLOPs); selective recompute optimizes it.
- **"Your long-context run OOMs on activations — what do you do?"** FlashAttention, selective checkpointing, sequence parallelism, smaller micro-batch, offload.
- **"DSA's indexer is still $O(L^2)$ — why does it save compute?"** The indexer uses a few low-dimensional FP8 heads with ReLU scores, so its quadratic term has a tiny constant; the expensive MLA attention then runs over only top-$k$ tokens, making the main cost $O(Lk)$.

## Open Problems
Pushing MFU higher as models grow more communication-bound (MoE, long context, multi-datacenter) is an open systems challenge. The optimal automatic tradeoff between recomputation, offloading, and parallelism (vs hand-tuning) is unsolved, and energy/cost-per-token efficiency at the frontier is increasingly constrained by power and networking rather than raw FLOPs. Whether sparse or hybrid-linear attention can match full attention on multi-hop reasoning at 1M context, and how to report efficiency fairly across precisions and architectures, remain open.

## References
- Dao, T. et al. (2022). *FlashAttention.* arXiv:2205.14135.
- Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174.
- Korthikanti, V. et al. (2022). *Reducing Activation Recomputation.* arXiv:2205.05198.
- Chowdhery, A. et al. (2022). *PaLM (MFU).* arXiv:2204.02311.
- Kaplan, J. et al. (2020). *Scaling Laws (6N rule).* arXiv:2001.08361.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- NVIDIA (2025). *NVIDIA Nemotron Nano 2.* arXiv:2508.14444.
