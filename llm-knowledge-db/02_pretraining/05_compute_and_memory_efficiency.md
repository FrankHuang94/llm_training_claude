# Compute and Memory Efficiency

> **Last Updated:** 2026-06-18
> **Related Files:** [Distributed Training](01_distributed_training.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Mixed Precision & Quantization](04_mixed_precision_and_quantization.md)
> **Key Papers:** Dao et al. 2022 FlashAttention ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) · Chen et al. 2016 Gradient Checkpointing ([arXiv:1604.06174](https://arxiv.org/abs/1604.06174)) · Korthikanti et al. 2022 Activation Recomputation / Sequence Parallel ([arXiv:2205.05198](https://arxiv.org/abs/2205.05198)) · Chowdhery et al. 2022 PaLM (MFU) ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311))

## Overview
Beyond parallelism (how to *split* work across devices) lies the orthogonal problem of using each device efficiently: fitting the model and its activations in memory, and keeping the expensive tensor cores busy rather than stalled on memory traffic or communication. The headline metric is **MFU (Model FLOPs Utilization)** — the fraction of the hardware's peak FLOPs your training actually achieves for *useful* model math. Frontier runs target 35–55% MFU; the gap to 100% comes from memory-boundedness, communication, pipeline bubbles, and non-matmul overheads.

## Core Concepts
**The 6N FLOPs rule.** Training compute per token is $\approx 6N$ FLOPs for an $N$-parameter dense model: $2N$ forward + $4N$ backward (the backward pass costs ~2× the forward). Total training FLOPs $\approx 6ND$ for $D$ tokens — the basis of compute budgeting and scaling laws (see [Scaling Laws](../00_foundations/04_scaling_laws.md)). (Attention adds a sequence-length-dependent term; MoE uses *active* params.)

**MFU.** $\text{MFU}=\frac{6ND/T}{\text{peak FLOP/s}\times \text{#devices}}$ where $T$ is wall-clock training time. It measures real efficiency end-to-end (unlike "hardware FLOPs utilization" which counts recomputation). PaLM reported ~46% MFU at 540B — a benchmark figure.

**Activation memory.** Forward activations stored for the backward pass scale as $O(\text{batch}\times \text{seq}\times d\times \text{layers})$ and frequently *exceed* parameter memory at long context — the binding constraint for long-context training.

**Gradient checkpointing (activation recomputation).** Don't store all activations; store a subset ("checkpoints") and **recompute** the rest during backward. Trades ~33% extra compute for $O(\sqrt{L})$ (or better) activation memory. **Selective recomputation** (Korthikanti et al.) recomputes only cheap, memory-heavy ops (e.g., attention) and keeps expensive matmul activations — a better Pareto point.

**FlashAttention.** Removes the $O(n^2)$ attention activation memory entirely via tiling + online softmax, making long context feasible and speeding up training (see [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md)).

**Offloading.** Move optimizer states / params / activations to CPU RAM or NVMe (ZeRO-Offload/Infinity) when GPU memory is exhausted — trades PCIe/NVMe bandwidth for capacity; useful for memory-bound or budget-constrained setups.

## Key Challenges
- **Memory-bound operations.** Attention, LayerNorm, softmax, and elementwise ops are limited by HBM bandwidth, not FLOPs — leaving tensor cores idle.
- **Activation explosion at long context.** Long sequences blow up activation memory; checkpointing helps but adds compute.
- **Kernel launch & non-matmul overhead.** Many small ops underutilize the GPU; fusion is needed.
- **Recompute vs memory tradeoff.** Over-checkpointing wastes FLOPs; under-checkpointing OOMs — the sweet spot is workload-specific.

## Solutions & Current Best Practices
**FlashAttention(-2/3) + selective gradient checkpointing + bf16/fp8 + kernel fusion** (via `torch.compile`, Triton, CUDA graphs, or fused kernels) is the standard efficiency stack. **Sequence parallelism** distributes activation memory for long context. Tune **micro-batch size** to balance memory and pipeline bubbles. Profile to find memory-bound bottlenecks and fuse them. Report and optimize **MFU** as the north-star metric. Use **CPU/NVMe offload** only when necessary (bandwidth-limited).

## Lab Perspectives
- **Google** popularized **MFU** as the standard efficiency metric (PaLM) and optimizes heavily for TPU systolic arrays.
- **NVIDIA/Meta** drive FlashAttention, selective recomputation, and `torch.compile`-based fusion in the GPU ecosystem.
- **DeepSeek** achieved remarkable MFU/cost efficiency on H800s (limited interconnect) through custom kernels, fp8, and communication-overlapping schedules.
- **OpenAI/Anthropic** keep specifics closed but clearly optimize MFU and memory aggressively to control run cost.

## Latest Developments (2023–2026)
**FlashAttention-3** exploits Hopper async + fp8. `torch.compile`/Triton fusion and **CUDA graphs** reduce launch overhead. **fp8** raises achievable throughput (and complicates MFU accounting). **MoE** shifts the calculus toward *active*-parameter efficiency and communication overlap. Cross-cluster and power-constrained training pushes interest in compute/communication co-optimization and energy-per-token as an emerging metric.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Derive training FLOPs for an N-parameter model."** $6ND$: $2N$ fwd, $4N$ bwd per token; attention term separately.
- **"What is MFU and why is it usually <60%?"** Useful-FLOPs / peak; memory-boundedness, comm, bubbles, non-matmul ops.
- **"How does gradient checkpointing trade compute for memory?"** Store fewer activations, recompute in backward (~33% extra FLOPs); selective recompute optimizes it.
- **"Your long-context run OOMs on activations — what do you do?"** FlashAttention, selective checkpointing, sequence parallelism, smaller micro-batch, offload.

## Open Problems
Pushing MFU higher as models grow more communication-bound (MoE, long context, multi-datacenter) is an open systems challenge. The optimal automatic tradeoff between recomputation, offloading, and parallelism (vs hand-tuning) is unsolved, and energy/cost-per-token efficiency at the frontier is increasingly constrained by power and networking rather than raw FLOPs.

## References
- Dao, T. et al. (2022). *FlashAttention.* arXiv:2205.14135.
- Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174.
- Korthikanti, V. et al. (2022). *Reducing Activation Recomputation.* arXiv:2205.05198.
- Chowdhery, A. et al. (2022). *PaLM (MFU).* arXiv:2204.02311.
- Kaplan, J. et al. (2020). *Scaling Laws (6N rule).* arXiv:2001.08361.
