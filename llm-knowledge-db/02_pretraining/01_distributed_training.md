# Distributed Training

> **Last Updated:** 2026-06-18
> **Related Files:** [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md) · [Mixed Precision & Quantization](04_mixed_precision_and_quantization.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md)
> **Key Papers:** Shoeybi et al. 2019 Megatron-LM ([arXiv:1909.08053](https://arxiv.org/abs/1909.08053)) · Rajbhandari et al. 2019 ZeRO ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)) · Narayanan et al. 2021 Efficient Large-Scale Training (3D) ([arXiv:2104.04473](https://arxiv.org/abs/2104.04473)) · Zhao et al. 2023 PyTorch FSDP ([arXiv:2304.11277](https://arxiv.org/abs/2304.11277))

## Overview
Frontier models have hundreds of billions to trillions of parameters and train on tens of thousands of accelerators — far exceeding any single device's memory and compute. Distributed training is the discipline of **partitioning the model, data, and computation** across devices while minimizing the communication and idle time that erode efficiency. The core challenge is the **memory wall** (model + optimizer states + activations don't fit) and the **communication wall** (synchronizing gradients/activations across a network). Mastery of the parallelism taxonomy and ZeRO/FSDP is table-stakes for any large-scale training role.

## Core Concepts
**Memory budget.** For a model with $\Psi$ parameters trained with Adam in mixed precision, the steady-state memory is roughly: fp16 params ($2\Psi$) + fp16 grads ($2\Psi$) + fp32 optimizer states (master weights $4\Psi$ + momentum $4\Psi$ + variance $4\Psi$) $\approx 16\Psi$ bytes — *plus activations*. This is why a 70B model needs ~1.1TB just for states, forcing partitioning.

**Data Parallelism (DP).** Replicate the model, split the batch; all-reduce gradients each step. Simple and communication-light per step but replicates all states — memory-bound.

**Tensor Parallelism (TP, Megatron).** Shard individual matrices across devices. For an FFN $Y=\text{GeLU}(XA)B$: split $A$ **column-wise** (no comm after) and $B$ **row-wise** (all-reduce after). Attention shards by heads. TP needs **high-bandwidth intra-node** links (NVLink) because it communicates *within every layer*; typically confined to one node (e.g., TP=8).

**Pipeline Parallelism (PP).** Partition layers into stages across devices; micro-batches flow through like an assembly line. The **bubble** (idle time at fill/drain) is the cost: bubble fraction $\approx \frac{p-1}{m}$ for $p$ stages and $m$ micro-batches. **GPipe** vs **1F1B** (interleaved, PipeDream) schedules reduce the bubble and activation memory.

**Sequence/Context Parallelism (SP).** Shard along the sequence dimension (and the parts of attention/LayerNorm not covered by TP), crucial for long context; **Ring Attention** is a form of context parallelism.

**ZeRO (Zero Redundancy Optimizer).** Eliminates DP's state replication by *partitioning* states across data-parallel ranks: **Stage 1** partitions optimizer states ($4\times$ savings on those), **Stage 2** also partitions gradients, **Stage 3** also partitions parameters (gathered on demand via all-gather, reduce-scatter for grads). ZeRO-3 ≈ PyTorch **FSDP**: each rank holds a shard, materializes full layers just-in-time, then frees them. ZeRO-Offload/Infinity push states to CPU/NVMe.

**3D Parallelism.** Compose **DP × TP × PP** (and SP) to map a model onto a cluster: TP within a node (NVLink), PP across nodes, DP across replicas. MoE adds **expert parallelism** (experts sharded across devices).

## Key Challenges
- **Communication overhead.** All-reduce/all-gather volume can dominate; bandwidth and topology (intra- vs inter-node) determine feasible configs.
- **Pipeline bubbles.** Idle stages waste compute unless micro-batch count is high and scheduling is clever.
- **Activation memory.** Often exceeds parameter memory at long context; needs checkpointing/offloading (see [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md)).
- **Straggler & fault tolerance.** At 10k+ GPUs, hardware failures are frequent; a single slow/dead node stalls a synchronous run.

## Solutions & Current Best Practices
The standard mapping: **TP=8 within a node** (NVLink), **PP across nodes** with interleaved 1F1B, **ZeRO/FSDP data parallelism** across the rest, plus **sequence parallelism** for long context. Frameworks: **Megatron-LM**, **DeepSpeed**, **PyTorch FSDP**, **Megatron-DeepSpeed**, and newer **TorchTitan**/**nanotron**. Overlap communication with computation, use bf16, gradient checkpointing, and tune micro-batch size to hide bubbles. Target high **MFU** (35–55% is good at scale).

## Lab Perspectives
- **NVIDIA/Microsoft** drive Megatron-LM and DeepSpeed/ZeRO — the de facto open stack.
- **Meta** drives **PyTorch FSDP** (ZeRO-3-style) and TorchTitan; trained Llama 3 on 16k H100s.
- **Google DeepMind** uses TPU pods with **GSPMD/Pathways** and JAX (`jax.pmap`/`shard_map`), a different but analogous parallelism model.
- **DeepSeek** engineered extreme efficiency on H800s (limited interconnect) via custom pipeline schedules (DualPipe) and communication-aware MoE (V3).

## Latest Developments (2023–2026)
**FSDP2**, **DualPipe** (DeepSeek-V3's near-zero-bubble schedule), **fp8 training** on Hopper (Transformer Engine), and **expert-parallel + communication-overlap** for huge MoE models. Fault-tolerant/elastic training (async checkpointing, hot spares) matured for 10k–100k-GPU runs. Cross-datacenter and lower-bandwidth training (DiLoCo, distributed low-communication methods) emerged as frontier clusters strain power/networking limits.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain ZeRO stages 1/2/3 and how stage 3 relates to FSDP."** Partition optimizer/grad/param states; ZeRO-3 ≈ FSDP all-gather/reduce-scatter.
- **"Design parallelism for a 400B model on 8k GPUs."** TP intra-node, PP inter-node, DP/FSDP outer, SP for context; justify by bandwidth.
- **"What is the pipeline bubble and how do you reduce it?"** $\frac{p-1}{m}$; more micro-batches, interleaved 1F1B.
- **"Why keep TP within a node?"** It communicates every layer; needs NVLink bandwidth.

## Open Problems
Scaling synchronous training past ~100k accelerators strains power, networking, and reliability; **low-communication / decentralized** training (cross-datacenter, federated) is unproven at frontier quality. Optimal automatic parallelization (vs hand-tuned configs) and robust fault tolerance without throughput loss remain open engineering frontiers.

## References
- Shoeybi, M. et al. (2019). *Megatron-LM.* arXiv:1909.08053.
- Rajbhandari, S. et al. (2019). *ZeRO.* arXiv:1910.02054.
- Narayanan, D. et al. (2021). *Efficient Large-Scale LM Training on GPU Clusters.* arXiv:2104.04473.
- Zhao, Y. et al. (2023). *PyTorch FSDP.* arXiv:2304.11277.
- DeepSeek-AI (2024). *DeepSeek-V3 (DualPipe).* arXiv:2412.19437.
