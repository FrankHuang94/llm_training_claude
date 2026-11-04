# Distributed Training

> **Last Updated:** 2026-11-04
> **Related Files:** [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md) · [Mixed Precision & Quantization](04_mixed_precision_and_quantization.md) · [Mixture of Experts](../06_efficiency/05_mixture_of_experts.md)
> **Key Papers:** Shoeybi et al. 2019 Megatron-LM ([arXiv:1909.08053](https://arxiv.org/abs/1909.08053)) · Rajbhandari et al. 2019 ZeRO ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)) · Narayanan et al. 2021 Efficient Large-Scale Training (3D) ([arXiv:2104.04473](https://arxiv.org/abs/2104.04473)) · Zhao et al. 2023 PyTorch FSDP ([arXiv:2304.11277](https://arxiv.org/abs/2304.11277))

## Overview
Frontier models have hundreds of billions to trillions of parameters (open-weight MoEs alone now reach 1–2.8T total) and train on tens to hundreds of thousands of accelerators — far exceeding any single device's memory and compute. Distributed training is the discipline of **partitioning the model, data, and computation** across devices while minimizing the communication and idle time that erode efficiency. The core challenge is the **memory wall** (model + optimizer states + activations don't fit) and the **communication wall** (synchronizing gradients/activations across a network). Mastery of the parallelism taxonomy and ZeRO/FSDP is table-stakes for any large-scale training role.

## Core Concepts
**Memory budget.** For a model with $\Psi$ parameters trained with Adam in mixed precision, the steady-state memory is roughly: fp16 params ($2\Psi$) + fp16 grads ($2\Psi$) + fp32 optimizer states (master weights $4\Psi$ + momentum $4\Psi$ + variance $4\Psi$) $\approx 16\Psi$ bytes — *plus activations*. This is why a 70B model needs ~1.1TB just for states, forcing partitioning.

**Data Parallelism (DP).** Replicate the model, split the batch; all-reduce gradients each step. Simple and communication-light per step but replicates all states — memory-bound.

**Tensor Parallelism (TP, Megatron).** Shard individual matrices across devices. For an FFN $Y=\text{GeLU}(XA)B$: split $A$ **column-wise** (no comm after) and $B$ **row-wise** (all-reduce after). Attention shards by heads. TP needs **high-bandwidth scale-up** links (NVLink) because it communicates *within every layer*; typically confined to one NVLink domain (TP=8 on an HGX node; rack-scale domains such as NVL72 widen this).

**Pipeline Parallelism (PP).** Partition layers into stages across devices; micro-batches flow through like an assembly line. The **bubble** (idle time at fill/drain) is the cost: bubble fraction $\approx \frac{p-1}{m}$ for $p$ stages and $m$ micro-batches. **GPipe** vs **1F1B** (interleaved, PipeDream) schedules reduce the bubble and activation memory.

**Sequence/Context Parallelism (SP).** Shard along the sequence dimension (and the parts of attention/LayerNorm not covered by TP), crucial for long context; **Ring Attention** is a form of context parallelism.

**ZeRO (Zero Redundancy Optimizer).** Eliminates DP's state replication by *partitioning* states across data-parallel ranks: **Stage 1** partitions optimizer states ($4\times$ savings on those), **Stage 2** also partitions gradients, **Stage 3** also partitions parameters (gathered on demand via all-gather, reduce-scatter for grads). ZeRO-3 ≈ PyTorch **FSDP**: each rank holds a shard, materializes full layers just-in-time, then frees them. ZeRO-Offload/Infinity push states to CPU/NVMe.

**3D Parallelism.** Compose **DP × TP × PP** (and SP) to map a model onto a cluster: TP within a node (NVLink), PP across nodes, DP across replicas. MoE adds **expert parallelism** (experts sharded across devices, with all-to-all token dispatch).

**Sharding vs matrix optimizers.** Element-wise optimizers (Adam) shard trivially under ZeRO/FSDP. Matrix optimizers (Muon) need the *whole* momentum matrix for Newton–Schulz orthogonalization, forcing an all-gather per weight matrix — the motivation for sharding-native designs like Dion.

## Key Challenges
- **Communication overhead.** All-reduce/all-gather volume can dominate; bandwidth and topology (intra- vs inter-node) determine feasible configs.
- **Pipeline bubbles.** Idle stages waste compute unless micro-batch count is high and scheduling is clever.
- **Activation memory.** Often exceeds parameter memory at long context; needs checkpointing/offloading (see [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md)).
- **Straggler & fault tolerance.** At 10k+ GPUs, hardware failures are frequent; a single slow/dead node stalls a synchronous run.
- **Extreme expert sparsity.** Hundreds of fine-grained experts (e.g., 896 routed, 16 active in Kimi K3) make all-to-all volume and per-expert load skew the binding constraint.

## Solutions & Current Best Practices
The standard mapping: **TP within the NVLink domain**, **PP across nodes** with interleaved 1F1B, **ZeRO/FSDP data parallelism** across the rest, **expert parallelism** for MoE layers, plus **sequence parallelism** for long context. Frameworks: **Megatron-LM**, **DeepSpeed**, **PyTorch FSDP**, **Megatron-DeepSpeed**, and newer **TorchTitan**/**nanotron**. Overlap communication with computation, use bf16/fp8 (and now NVFP4 on Blackwell-class parts), gradient checkpointing, and tune micro-batch size to hide bubbles. Target high **MFU** (35–55% is good at scale). For RL post-training, separate **rollout (inference-engine) workers from trainer workers** and treat weight synchronization as part of the parallel plan.

## Lab Perspectives
- **NVIDIA/Microsoft** drive Megatron-LM and DeepSpeed/ZeRO — the de facto open stack; Microsoft also proposed Dion for sharding-friendly orthonormalized updates.
- **Meta** drives **PyTorch FSDP** (ZeRO-3-style) and TorchTitan; trained Llama 3 on 16k H100s.
- **Google DeepMind** uses TPU pods with **GSPMD/Pathways** and JAX (`jax.pmap`/`shard_map`), a different but analogous parallelism model; Ironwood (TPU v7) reached GA in 2026-03.
- **DeepSeek** engineered extreme efficiency on H800s (limited interconnect) via custom pipeline schedules (DualPipe) and communication-aware MoE (V3), now extended to a 1.6T-parameter MoE (V4-Pro).
- **Moonshot** trained a 1T-parameter MoE (Kimi K2) on 15.5T tokens with distributed Muon (MuonClip); **Zhipu** built asynchronous agentic-RL infrastructure (slime) and GLM-5 was reportedly trained entirely on Huawei Ascend + MindSpore (unconfirmed by Z.ai).

## Latest Developments (2023–Nov 2026)
**FSDP2**, **DualPipe** (DeepSeek-V3's near-zero-bubble schedule), **fp8 training** on Hopper (Transformer Engine), and **expert-parallel + communication-overlap** for huge MoE models. Fault-tolerant/elastic training (async checkpointing, hot spares) matured for 10k–100k-GPU runs. Cross-datacenter and lower-bandwidth training (DiLoCo, distributed low-communication methods) emerged as frontier clusters strain power/networking limits.

**Mid-2025 → Nov 2026 update:**
- **Trillion-scale open MoEs stress expert parallelism (2025-07 → 2026-08).** Kimi K2 (1T / 32B active, 384 experts, 8 active), DeepSeek-V4-Pro (1.6T / 49B, 2026-04), Kimi K3 (2.8T / 104B, 896 routed experts with 16 active + 2 shared, 2026-07) and Qwen3.8 (2.4T / 95B, 512 experts, 2026-08). NVIDIA's serving guide targets Qwen3.8 on a GB300 NVL72 — the rack-scale NVLink domain is becoming the unit for EP.
- **Sharding-native matrix optimizers (2025-04, rev. 2025-09).** Dion replaces Newton–Schulz with amortized power iteration on the momentum, orthonormalizing only a rank-$r$ subspace; it is compatible with FSDP and TP sharding without a full-matrix gather ([arXiv:2504.05295](https://arxiv.org/abs/2504.05295)). Kimi K2 showed distributed Muon is viable at 1T parameters ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)).
- **Rack-scale scale-up domains (2025-12 → 2026).** Trainium3 UltraServers hold 144 chips (362 PF FP8, all-to-all NeuronSwitch fabric, clusters up to 1M chips); NVIDIA's Vera Rubin NVL144 rack is ~3.6 EF NVFP4 across 72 packages; Google split its 8th-gen TPU into training (8t) and inference (8i) parts (2026-04). Larger scale-up domains let TP/EP span far more than 8 devices.
- **Gigawatt-scale clusters.** Compute is now contracted in GW (OpenAI–AMD 6 GW of MI450 from H2 2026; Anthropic up to 1M TPUs and >1 GW; Project Rainier with ~500K Trainium2). OpenAI's GPT-6 Astra (2026-09) was reportedly trained on >100k GPUs (single source). Power, not chips, increasingly bounds synchronous scale.
- **RL systems as distributed training (2025-05 → 2026-02).** Fully asynchronous RL decouples rollout and training workers with staleness-bounded PPO (AReaL, up to 2.77× faster, [arXiv:2505.24298](https://arxiv.org/abs/2505.24298)); PipelineRL pushes **in-flight weight updates** to generators mid-sequence (~2× faster on long-CoT, [arXiv:2509.19128](https://arxiv.org/abs/2509.19128)); slime (Megatron + SGLang) powers GLM-5's agentic RL.
- **Numerics across the trainer/inference boundary (2025-09/10).** Non-batch-invariant kernels and BF16 rounding make the rollout engine's log-probs diverge from the trainer's; batch-invariant kernels (Thinking Machines) and switching RL to FP16 ([arXiv:2510.26788](https://arxiv.org/abs/2510.26788)) restore on-policyness — a new correctness requirement for disaggregated RL stacks.
- **Hardware-aware formats on non-NVIDIA silicon (2025-08 →).** DeepSeek's UE8M0 (power-of-two) FP8 scales were framed as targeting domestic Chinese chips; the training hardware of DeepSeek-V4 remains disputed.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain ZeRO stages 1/2/3 and how stage 3 relates to FSDP."** Partition optimizer/grad/param states; ZeRO-3 ≈ FSDP all-gather/reduce-scatter.
- **"Design parallelism for a 400B model on 8k GPUs."** TP intra-node, PP inter-node, DP/FSDP outer, SP for context; justify by bandwidth.
- **"What is the pipeline bubble and how do you reduce it?"** $\frac{p-1}{m}$; more micro-batches, interleaved 1F1B.
- **"Why keep TP within a node?"** It communicates every layer; needs NVLink bandwidth.
- **"How would you run Muon under FSDP?"** Newton–Schulz needs the full matrix → gather per matrix (or assign whole matrices to ranks), or use a sharding-native variant (Dion: low-rank power iteration on sharded momentum).
- **"Design an RL post-training system for a 1T MoE."** Separate rollout engines from trainers, async with bounded staleness, fast weight broadcast, and matched numerics (batch-invariant kernels / FP16 / routing consistency) so 'on-policy' stays on-policy.

## Open Problems
Scaling synchronous training past ~100k accelerators strains power, networking, and reliability; **low-communication / decentralized** training (cross-datacenter, federated) is unproven at frontier quality. Optimal automatic parallelization (vs hand-tuned configs) and robust fault tolerance without throughput loss remain open engineering frontiers, as do communication-efficient matrix optimizers at trillion scale and the throughput/staleness trade-off in asynchronous RL.

## References
- Shoeybi, M. et al. (2019). *Megatron-LM.* arXiv:1909.08053.
- Rajbhandari, S. et al. (2019). *ZeRO.* arXiv:1910.02054.
- Narayanan, D. et al. (2021). *Efficient Large-Scale LM Training on GPU Clusters.* arXiv:2104.04473.
- Zhao, Y. et al. (2023). *PyTorch FSDP.* arXiv:2304.11277.
- DeepSeek-AI (2024). *DeepSeek-V3 (DualPipe).* arXiv:2412.19437.
- Ahn, K. et al. (2025). *Dion: Distributed Orthonormalized Updates.* arXiv:2504.05295.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Fu, W. et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* arXiv:2505.24298.
- Piché, A. et al. (2025). *PipelineRL.* arXiv:2509.19128.
