# Checkpointing and Resumption

> **Last Updated:** 2026-06-18
> **Related Files:** [Distributed Training](01_distributed_training.md) · [Training Stability](02_training_stability.md) · [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md)
> **Key Papers:** Mohan et al. 2021 CheckFreq ([USENIX FAST](https://www.usenix.org/conference/fast21/presentation/mohan)) · Eisenman et al. 2022 Check-N-Run ([arXiv:2010.08679](https://arxiv.org/abs/2010.08679)) · Wang et al. 2023 Gemini/CheckN (in-memory checkpointing, SOSP) · Meta 2024 Llama 3 (fault tolerance) ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783))

## Overview
A frontier training run executes for weeks-to-months across tens of thousands of accelerators, where **hardware failures are not exceptional but routine** — Meta reported hundreds of interruptions during Llama 3's 16k-GPU run, dominated by GPU/HBM failures. Without robust checkpointing, a single failure would forfeit days of compute. Checkpointing and resumption are therefore core reliability infrastructure: the system must persist consistent training state frequently and cheaply, and resume **bitwise-equivalently** so the run continues as if nothing happened. This is unglamorous but heavily tested in systems-oriented interviews.

## Core Concepts
**What must be saved.** A resumable checkpoint includes: model parameters, **optimizer states** (Adam $m,v$ — often the largest component), the **LR scheduler / step counter**, **RNG states** (for dropout/data-shuffle determinism), and the **data-loader position** (which samples have been consumed). Omitting any of these breaks exact resumption.

**Sharded checkpoints.** Under 3D parallelism, each rank holds a *shard* of params/optimizer state. Checkpoints are saved **distributed** (each rank writes its shard) for speed, with metadata to reconstruct the global state and to **reshard** when resuming on a different parallelism layout (distributed checkpoint / DCP in PyTorch). Naively gathering everything to rank 0 doesn't scale.

**Checkpoint cost.** State can be terabytes (e.g., a 70B model's bf16 params + fp32 Adam states ≈ ~1TB+). Writing this to persistent storage stalls training (the "stop-the-world" cost), so frequency trades reliability against throughput.

**Asynchronous / in-memory checkpointing.** Reduce the stall: **snapshot to host (CPU) RAM** quickly (fast device→host copy), then **flush to durable storage in the background** while training proceeds. Systems like CheckFreq, Check-N-Run, and Gemini/ByteCheckpoint pipeline and overlap saving. Some keep redundant in-memory copies across nodes for near-instant recovery without hitting disk.

**Resumption & determinism.** Exact resumption requires restoring RNG and data position so the post-resume trajectory matches an uninterrupted run. Deterministic, resumable data loading (index-based, seedable) is essential; non-determinism complicates debugging instabilities (see [Training Stability](02_training_stability.md)).

## Key Challenges
- **I/O bottleneck.** Multi-TB writes saturate storage/network; synchronous saving wastes expensive GPU time.
- **Frequency tradeoff.** Frequent checkpoints = less lost work per failure but more overhead; the optimal interval depends on failure rate and checkpoint cost (Young/Daly formula: optimal interval $\approx \sqrt{2\cdot C\cdot \text{MTBF}}$).
- **Resharding.** Resuming on a different GPU count / parallelism config requires repartitioning saved shards — nontrivial.
- **Silent corruption.** A faulty node can write corrupt shards; checkpoints need validation/checksums.
- **Straggler & partial failure.** One dead rank can block a synchronous save or require a full rollback.

## Solutions & Current Best Practices
**Asynchronous, sharded checkpointing** (snapshot to host RAM, flush in background) with **distributed checkpoint formats** that support resharding. Save **all** resumption state (optimizer, RNG, scheduler, data position). Set the **checkpoint interval** via failure rate and cost (Young/Daly). Add **redundancy** (in-memory copies across nodes) for fast recovery, plus periodic durable snapshots for catastrophic failures. **Validate** checkpoints (checksums) and support **fast detection + automatic restart** (elastic training, hot spares). Pair with stability tooling so you can roll back and **skip offending batches** on a loss spike.

## Lab Perspectives
- **Meta** (Llama 3) documented frequent interruptions and an automated detect-restart pipeline achieving high "effective training time"; drives PyTorch DCP.
- **Google** uses TPU-pod checkpointing with JAX/Orbax and in-memory replication (Pathways resilience).
- **ByteDance/others** published **ByteCheckpoint** and similar high-throughput, resharding-capable systems.
- **OpenAI/Anthropic** keep specifics closed but emphasize reliable, deterministic, resumable infrastructure as a competitive moat.

## Latest Developments (2023–2026)
**Asynchronous + in-memory + multi-tier (RAM→local SSD→remote)** checkpointing is now standard at scale. **Elastic/fault-tolerant training** (automatic node replacement, hot spares, continued training despite failures) matured. Research on **near-zero-overhead** checkpointing and **partial/incremental** checkpoints (only changed shards) continues, alongside resilience for ultra-large (100k-GPU) and cross-datacenter runs.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What state must a resumable checkpoint contain?"** Params, optimizer states, scheduler/step, RNG, data-loader position.
- **"How do you checkpoint a 70B model without stalling training?"** Sharded + async: fast snapshot to host RAM, background flush; in-memory redundancy.
- **"How often should you checkpoint?"** Young/Daly: balance checkpoint cost against MTBF (~$\sqrt{2C\cdot\text{MTBF}}$).
- **"How do you resume on a different GPU count?"** Distributed/resharding-capable checkpoint format.

## Open Problems
Achieving truly **zero-overhead** checkpointing as state grows to many terabytes is unsolved. Robust, automatic resharding across arbitrary parallelism layouts, end-to-end silent-corruption protection, and resilient training across unreliable/heterogeneous or geographically distributed hardware remain active systems-research frontiers.

## References
- Mohan, J. et al. (2021). *CheckFreq.* USENIX FAST.
- Eisenman, A. et al. (2022). *Check-N-Run.* arXiv:2010.08679.
- Wan, B. et al. (2024). *ByteCheckpoint.* arXiv:2407.20143.
- Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Daly, J. (2006). *A Higher Order Estimate of the Optimum Checkpoint Interval.* FGCS.
