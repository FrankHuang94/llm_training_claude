# System Design Questions

> **Last Updated:** 2026-06-18
> **Related Files:** [Distributed Training](../02_pretraining/01_distributed_training.md) · [RLHF](../03_posttraining/03_RLHF.md) · [Evaluation & Benchmarks](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)
> **Key Papers:** Narayanan et al. 2021 (3D parallelism) · Ouyang et al. 2022 InstructGPT · Grattafiori et al. 2024 Llama 3 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783))

## Overview
LLM system-design interviews test whether you can **reason end-to-end about building and operating large models** — making and justifying tradeoffs under compute, memory, data, and reliability constraints. Unlike a single fact question, these are open-ended: the interviewer wants to see structured thinking, awareness of bottlenecks, back-of-envelope math, and knowledge of what real systems do. This file gives a **framework** plus worked outlines for the canonical prompts.

## A General Framework (use for any prompt)
1. **Clarify requirements & constraints:** model size, token budget, hardware (count/type/interconnect), latency/throughput targets, deadlines, budget.
2. **Back-of-envelope math:** FLOPs ($C\approx 6ND$), memory ($\approx 16\Psi$ bytes Adam states + activations), KV cache, MFU target, wall-clock estimate.
3. **Core design:** parallelism strategy, data pipeline, optimizer/schedule, precision, checkpointing.
4. **Bottlenecks & mitigations:** memory wall, communication, pipeline bubbles, stragglers, instability.
5. **Reliability & monitoring:** fault tolerance, loss/grad monitoring, eval cadence.
6. **Tradeoffs & alternatives:** state what you'd change under different constraints.

## Worked Outlines

### "Design GPT-4-scale training infrastructure" / "Set up a 1000-GPU run"
- **Math first:** e.g., a 70B model on 15T tokens → $C\approx 6\cdot70\text{e}9\cdot15\text{e}12 \approx 6.3\text{e}24$ FLOPs. At ~40% MFU on 1000 H100s (~1e15 bf16 FLOP/s each) → ~$4\text{e}17$ eff FLOP/s → ~**180 days**. Show you can estimate and then *adjust* (more GPUs, smaller model, fewer tokens).
- **Parallelism:** **TP=8 within node** (NVLink), **PP across nodes** (interleaved 1F1B to shrink bubbles), **ZeRO/FSDP DP** outer, **sequence parallel** for long context (see [Distributed Training](../02_pretraining/01_distributed_training.md)). Justify by interconnect bandwidth.
- **Memory:** bf16 compute + fp32 master, ZeRO-partitioned optimizer states, **selective gradient checkpointing**, FlashAttention.
- **Data:** sharded, deduplicated, decontaminated, **resumable** streaming loader; documented mixture + annealing phase.
- **Stability:** warmup + WSD/cosine, gradient clipping, z-loss/QK-norm, μP for HP transfer; monitor grad norm; **rollback + skip-batch** on spikes.
- **Reliability:** **async sharded checkpointing** (host-RAM snapshot + background flush), auto-detect-and-restart, hot spares (see [Checkpointing](../02_pretraining/06_checkpoint_and_resumption.md)).
- **Metric:** optimize **MFU**; report effective training time.

### "Design an RLHF pipeline from scratch"
- **Stages:** SFT (curated/synthetic demos, loss-masked) → **reward model** (Bradley-Terry, pairwise prefs, length-debiased, ensembled) → **PPO** (policy+ref+RM+value; KL penalty; clip) — or **DPO/GRPO** for simplicity/cost (see [RLHF](../03_posttraining/03_RLHF.md)).
- **Infra:** PPO needs **4 models in memory** + **generation rollouts** (an inference engine, e.g., vLLM) feeding training — design the **actor/learner** split, batching, and KV reuse. Frameworks: TRL/OpenRLHF/verl.
- **Failure modes:** reward hacking/overoptimization → tune **β**, monitor reward-vs-KL, iterate preference data; length bias → debias RM.
- **Data flywheel:** iterative rounds (re-collect prefs with updated policy); rejection sampling for SFT data.
- **Eval:** preference win-rate, safety/refusal, capability regressions (alignment tax), Arena-style + held-out (see [Evaluation](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)).

### "Design a high-throughput inference/serving system"
- **Bottleneck:** decoding is **memory-bandwidth-bound**; KV cache dominates memory.
- **Techniques:** **continuous batching**, **PagedAttention** (vLLM), **prefix caching**, **quantization** (W4A16/fp8 + KV quant), **speculative decoding** (EAGLE/Medusa), **GQA/MLA**, tensor/pipeline parallel for big models, **disaggregated prefill/decode**.
- **Tradeoffs:** latency (small batch, speculative) vs throughput (large batch); for reasoning models, budget **test-time compute**.

### "How would you evaluate a new reasoning model?"
- Portfolio: AIME/FrontierMath, GPQA, **SWE-bench Verified**, LiveCodeBench; **contamination checks** + **perturbation tests**; controlled **inference budget** (pass@1 vs pass@k); human review; safety/refusal evals. Beware saturation/contamination (see [Evaluation](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Estimate the time/cost to train model X on N GPUs."** Always do the $6ND$ + MFU math out loud.
- **"Your run OOMs / has loss spikes / a node dies — what do you do?"** Map to memory (checkpointing/parallelism), stability (clip/z-loss/rollback), reliability (async checkpoint + restart).
- **"Pick a parallelism strategy and justify it by hardware."** TP intra-node, PP inter-node, FSDP outer, SP for context.
- **"DPO vs PPO for your RLHF system — which and why?"** Cost/stability vs on-policy quality.

## Common Pitfalls to Avoid
- Jumping to a solution without **clarifying constraints** or doing the **math**.
- Ignoring the **memory wall** (focusing only on FLOPs).
- Forgetting **reliability/checkpointing** at scale (failures are routine).
- Not stating **tradeoffs** or how the design changes under different constraints.

## References
- Narayanan, D. et al. (2021). *Efficient Large-Scale LM Training (3D parallelism).* arXiv:2104.04473.
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Grattafiori, A. et al. (2024). *Llama 3 Herd.* arXiv:2407.21783.
- Kwon, W. et al. (2023). *vLLM / PagedAttention.* arXiv:2309.06180.
