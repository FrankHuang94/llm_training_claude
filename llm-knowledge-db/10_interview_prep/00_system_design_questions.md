# System Design Questions

> **Last Updated:** 2026-11-04
> **Related Files:** [Distributed Training](../02_pretraining/01_distributed_training.md) · [RLHF](../03_posttraining/03_RLHF.md) · [Evaluation & Benchmarks](../05_alignment_and_safety/05_evaluation_and_benchmarks.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Diffusion & Hybrid LMs](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)
> **Key Papers:** Narayanan et al. 2021 (3D parallelism) · Ouyang et al. 2022 InstructGPT · Grattafiori et al. 2024 Llama 3 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) · Khatri et al. 2025 ScaleRL ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786))

## Overview
LLM system-design interviews test whether you can **reason end-to-end about building and operating large models** — making and justifying tradeoffs under compute, memory, data, and reliability constraints. Unlike a single fact question, these are open-ended: the interviewer wants to see structured thinking, awareness of bottlenecks, back-of-envelope math, and knowledge of what real systems do. This file gives a **framework** plus worked outlines for the canonical prompts — including the 2025–2026 additions interviewers now favor: **agentic RL infrastructure**, **serving trillion-parameter MoEs at 1M context**, and **release gating for dual-use capabilities**.

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
- **2026 options to mention:** **Muon** (with weight decay, RMS-matched updates, and **QK-Clip** — Kimi K2 trained 15.5T tokens with zero loss spikes; expect ≲1.4× gains vs well-tuned AdamW), **FP8 → NVFP4** pretraining (Hadamard transforms on Wgrad inputs, stochastic rounding, a few BF16 layers; Nemotron 3 Super at 25T tokens), and on Blackwell-class hardware quote FLOPs in the precision you actually train in.
- **Reliability:** **async sharded checkpointing** (host-RAM snapshot + background flush), auto-detect-and-restart, hot spares (see [Checkpointing](../02_pretraining/06_checkpoint_and_resumption.md)).
- **Metric:** optimize **MFU**; report effective training time.

### "Design an RLHF pipeline from scratch"
- **Stages:** SFT (curated/synthetic demos, loss-masked) → **reward model** (Bradley-Terry, pairwise prefs, length-debiased, ensembled) → **PPO** (policy+ref+RM+value; KL penalty; clip) — or **DPO/GRPO** for simplicity/cost (see [RLHF](../03_posttraining/03_RLHF.md)).
- **Infra:** PPO needs **4 models in memory** + **generation rollouts** (an inference engine, e.g., vLLM) feeding training — design the **actor/learner** split, batching, and KV reuse. Frameworks: TRL/OpenRLHF/verl (most common open), slime (Megatron + SGLang), AReaL.
- **Failure modes:** reward hacking/overoptimization → tune **β**, monitor reward-vs-KL, iterate preference data; length bias → debias RM.
- **Data flywheel:** iterative rounds (re-collect prefs with updated policy); rejection sampling for SFT data.
- **Eval:** preference win-rate, safety/refusal, capability regressions (alignment tax), Arena-style + held-out (see [Evaluation](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)).

### "Design a high-throughput inference/serving system"
- **Bottleneck:** decoding is **memory-bandwidth-bound**; KV cache dominates memory.
- **Techniques:** **continuous batching**, **PagedAttention** (vLLM), **prefix caching**, **quantization** (W4A16/fp8 + KV quant), **speculative decoding** (EAGLE/Medusa), **GQA/MLA**, tensor/pipeline parallel for big models, **disaggregated prefill/decode**.
- **Tradeoffs:** latency (small batch, speculative) vs throughput (large batch); for reasoning models, budget **test-time compute** (now a product knob: effort levels, adaptive thinking).

### "Design an RL post-training system for long-horizon agentic coding" (2026 favorite)
- **Environments are the bottleneck:** thousands of containerized, *verifiable* repo tasks (agent-synthesized tasks, harness randomization), sandboxed with **no open internet** — 2026 eval incidents showed agents exploiting misconfigured network access.
- **Async architecture:** decouple rollout workers (SGLang/vLLM) from trainers (Megatron/FSDP); bound staleness (AReaL staleness-aware PPO; PipelineRL in-flight weight updates; slime). Long trajectories make synchronous RL idle most GPUs.
- **Objective:** GRPO-family with token-level aggregation, **clip-higher** (DAPO) or **IS-weight clipping** (CISPO), or **sequence-level ratios** (GSPO) for MoE stability; **zero-variance filtering** (drop all-pass/all-fail groups); overlong shaping.
- **Training–inference mismatch:** sampler and trainer log-probs differ (BF16 rounding, non-batch-invariant kernels, MoE routing) → quietly off-policy. Remedies: **FP16** RL, **batch-invariant kernels**, truncated IS, **routing replay**, **FP32 LM-head logits**; monitor sampler-vs-trainer KL.
- **Predictability:** fit ScaleRL's sigmoid $R(C)=R_0+(A-R_0)/(1+(C_{\text{mid}}/C)^B)$ on early compute to forecast the asymptote $A$; most tricks change efficiency $B$, not $A$.
- **Safety & reward hacking:** detect test tampering (`sys.exit(0)`, editing tests), hold-out graders, **inoculation prompting**, monitors on trajectories; track CoT monitorability; decide which capabilities (e.g., offensive cyber) to leave out of the environment mix.
- **Cheap stages:** **on-policy distillation** to merge domain experts or recover behaviors; LoRA works for RL even at low rank.

### "Serve a ~1–3T-parameter MoE with 1M-token context"
- **Memory math first:** weights at MXFP4/NVFP4 (~4.25 bits/param → ~0.5 TB per 1T params) need a **rack-scale NVLink domain** (GB300 NVL72-class) with expert parallelism; KV at 1M must be cut by architecture — MLA, **DSA/CSA top-k sparse attention** (DeepSeek-V4 reports ~10% of V3.2's KV at 1M) or **3:1 linear hybrids** (KDA/GDN: only full-attention layers hold KV).
- **Disaggregation:** split prefill (compute-bound; prefill-specialized parts like Rubin CPX) from decode (bandwidth-bound); for cheap-attention models consider **attention–FFN disaggregation**; note DSA's **indexer K cache** complicates KV transfer.
- **Decode speed:** MTP heads as self-drafts / EAGLE-3; INT4/FP4 QAT checkpoints (Kimi K2 Thinking INT4, gpt-oss MXFP4) avoid post-hoc quantization loss.
- **Tradeoffs:** sparse vs hybrid vs full attention (MiniMax reverted to full attention over multi-hop losses), expert count vs all-to-all cost, prefix caching for agent loops.

### "How would you evaluate a new reasoning model?"
- Portfolio: FrontierMath, GPQA, HLE (report with/without tools), LiveCodeBench, **SWE-bench Pro** (OpenAI stopped reporting **SWE-bench Verified** in 2026-02: ≥59.4% of audited hard items had flawed tests, plus contamination), **Terminal-Bench**, OSWorld, **ARC-AGI-3**, METR-style **time horizons**; **contamination checks** + **perturbation tests**; fixed **harness/scaffold** (vendor scaffolds swing SWE-bench Pro by 10–30 points); controlled **inference budget** and **tokens/cost per task**; human review; safety/refusal evals. Beware saturation (~12–18-month benchmark half-life) and **eval awareness** (see [Evaluation](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)).

### "Design the release-gating pipeline for a model with dual-use (e.g., cyber) capability"
- **Capability evals** against framework thresholds (Anthropic RSP v3 ASLs + periodic Risk Reports; OpenAI Preparedness High/Critical; GDM FSF CCLs); pre-release third-party testing (METR, AISIs) in **internet-isolated** sandboxes.
- **Training-time shaping:** decide what goes into the RL mix (Anthropic did not train Opus 5 on cyber tasks); filter data.
- **Deployment stack:** cheap activation probes → escalation classifiers → exchange-level classifiers (Constitutional Classifiers++: ~1% compute, 0.05% refusal rate); **capability routing** of flagged requests to a weaker fallback model; **tiered access** (same weights, different safeguards: Fable vs Mythos; OpenAI Daybreak; verification programs).
- **Monitoring & response:** online action monitors for agents, incident reporting (SB 53: 15 days), rollback/suspension playbooks (Fable 5/Mythos 5 were suspended and redeployed after a safeguard bypass in 2026-06), anti-distillation (preserved thinking).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Estimate the time/cost to train model X on N GPUs."** Always do the $6ND$ + MFU math out loud.
- **"Your run OOMs / has loss spikes / a node dies — what do you do?"** Map to memory (checkpointing/parallelism), stability (clip/z-loss/rollback), reliability (async checkpoint + restart).
- **"Pick a parallelism strategy and justify it by hardware."** TP intra-node, PP inter-node, FSDP outer, SP for context.
- **"DPO vs PPO for your RLHF system — which and why?"** Cost/stability vs on-policy quality.
- **"Your RL run's reward climbs but evals fall / it collapses after N steps — debug it."** Check sampler-vs-trainer log-prob mismatch, entropy collapse (clip-higher, Clip-Cov/KL-Cov), reward hacking (inspect trajectories), staleness in async rollouts, MoE routing drift (GSPO/routing replay).
- **"How much KV cache does 1M context cost, and how would you cut it 10×?"** $2\,n_{\text{layers}}\,n_{kv}\,d_h\,L$ elements × bytes/element; MLA, sparse top-k over compressed KV, linear hybrids, KV quantization.

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
- Khatri, D. et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
- Fu, W. et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* arXiv:2505.24298.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
