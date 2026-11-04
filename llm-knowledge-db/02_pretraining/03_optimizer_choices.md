# Optimizer Choices

> **Last Updated:** 2026-11-04
> **Related Files:** [Training Stability](02_training_stability.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md) · [Data Mixing & Curriculum](../01_data/03_data_mixing_and_curriculum.md)
> **Key Papers:** Kingma & Ba 2014 Adam ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) · Loshchilov & Hutter 2017 AdamW ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) · Chen et al. 2023 Lion ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)) · Liu et al. 2023 Sophia ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)) · Jordan et al. 2024 Muon · Liu et al. 2025 Muon is Scalable ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)) · Wen et al. 2025 Fantastic Pretraining Optimizers ([arXiv:2509.02046](https://arxiv.org/abs/2509.02046))

## Overview
The optimizer maps gradients to parameter updates, and at LLM scale this choice affects convergence speed, stability, and final loss — translating directly into millions of dollars of compute. For most of the past decade **AdamW was the overwhelming default** for transformer pretraining, and it remains the reference baseline. Since 2025, however, **Muon** (orthogonalized updates for matrix parameters) has become production-grade at trillion-parameter scale in open frontier models (Kimi K2, GLM-5, DeepSeek-V4 by report), while careful benchmarks show its true advantage over a *well-tuned* AdamW is modest. Understanding *why* Adam dominated, where it fails, what Muon-family methods actually buy, and the **learning-rate schedule** (arguably as important as the optimizer itself) is a common interview theme.

## Core Concepts
**Adam / AdamW.** Adam maintains exponential moving averages of the gradient (first moment $m_t$) and its square (second moment $v_t$):
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
with bias correction $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$, and update $\theta_t=\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. The per-coordinate adaptive scaling handles the heterogeneous gradient magnitudes across a transformer's parameters. **AdamW** decouples weight decay from the gradient (applying $-\eta\lambda\theta$ directly) rather than folding it into $g$ — the correct formulation for adaptive optimizers and now universal. Typical: $\beta_1=0.9$, $\beta_2=0.95$ (lower than vision's 0.999 for LLMs), $\epsilon=10^{-8}$, weight decay 0.1.

**Adafactor.** Factorizes the second-moment matrix to $O(n+m)$ instead of $O(nm)$ memory (sublinear), trading a little quality for large memory savings; used when optimizer-state memory is binding (T5, some TPU runs).

**Lion.** Sign-based: update uses the *sign* of an interpolated momentum, $\theta\mathrel{-}=\eta\,\text{sign}(\beta_1 m + (1-\beta_1)g)$, storing only momentum (half Adam's state). Competitive, memory-light, but more sensitive to LR/weight-decay tuning.

**Sophia.** A lightweight **second-order** method using a diagonal Hessian estimate to precondition updates, clipping per-coordinate; claims ~2× faster pretraining in some settings, though robustness at frontier scale is debated.

**Muon.** (2024) Optimizes 2D weight matrices by **orthogonalizing** the momentum update via Newton-Schulz iteration (approximating $UV^\top$ from the momentum's SVD $M=U\Sigma V^\top$), giving more uniform update geometry across singular directions. It stores only momentum (one state buffer) and is combined with AdamW for non-matrix params (embeddings, output head, norms). Making it work at scale requires (Moonlight, 2025): **decoupled weight decay** and an **update-RMS match** to AdamW (scaling the orthogonalized update by $\approx 0.2\sqrt{\max(A,B)}$ for an $A\times B$ matrix) so AdamW-tuned LRs transfer; at very large scale, **QK-Clip** to stop attention-logit growth (see [Training Stability](02_training_stability.md)).

**Dion.** A communication-efficient orthonormalized optimizer: amortized power iteration on the momentum orthonormalizes only a rank-$r$ subspace, so it works directly on FSDP/TP-sharded matrices without a full-matrix gather (see [Distributed Training](01_distributed_training.md)).

**Learning-rate schedules.**
- **Cosine decay**: warmup then cosine-anneal to a small final LR; the long-time default. Requires committing to a total step count in advance.
- **WSD (Warmup-Stable-Decay)**: warmup → long *constant* LR (stable phase) → short rapid decay. Decouples the schedule from total length (you can decay from any checkpoint), enables **continued/branching training**, and pairs naturally with **data annealing** (decay LR while shifting to high-quality data). Increasingly preferred (MiniCPM, DeepSeek, OLMo 2).

## Key Challenges
- **Optimizer-state memory.** Adam stores $2\times$ params in fp32 ($8\Psi$ bytes) — a major contributor to the memory wall (see [Distributed Training](01_distributed_training.md)).
- **Hyperparameter sensitivity.** LR, $\beta_2$, weight decay, warmup length all interact; bad choices cause spikes or slow convergence.
- **Scale generalization of new optimizers.** Many alternatives win at small scale but fail to robustly beat AdamW at 100B+; the bar is high, and **under-tuned baselines** have inflated many reported speedups.
- **Schedule rigidity.** Cosine requires a fixed horizon; changing token budget mid-run is awkward (motivating WSD).
- **Sharding matrix optimizers.** Orthogonalization needs whole matrices, which conflicts with ZeRO/FSDP sharding and adds communication.

## Solutions & Current Best Practices
**AdamW with $\beta_2=0.95$, weight decay 0.1, gradient clipping 1.0, LR warmup + cosine or WSD** remains the safe default. **Muon (or MuonClip) for 2D hidden matrices + AdamW for embeddings/head/norms** is now a proven frontier option — with weight decay, RMS-matched update scale, μP-style width scaling (including ~$1/\text{width}$ weight decay) so gains survive scale-up, and QK-Clip or QK-norm for logit control. Benchmark any new optimizer against a **per-optimizer-tuned** AdamW. Use **μP** to transfer LR across scales (see [Training Stability](02_training_stability.md)). Adopt **WSD** when you want checkpoint-flexible decay and clean annealing. Partition optimizer state with ZeRO/FSDP to fit memory.

## Lab Perspectives
- **OpenAI/Anthropic/Meta** have historically used AdamW variants as the production default (current recipes undisclosed).
- **Google** has long used **Adafactor** on TPUs for memory efficiency (T5, PaLM-era), alongside Adam.
- **DeepSeek/MiniCPM/OLMo** popularized **WSD** schedules and published μP/hyperparameter scaling; DeepSeek-V4 reportedly uses **Muon** as its main optimizer, with AdamW for embeddings, output heads and RMSNorm weights.
- **Moonshot** made Muon scalable (Moonlight) and trained Kimi K2 (1T) with **MuonClip**; **Zhipu** (GLM-5) is also reported to train with Muon.
- **Microsoft** proposed **Dion** for sharding-native orthonormalized updates; **Stanford/Marin** published the most rigorous optimizer benchmark to date.

## Latest Developments (2023–Nov 2026)
**Muon** is the most discussed new optimizer (orthogonalized updates; strong wall-clock gains, adopted in some large 2025 models). **WSD** schedules and "**mini-batch warmup → constant → anneal**" recipes are mainstream. Research on **hyperparameter scaling laws** (optimal LR/batch-size vs compute, DeepSeek/Cerebras) reduces tuning cost. fp8 training raises new questions about optimizer-state precision.

**Mid-2025 → Nov 2026 update:**
- **Muon is Scalable / Moonlight (2025-02).** Adding weight decay and matching per-parameter update RMS to AdamW makes Muon scale; ~2× compute efficiency (≈52% of AdamW's FLOPs) reported, demonstrated on a 16B-A3B MoE trained on 5.7T tokens ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)).
- **MuonClip at 1T (2025-07).** Kimi K2 trains a 1T-parameter MoE on 15.5T tokens with Muon + per-head QK-Clip and zero loss spikes ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)).
- **Dion (2025-04, rev. 2025-09).** Power-iteration-based rank-$r$ orthonormalization compatible with FSDP/TP sharding — the communication-aware answer to Newton–Schulz's full-matrix requirement ([arXiv:2504.05295](https://arxiv.org/abs/2504.05295)).
- **Reality check: Fantastic Pretraining Optimizers (2025-09, ICLR 2026).** Benchmarking 11 optimizers with rigorous per-optimizer tuning, no speedup over a well-tuned AdamW exceeds **1.4×**, and the speedup **shrinks with model size**; matrix-preconditioned methods (Muon, SOAP) win at small scale ([arXiv:2509.02046](https://arxiv.org/abs/2509.02046)).
- **Hyperparameter transfer decides whether gains survive (2025-12).** With μP and $1/\text{width}$ weight decay, Muon keeps ~1.4× over AdamW from 190M to 1.4B; with the wrong scaling the gain disappears ([arXiv:2512.05620](https://arxiv.org/abs/2512.05620)).
- **Production adoption (2026).** Frontier open models reportedly trained with Muon include Kimi K2, GLM-5 and DeepSeek-V4 (per "Spectral Scaling Laws of Muon", [arXiv:2606.04058](https://arxiv.org/abs/2606.04058)); μP-style LR transfer for MLA + Muon MoE has been proposed (2026-08, single source, [arXiv:2608.20061](https://arxiv.org/abs/2608.20061)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does Adam dominate over SGD for transformers?"** Per-coordinate adaptivity handles heterogeneous gradient scales; SGD needs heavy tuning and underperforms here.
- **"AdamW vs Adam — what's the difference and why does it matter?"** Decoupled weight decay; correct regularization for adaptive methods.
- **"Cosine vs WSD — tradeoffs?"** Fixed horizon vs checkpoint-flexible decay + annealing synergy.
- **"How much memory does Adam add, and how do you manage it?"** $\sim 8\Psi$ bytes fp32 states; partition via ZeRO/FSDP, or use Adafactor/Lion/Muon (one buffer).
- **"What does Muon do, and what did it take to make it work at 1T?"** Orthogonalize the momentum ($\approx UV^\top$) via Newton–Schulz; add weight decay, RMS-match the update to AdamW, keep AdamW for embeddings/head/norms, and add QK-Clip against logit explosion.
- **"A paper claims 2× over AdamW — how do you evaluate it?"** Check per-optimizer tuning of the baseline, scale trend (gains shrink with size), HP-transfer rules (μP, weight-decay scaling), wall-clock and communication cost, not just steps.

## Open Problems
Why Adam works so well (vs SGD) for transformers lacks a complete theory (recent work links it to heavy-tailed/heterogeneous gradients and loss-landscape conditioning). Whether orthogonalized (Muon/Dion) or second-order (SOAP, Sophia) methods keep a material advantage at the largest scales is contested: production adoption is rising even as fair benchmarks show shrinking gains with size. Spectral scaling laws for Muon, optimal joint scaling of LR, batch size, weight decay and schedule, and communication-efficient matrix optimizers remain active frontiers.

## References
- Kingma, D., Ba, J. (2014). *Adam.* arXiv:1412.6980.
- Loshchilov, I., Hutter, F. (2017). *Decoupled Weight Decay (AdamW).* arXiv:1711.05101.
- Chen, X. et al. (2023). *Symbolic Discovery of Optimization Algorithms (Lion).* arXiv:2302.06675.
- Liu, H. et al. (2023). *Sophia.* arXiv:2305.14342.
- Hu, S. et al. (2024). *MiniCPM (WSD schedule).* arXiv:2404.06395.
- Liu, J. et al. (2025). *Muon is Scalable for LLM Training.* arXiv:2502.16982.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence (MuonClip).* arXiv:2507.20534.
- Ahn, K. et al. (2025). *Dion: Distributed Orthonormalized Updates.* arXiv:2504.05295.
- Wen, K. et al. (2025). *Fantastic Pretraining Optimizers and Where to Find Them.* arXiv:2509.02046.
