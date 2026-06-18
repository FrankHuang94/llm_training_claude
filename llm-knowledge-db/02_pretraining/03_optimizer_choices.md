# Optimizer Choices

> **Last Updated:** 2026-06-18
> **Related Files:** [Training Stability](02_training_stability.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md) · [Data Mixing & Curriculum](../01_data/03_data_mixing_and_curriculum.md)
> **Key Papers:** Kingma & Ba 2014 Adam ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) · Loshchilov & Hutter 2017 AdamW ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) · Chen et al. 2023 Lion ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)) · Liu et al. 2023 Sophia ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)) · Jordan et al. 2024 Muon

## Overview
The optimizer maps gradients to parameter updates, and at LLM scale this choice affects convergence speed, stability, and final loss — translating directly into millions of dollars of compute. Despite a decade of proposed alternatives, **AdamW remains the overwhelming default** for transformer pretraining. Understanding *why* Adam dominates, where it fails, and what the credible challengers (Sophia, Muon) offer is a common interview theme, as is the **learning-rate schedule**, which is arguably as important as the optimizer itself.

## Core Concepts
**Adam / AdamW.** Adam maintains exponential moving averages of the gradient (first moment $m_t$) and its square (second moment $v_t$):
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
with bias correction $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$, and update $\theta_t=\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. The per-coordinate adaptive scaling handles the heterogeneous gradient magnitudes across a transformer's parameters. **AdamW** decouples weight decay from the gradient (applying $-\eta\lambda\theta$ directly) rather than folding it into $g$ — the correct formulation for adaptive optimizers and now universal. Typical: $\beta_1=0.9$, $\beta_2=0.95$ (lower than vision's 0.999 for LLMs), $\epsilon=10^{-8}$, weight decay 0.1.

**Adafactor.** Factorizes the second-moment matrix to $O(n+m)$ instead of $O(nm)$ memory (sublinear), trading a little quality for large memory savings; used when optimizer-state memory is binding (T5, some TPU runs).

**Lion.** Sign-based: update uses the *sign* of an interpolated momentum, $\theta\mathrel{-}=\eta\,\text{sign}(\beta_1 m + (1-\beta_1)g)$, storing only momentum (half Adam's state). Competitive, memory-light, but more sensitive to LR/weight-decay tuning.

**Sophia.** A lightweight **second-order** method using a diagonal Hessian estimate to precondition updates, clipping per-coordinate; claims ~2× faster pretraining in some settings, though robustness at frontier scale is debated.

**Muon.** (2024) Optimizes 2D weight matrices by **orthogonalizing** the momentum update via Newton-Schulz iteration (approximating $UV^\top$ from the momentum's SVD), giving more uniform update geometry. Showed strong wall-clock speedups on GPT-2-scale and growing adoption (e.g., reported use in large 2025 runs / Kimi); typically combined with AdamW for non-matrix params (embeddings, norms).

**Learning-rate schedules.**
- **Cosine decay**: warmup then cosine-anneal to a small final LR; the long-time default. Requires committing to a total step count in advance.
- **WSD (Warmup-Stable-Decay)**: warmup → long *constant* LR (stable phase) → short rapid decay. Decouples the schedule from total length (you can decay from any checkpoint), enables **continued/branching training**, and pairs naturally with **data annealing** (decay LR while shifting to high-quality data). Increasingly preferred (MiniCPM, DeepSeek, OLMo 2).

## Key Challenges
- **Optimizer-state memory.** Adam stores $2\times$ params in fp32 ($8\Psi$ bytes) — a major contributor to the memory wall (see [Distributed Training](01_distributed_training.md)).
- **Hyperparameter sensitivity.** LR, $\beta_2$, weight decay, warmup length all interact; bad choices cause spikes or slow convergence.
- **Scale generalization of new optimizers.** Many alternatives win at small scale but fail to robustly beat AdamW at 100B+; the bar is high.
- **Schedule rigidity.** Cosine requires a fixed horizon; changing token budget mid-run is awkward (motivating WSD).

## Solutions & Current Best Practices
**AdamW with $\beta_2=0.95$, weight decay 0.1, gradient clipping 1.0, LR warmup + cosine or WSD** is the safe frontier default. Use **μP** to transfer LR across scales (see [Training Stability](02_training_stability.md)). Adopt **WSD** when you want checkpoint-flexible decay and clean annealing. **Muon/Sophia** are credible efficiency upgrades being validated at scale but are not yet universal. Partition optimizer state with ZeRO/FSDP to fit memory.

## Lab Perspectives
- **OpenAI/Anthropic/Meta** use AdamW variants as the production default.
- **Google** has long used **Adafactor** on TPUs for memory efficiency (T5, PaLM-era), alongside Adam.
- **DeepSeek/MiniCPM/OLMo** popularized **WSD** schedules and published μP/hyperparameter scaling.
- **Frontier 2024–2025 runs** increasingly experiment with **Muon** for matrix parameters.

## Latest Developments (2023–2026)
**Muon** is the most discussed new optimizer (orthogonalized updates; strong wall-clock gains, adopted in some large 2025 models). **WSD** schedules and "**mini-batch warmup → constant → anneal**" recipes are mainstream. Research on **hyperparameter scaling laws** (optimal LR/batch-size vs compute, DeepSeek/Cerebras) reduces tuning cost. fp8 training raises new questions about optimizer-state precision.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does Adam dominate over SGD for transformers?"** Per-coordinate adaptivity handles heterogeneous gradient scales; SGD needs heavy tuning and underperforms here.
- **"AdamW vs Adam — what's the difference and why does it matter?"** Decoupled weight decay; correct regularization for adaptive methods.
- **"Cosine vs WSD — tradeoffs?"** Fixed horizon vs checkpoint-flexible decay + annealing synergy.
- **"How much memory does Adam add, and how do you manage it?"** $\sim 8\Psi$ bytes fp32 states; partition via ZeRO/FSDP, or use Adafactor/Lion.

## Open Problems
Why Adam works so well (vs SGD) for transformers lacks a complete theory (recent work links it to heavy-tailed/heterogeneous gradients and loss-landscape conditioning). Whether second-order (Sophia) or orthogonalized (Muon) methods robustly beat AdamW at the largest scales is unresolved. Optimal joint scaling of LR, batch size, and schedule remains an active empirical frontier.

## References
- Kingma, D., Ba, J. (2014). *Adam.* arXiv:1412.6980.
- Loshchilov, I., Hutter, F. (2017). *Decoupled Weight Decay (AdamW).* arXiv:1711.05101.
- Chen, X. et al. (2023). *Symbolic Discovery of Optimization Algorithms (Lion).* arXiv:2302.06675.
- Liu, H. et al. (2023). *Sophia.* arXiv:2305.14342.
- Hu, S. et al. (2024). *MiniCPM (WSD schedule).* arXiv:2404.06395.
