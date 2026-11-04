# Test-Time Training

> **Last Updated:** 2026-11-04
> **Related Files:** [Non-Transformer Architectures](04_neuromorphic_and_non_transformer.md) · [Diffusion & Hybrid LMs](06_diffusion_and_hybrid_language_models.md) · [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md) · [Continual Learning](01_continual_learning.md)
> **Key Papers:** Sun et al. 2020 Test-Time Training ([arXiv:1909.13231](https://arxiv.org/abs/1909.13231)) · Sun et al. 2024 TTT layers / "Learning to (Learn at Test Time)" ([arXiv:2407.04620](https://arxiv.org/abs/2407.04620)) · Gu & Dao 2023 Mamba ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) · Akyürek et al. 2024 TTT for ARC ([arXiv:2411.07279](https://arxiv.org/abs/2411.07279)) · TTT-E2E 2025 ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675))

## Overview
Test-time training (TTT) is the idea that a model can **update its own parameters during inference**, adapting to the specific input or context rather than running purely with frozen weights. It blurs the train/test boundary: instead of (or in addition to) generating more tokens (inference-time *scaling*), the model *learns* from the test instance itself. TTT spans two related threads: (1) **TTT layers** — a new sequence-modeling primitive whose hidden state is itself a small model updated by gradient descent over the sequence (a generalization of RNN/SSM state); and (2) **test-time adaptation** for hard reasoning (fine-tune on augmented versions of the test problem before answering, as in record ARC-AGI results). Both are active frontiers connecting architecture, continual learning, and reasoning.

## Core Concepts
**Classic TTT (Sun et al. 2020).** At test time, adapt the model to each input via a **self-supervised auxiliary task** (e.g., rotation prediction) on that input before making the main prediction — improving robustness to distribution shift. The model takes a few gradient steps per test example.

**TTT layers (Sun et al. 2024) — the sequence-model view.** Reframe the **hidden state of a sequence model as a (small) model** whose weights are updated by **self-supervised gradient descent at each timestep**. Where an RNN compresses history into a fixed vector and attention keeps all of it (quadratic), a TTT layer compresses history into the *weights of a fast inner model* updated online — giving **linear complexity** with an *expressive*, learnable memory. The "update rule" is itself learned (meta-learning: "learning to learn at test time"). TTT-Linear/TTT-MLP are competitive with Mamba and Transformers on long context, with the appeal that the memory can keep improving with more tokens.

**TTT for reasoning (test-time adaptation).** For very hard, out-of-distribution tasks (e.g., **ARC-AGI** abstract reasoning), generate augmented variants of the test problem (transformations, leave-one-out demonstrations), **fine-tune the model on these at inference**, then solve — yielding large accuracy jumps (state-of-the-art on ARC). This is "learning from the test problem's own structure."

**Relation to SSMs and fast weights.** TTT layers generalize **linear attention / state-space models** (the recurrent state update is a special case) and revive the classic "**fast weights**" idea (Schmidhuber) — weights that change quickly as a function of input, separate from slow base weights. The cleanest bridge is the **gated delta rule**: $S_t=\alpha_t S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top$ is exactly one step of online gradient descent (step size $\beta_t$, weight decay via $\alpha_t$) on the regression loss $\tfrac12\|S k_t-v_t\|^2$ — a linear TTT layer. Its *overwrite-not-accumulate* behavior with bounded state is why Gated DeltaNet/KDA beat additive linear attention, and these layers now run in production hybrids (Qwen3-Next, Kimi Linear/K3).

**End-to-end TTT for long context.** Instead of a special recurrent layer, take a standard (sliding-window) Transformer and let it update a subset of its own MLP weights by next-token gradient steps on the context at inference; meta-learn the initialization (gradients of gradients) so those steps are maximally useful. Long context becomes *continual learning on the prompt*.

## Key Challenges
- **Compute cost at inference.** Gradient updates per token (TTT layers) or per problem (reasoning TTT) add significant inference cost/latency.
- **Stability.** Online updates risk divergence; the inner-loop learning rate/update rule must be carefully designed/meta-learned.
- **Hardware efficiency.** TTT layers need hardware-efficient inner-loop kernels (mini-batch TTT) to be competitive with optimized attention/Mamba kernels.
- **Forgetting / scope.** Test-time updates are usually transient (reset per input); persistent updates reintroduce continual-learning problems (see [Continual Learning](01_continual_learning.md)).
- **Generality.** Reasoning-TTT gains are strongest on narrow, structured OOD tasks; general applicability is unproven — and interactive benchmarks (ARC-AGI-3) require exploration, which per-instance fine-tuning alone does not supply.
- **Serving.** Per-request weight updates break batching/prefix-caching assumptions of inference engines; meta-learned TTT also needs second-order training.

## Solutions & Current Best Practices
For **architecture**: delta-rule layers (Gated DeltaNet, KDA) — the production-grade descendants of TTT layers — are deployed in **~3:1 hybrids with full attention**; richer TTT layers (TTT-MLP, Titans/ATLAS memories) and **TTT-E2E** remain research-stage alternatives for **long context with learnable, linear-cost memory**; use hardware-aware chunked/mini-batch inner loops. For **hard reasoning**: **test-time fine-tuning on augmented test instances** is a powerful, if expensive, technique (ARC-AGI). In practice, most deployed "test-time compute" is still **inference-time *scaling*** (more tokens/search, see [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md)) rather than parameter updates, but TTT is a fast-growing complementary direction.

## Lab Perspectives
- **Academia (Stanford/CMU/MIT):** TTT layers (Sun et al.), TTT-for-ARC (Akyürek et al.) — the research drivers.
- **SSM/efficient-architecture community:** views TTT as part of the linear-attention/SSM continuum (Mamba, DeltaNet, Gated DeltaNet, RWKV-7 "expressive state").
- **Frontier labs:** primarily pursue inference-time *scaling* (reasoning tokens/search, now productized as effort levels and adaptive thinking); TTT-style parameter adaptation is more exploratory. **Google** is the most active closed lab (Titans → ATLAS → Nested Learning/HOPE).
- **Chinese open labs (Qwen, Moonshot):** ship delta-rule fast-weight layers at scale (Qwen3-Next/3.5 Gated DeltaNet; Kimi Linear and Kimi K3 with 69 KDA + 24 MLA layers).
- **ARC Prize / reasoning researchers:** test-time adaptation as a route to abstract, OOD generalization.

## Latest Developments (2023–Nov 2026)
**TTT layers** emerged as a credible linear-complexity sequence primitive with *learnable* memory, competitive with Mamba/Transformers on long context. **Test-time training for ARC-AGI** produced standout abstract-reasoning results, spotlighting parameter-level test-time adaptation. The broader **"expressive recurrent state"** trend (Gated DeltaNet, RWKV-7, Titans) overlaps heavily with TTT, unifying SSMs, fast weights, and online learning. Debate: is parameter-level test-time learning the next axis beyond token-level inference scaling?

**Mid-2025 → Nov 2026 update:**
- **ATLAS (2025-05, Google; ICML 2026).** Titans successor whose memory is optimized over a *sliding window* of past tokens (Omega rule) instead of only the latest token, with higher-capacity memory and "DeepTransformers"; reports +80% on BABILong at 10M tokens ([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)).
- **Delta-rule fast weights reach production (2025-09 → 2026-07).** Qwen3-Next (3 Gated DeltaNet : 1 gated attention), Kimi Linear's KDA (channel-wise gates; claims to beat full attention under fair comparison, up to 6× decode at 1M; [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) and Kimi K3 (2.8T total, 69 KDA + 24 MLA layers) make TTT-style online regression a frontier-scale primitive.
- **Nested Learning / HOPE (2025-11, Google, NeurIPS 2025).** Architecture and optimizer as nested learners at different update frequencies; HOPE adds a continuum memory system of MLPs updated at multiple rates — TTT generalized to a hierarchy of timescales.
- **TTT-E2E (2025-12).** Sliding-window Transformer + test-time next-token gradient steps on some MLPs + meta-learned init; at 3B it scales with context like full attention while keeping constant per-token latency, 2.7× faster than full attention at 128K ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)).
- **Mamba-3 (2026-03, ICLR 2026 oral).** Exponential-trapezoidal discretization, complex-valued transitions for state tracking, and MIMO updates for decode arithmetic intensity; Mamba-2 perplexity at half the state size ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)).
- **ARC moves past static puzzles (2026).** ARC-AGI-2 is near-saturated by frontier API models (90–95%, aggregator figures) and the efficiency-constrained Kaggle track reportedly topped 88% (single source); **ARC-AGI-3** (2026-03), an interactive game benchmark, had every frontier model under 1% at launch — shifting the test from per-instance adaptation to exploration and goal inference.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What is test-time training and how does it differ from inference-time scaling?"** Update *parameters* from the test input vs generate more *tokens*/search with frozen weights.
- **"Explain TTT layers as a sequence model."** Hidden state = a small model updated by self-supervised GD per token → linear cost, expressive learnable memory; generalizes RNN/SSM/linear attention.
- **"Why did TTT help on ARC-AGI?"** Fine-tuning on augmented test instances exploits the problem's own structure for OOD generalization.
- **"How does TTT relate to fast weights / SSMs?"** It's the modern, meta-learned form of fast weights; SSM recurrence is a special case.
- **"Show that the gated delta rule is a TTT layer."** Gradient of $\tfrac12\|Sk_t-v_t\|^2$ w.r.t. $S$ is $(Sk_t-v_t)k_t^\top$; one SGD step with rate $\beta_t$ after decay $\alpha_t$ gives $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ (keys are $L_2$-normalized for stability).

## Open Problems
Whether **parameter-level test-time learning** becomes a mainstream axis (vs token-level scaling), making TTT **hardware-efficient** at frontier scale, ensuring **stability** of online updates, and extending reasoning-TTT beyond narrow structured tasks are open. The deeper question — how to unify slow (pretraining) and fast (test-time) learning into robust continual/lifelong systems — remains unsolved.

## References
- Sun, Y. et al. (2020). *Test-Time Training.* arXiv:1909.13231.
- Sun, Y. et al. (2024). *Learning to (Learn at Test Time): TTT layers.* arXiv:2407.04620.
- Akyürek, E. et al. (2024). *The Surprising Effectiveness of TTT for Abstract Reasoning.* arXiv:2411.07279.
- Gu, A., Dao, T. (2023). *Mamba.* arXiv:2312.00752.
- Yang, S. et al. (2024). *Gated DeltaNet.* arXiv:2412.06464.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
- Stanford/NVIDIA et al. (2025). *End-to-End Test-Time Training for Long Context (TTT-E2E).* arXiv:2512.23675.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- CMU/Princeton/Together AI (Gu, Dao et al.) (2026). *Mamba-3.* arXiv:2603.15569.
