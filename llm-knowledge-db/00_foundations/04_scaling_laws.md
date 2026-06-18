# Scaling Laws

> **Last Updated:** 2026-06-18
> **Related Files:** [Compute & Memory Efficiency](../02_pretraining/05_compute_and_memory_efficiency.md) · [Data Mixing & Curriculum](../01_data/03_data_mixing_and_curriculum.md) · [Model Architecture Variants](05_model_architecture_variants.md)
> **Key Papers:** Kaplan et al. 2020 ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)) · Hoffmann et al. 2022 Chinchilla ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) · Schaeffer et al. 2023 "Emergent Abilities a Mirage?" ([arXiv:2304.15004](https://arxiv.org/abs/2304.15004)) · Wei et al. 2022 Emergent Abilities ([arXiv:2206.07682](https://arxiv.org/abs/2206.07682))

## Overview
Scaling laws are the empirical regularities relating model loss to compute $C$, parameters $N$, and data $D$. Their discovery — that test loss falls as a smooth **power law** across many orders of magnitude — transformed LLM development from craft into a predictable engineering discipline: you can forecast a large model's loss from small-scale runs and allocate a fixed compute budget optimally. Scaling laws are the intellectual justification for spending hundreds of millions of dollars on a single training run.

Two landmark results frame the field. **Kaplan et al. (2020, OpenAI)** established the power-law form and argued model size should grow faster than data. **Hoffmann et al. (2022, DeepMind "Chinchilla")** corrected this, showing compute-optimal training requires scaling $N$ and $D$ roughly *equally*, and that prior models (GPT-3, Gopher) were badly under-trained on data.

## Core Concepts
**Power-law form.** Loss decomposes into a reducible term plus an irreducible entropy floor $E$:
$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
with empirically $\alpha\approx0.34$, $\beta\approx0.28$ in the Chinchilla fit. Each term is a power law; the irreducible $E$ reflects the entropy of language.

**Compute-optimal allocation.** Training compute is $C\approx 6ND$ (forward+backward, 6 FLOPs/param/token). Minimizing $L$ subject to fixed $C$ yields optimal $N^\star\propto C^{a}$, $D^\star\propto C^{b}$ with $a\approx b\approx0.5$. The famous heuristic: **~20 tokens per parameter** at the compute-optimal point. Chinchilla (70B params, 1.4T tokens) outperformed Gopher (280B, 300B tokens) using the *same* compute — a 4× smaller, better-trained model.

**Kaplan vs Chinchilla discrepancy.** Kaplan's analysis under-counted because it (a) didn't tune the learning-rate schedule to each horizon and (b) excluded embedding parameters inconsistently, biasing toward "bigger models, less data." Chinchilla's corrected methodology gave the equal-scaling result that the field adopted.

**Overtraining (the LLaMA paradigm).** Compute-optimal minimizes *training* cost, but for a model you will *serve* billions of times, **inference cost dominates**. So labs deliberately "overtrain" smaller models far past 20 tokens/param (Llama 3 8B saw ~15T tokens — ~1800 tokens/param) to get a small, cheap-to-serve model that is strong for its size. The optimum shifts when you optimize total (train + inference) cost.

## Key Challenges
- **Methodology sensitivity.** Results depend critically on LR schedules, parameter counting, and fitting procedure — small errors flip conclusions (Kaplan vs Chinchilla).
- **Data exhaustion.** At Chinchilla ratios, frontier models are running out of high-quality public text ("the data wall"), pushing toward synthetic data and repeated epochs.
- **Downstream vs loss.** Scaling laws predict *loss*, not *capabilities*; the loss→benchmark mapping is noisy.
- **Transfer of laws.** Laws differ by architecture (MoE, SSM), data quality, and modality; each regime needs re-fitting.

## Solutions & Current Best Practices
Labs fit **their own** scaling laws on their data/architecture, run IsoFLOP sweeps (vary $N$ at fixed $C$, find the loss-minimizing model), and extrapolate. For deployment, they optimize **inference-adjusted** scaling (overtrain). MoE has its own laws relating *active* vs *total* parameters. Data-constrained scaling laws (Muennighoff et al. 2023) quantify the value of repeating data: up to ~4 epochs is nearly as good as fresh data, then returns diminish.

## Lab Perspectives
- **OpenAI** originated scaling laws (Kaplan) and o-series "inference scaling laws" — loss/accuracy improving with test-time compute (see [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)).
- **DeepMind** produced Chinchilla and continues IsoFLOP-driven design (Gemini, Gemma).
- **Meta** epitomizes the overtraining paradigm with the Llama series for serving efficiency.
- **DeepSeek** publishes MoE scaling laws and hyperparameter scaling laws (batch size, LR vs compute).

## Latest Developments (2023–2026)
Three shifts dominate. **(1) Inference-time scaling**: o1/o3/R1 show accuracy scaling with reasoning tokens at test time, a new axis beyond train compute. **(2) Data-constrained scaling**: with the public-text wall, work focuses on repetition, synthetic data, and quality-adjusted "effective data." **(3) Distillation scaling laws** (2024–2025) characterizing when training a small model on a large teacher's outputs beats training from scratch. Precision scaling laws (training in fp8/int4) are also emerging.

## Interview Angles
> 💡 **What labs actually ask:**
- **"State and explain the Chinchilla scaling law."** The $L(N,D)$ form, ~20 tokens/param, and the Gopher-was-undertrained punchline.
- **"Why did Kaplan and Chinchilla disagree?"** LR schedule per horizon + parameter counting.
- **"You have $10^{24}$ FLOPs — how big a model and how much data?"** Use $C=6ND$ and $N\approx D/20$ to solve.
- **"Why overtrain a small model past compute-optimal?"** Inference cost dominates lifetime cost.

## Open Problems
The **emergence debate** is unsettled: Wei et al. (2022) report capabilities appearing abruptly at scale, while Schaeffer et al. (2023) argue these are artifacts of discontinuous/nonlinear metrics — under smooth metrics the improvement is gradual. Whether truly discontinuous capability jumps exist matters for safety forecasting. Also open: scaling laws for reasoning/agentic capability, the true value of synthetic data, and whether the data wall caps the current paradigm.

## References
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal LLMs (Chinchilla).* arXiv:2203.15556.
- Wei, J. et al. (2022). *Emergent Abilities of LLMs.* arXiv:2206.07682.
- Schaeffer, R. et al. (2023). *Are Emergent Abilities a Mirage?* arXiv:2304.15004.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained LMs.* arXiv:2305.16264.
