# Scaling Laws

> **Last Updated:** 2026-11-04
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
Labs fit **their own** scaling laws on their data/architecture, run IsoFLOP sweeps (vary $N$ at fixed $C$, find the loss-minimizing model), and extrapolate. For deployment, they optimize **inference-adjusted** scaling (overtrain). MoE has its own laws relating *active* vs *total* parameters; for MoE, use $C\approx6N_{\text{active}}D$ (e.g., Kimi K2: $6\times32\text{B}\times15.5\text{T}\approx3\times10^{24}$ FLOPs despite 1T total parameters). 2025–26 open MoEs sit at roughly 500–700 tokens per *active* parameter (K2 15.5T/32B; GLM-5 28.5T/40B) — heavily overtrained by Chinchilla standards. Data-constrained scaling laws (Muennighoff et al. 2023) quantify the value of repeating data: up to ~4 epochs is nearly as good as fresh data, then returns diminish.

## Lab Perspectives
- **OpenAI** originated scaling laws (Kaplan) and o-series "inference scaling laws" — loss/accuracy improving with test-time compute (see [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)). It attributes GPT-6 Astra's (2026-09) gains to large-scale pretraining plus long-horizon RL, i.e., pretraining scale is back alongside RL scale.
- **DeepMind** produced Chinchilla and continues IsoFLOP-driven design (Gemini, Gemma).
- **Meta** epitomized the overtraining paradigm with the Llama series; its closed Muse Spark (2026-04) is pitched on compute efficiency (Llama-4-Maverick-level reasoning "using far less compute", vendor claim).
- **Moonshot** reports optimizer-level efficiency gains (Muon ≈2× compute efficiency vs AdamW in Moonlight; Kimi K3 claims ~2.5× scaling efficiency over K2) — claims that careful third-party benchmarks temper (below).
- **DeepSeek** publishes MoE scaling laws and hyperparameter scaling laws (batch size, LR vs compute).

## Latest Developments (2023–Nov 2026)
Three shifts dominate. **(1) Inference-time scaling**: o1/o3/R1 show accuracy scaling with reasoning tokens at test time, a new axis beyond train compute. **(2) Data-constrained scaling**: with the public-text wall, work focuses on repetition, synthetic data, and quality-adjusted "effective data." **(3) Distillation scaling laws** (2024–2025) characterizing when training a small model on a large teacher's outputs beats training from scratch. Precision scaling laws (training in fp8/int4) are also emerging.

**Mid-2025 → Nov 2026 update:**
- **RL compute has a predictable (sigmoidal) law (2025-10; arXiv:2510.13786, ICLR 2026 oral).** ScaleRL (>400K GPU-hours) fits performance vs RL compute with a saturating sigmoid — asymptote $A$ and efficiency exponent $B$ — and early-run fits extrapolate to 100K-GPU-hour runs. Most design choices move the efficiency $B$ rather than the ceiling $A$. Unlike pretraining power laws, the RL curve saturates, so the asymptote is the quantity to compare.
- **Optimizer "speedups" shrink with scale (2025-09; arXiv:2509.02046).** Benchmarking 11 optimizers with per-optimizer tuning, no speedup over well-tuned AdamW exceeds 1.4×, and it decreases with model size; with muP and $1/\text{width}$ weight decay Muon keeps ~1.4× from 190M to 1.4B (arXiv:2512.05620), but the gain vanishes with wrong hyperparameter scaling. Speedup claims must be stated as a function of scale.
- **MoE allocation laws (2026-03; arXiv:2603.10379).** The compute-optimal expert:attention compute ratio grows as a power law in total compute and depends on sparsity. Related ICLR 2026 work (secondary) finds optimal sparsity differs for memorization vs reasoning, with reasoning sometimes preferring denser models.
- **Precision enters the law as an effective-token tax (2025-09; arXiv:2509.25149).** A 12B model pretrained in NVFP4 on 10T tokens tracked FP8 loss, while MXFP4 needed ~36% more tokens to match — i.e., number format changes $D_{\text{eff}}$. Nemotron 3 Super (120B-A12B) was pretrained in NVFP4 on ~25T tokens.
- **Effective data beyond repetition (2026-04/05).** SynPro (arXiv:2605.17849) reports 3.7–5.2× the effective tokens of plain repetition in a 10%-of-Chinchilla data regime via RL-trained rephrasers; FinePhrase (arXiv:2604.13977) finds rephrasing generators above ~1B parameters add nothing.
- **Midtraining sets post-RL ceilings (2026-03; arXiv:2603.17074).** Adding science data at midtraining yields +17–28 points GPQA-Diamond after RL, while changing the RL data mix moves results <2 points; midtraining restructures >90% of weights while RL sparsely updates ~5%.
- **Capability trend lines.** METR's 50% time horizon originally doubled every ~7 months (arXiv:2503.14499); the Jan 2026 TH1.1 suite puts post-2023 doubling at ~4–4.5 months, with frontier horizons ≥16 h by mid-2026 where the suite saturates (secondary).

## Interview Angles
> 💡 **What labs actually ask:**
- **"State and explain the Chinchilla scaling law."** The $L(N,D)$ form, ~20 tokens/param, and the Gopher-was-undertrained punchline.
- **"Why did Kaplan and Chinchilla disagree?"** LR schedule per horizon + parameter counting.
- **"You have $10^{24}$ FLOPs — how big a model and how much data?"** Use $C=6ND$ and $N\approx D/20$ to solve.
- **"Why overtrain a small model past compute-optimal?"** Inference cost dominates lifetime cost.
- **"Estimate training FLOPs for a 1T-total / 32B-active MoE on 15.5T tokens."** $6\cdot N_{\text{active}}\cdot D\approx3\times10^{24}$; note memory and communication scale with *total* parameters.
- **"How would you decide whether an RL recipe is worth scaling?"** Fit a sigmoid to early compute–reward points (ScaleRL); compare asymptotes $A$, not just early slopes.

## Open Problems
The **emergence debate** is unsettled: Wei et al. (2022) report capabilities appearing abruptly at scale, while Schaeffer et al. (2023) argue these are artifacts of discontinuous/nonlinear metrics — under smooth metrics the improvement is gradual. Whether truly discontinuous capability jumps exist matters for safety forecasting. Also open: scaling laws for reasoning/agentic capability (RL asymptotes, environment-count scaling), the true value of synthetic data, joint laws over total vs active parameters and precision, and whether the data wall caps the current paradigm.

## References
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal LLMs (Chinchilla).* arXiv:2203.15556.
- Wei, J. et al. (2022). *Emergent Abilities of LLMs.* arXiv:2206.07682.
- Schaeffer, R. et al. (2023). *Are Emergent Abilities a Mirage?* arXiv:2304.15004.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained LMs.* arXiv:2305.16264.
- Kwa, T. et al. (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- Wen, K. et al. (2025). *Fantastic Pretraining Optimizers and Where to Find Them.* arXiv:2509.02046.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
- Khatri, D. et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs.* arXiv:2510.13786.
