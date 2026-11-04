# Mixed Precision and Low-Precision Training

> **Last Updated:** 2026-11-04
> **Related Files:** [Training Stability](02_training_stability.md) · [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md) · [Quantization Techniques](../06_efficiency/02_quantization_techniques.md)
> **Key Papers:** Micikevicius et al. 2017 Mixed Precision Training ([arXiv:1710.03740](https://arxiv.org/abs/1710.03740)) · Kalamkar et al. 2019 bfloat16 ([arXiv:1905.12322](https://arxiv.org/abs/1905.12322)) · Micikevicius et al. 2022 FP8 Formats ([arXiv:2209.05433](https://arxiv.org/abs/2209.05433)) · DeepSeek-AI 2024 DeepSeek-V3 (FP8 training) ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) · NVIDIA 2025 Pretraining LLMs with NVFP4 ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149))

## Overview
Training in reduced numerical precision is one of the highest-leverage efficiency techniques: it roughly halves (fp16/bf16) or quarters (fp8) memory and doubles/quadruples tensor-core throughput versus fp32, with little to no quality loss when done carefully. The journey fp32 → fp16/bf16 → fp8 → **fp4** tracks GPU hardware generations (Volta → Ampere → Hopper → Blackwell → Rubin); by 2026 4-bit (NVFP4) pretraining has moved from research to production-scale runs. This file covers the *training-time* numerics; **inference quantization** (GPTQ/AWQ/int4) is in [Quantization Techniques](../06_efficiency/02_quantization_techniques.md).

## Core Concepts
**Floating-point formats.** A float = sign + exponent (range) + mantissa (precision).
- **fp32**: 1/8/23 — baseline.
- **fp16**: 1/5/10 — high precision but *narrow exponent* (max ~65504); prone to overflow/underflow → needs loss scaling.
- **bf16**: 1/8/7 — *same exponent range as fp32* (huge dynamic range) but only 7 mantissa bits. Trades precision for range; the **default for LLM pretraining** because range matters more than precision for stability, and it needs no loss scaling.
- **fp8**: E4M3 (1/4/3, more precision) and E5M2 (1/5/2, more range). Hopper/Blackwell tensor cores support fp8 matmul.
- **fp4 (E2M1)**: only 16 representable values, usable only with fine-grained **block scaling** (microscaling).

**Microscaling (block-scaled) formats.** Each small block of elements shares a scale factor. **MXFP4** (OCP MX): 32-element blocks with a power-of-two (E8M0) scale. **NVFP4** (Blackwell): 16-element blocks with an **E4M3** scale *plus* a per-tensor FP32 scale — smaller blocks and non-power-of-two scales track local dynamic range more tightly, which is why NVFP4 beats MXFP4 in training. DeepSeek's **UE8M0** FP8 scales are likewise exponent-only (power-of-two) block scales.

**Mixed-precision training (the classic recipe).** Keep an **fp32 master copy** of weights; do forward/backward in fp16/bf16; accumulate matmuls in fp32. For **fp16**, apply **loss scaling**: multiply the loss by a large factor $S$ before backprop so small gradients don't underflow to zero, then unscale before the optimizer step (dynamic loss scaling adjusts $S$ on overflow). **bf16** typically skips loss scaling thanks to its range.

**FP8 training.** Cast matmul inputs to fp8 with **per-tensor (or finer) scaling factors** that track each tensor's dynamic range (delayed scaling uses a history of amax). NVIDIA's **Transformer Engine** automates this. Sensitive components (optimizer states, master weights, certain reductions, sometimes attention) stay in higher precision. fp8 gives ~2× throughput over bf16 on H100.

**FP4 training (NVFP4 recipe).** Quantize GEMM inputs to NVFP4 with **2D block scaling** for weights (so forward and backward see consistent quantization), **random Hadamard transforms** on the weight-gradient (Wgrad) GEMM inputs to spread outliers, **stochastic rounding** on gradients to keep them unbiased, and a few sensitive layers (mostly the final blocks) in BF16.

## Key Challenges
- **Overflow/underflow (fp16).** Narrow range causes NaNs/inf or silent gradient underflow; loss scaling is required and fiddly.
- **Precision loss (bf16/fp8).** Few mantissa bits cause rounding errors that can accumulate (e.g., in large reductions, LayerNorm, softmax).
- **FP8 stability.** Outliers and shifting dynamic ranges make per-tensor scaling delicate; some layers/operations must remain higher precision.
- **FP4 bias and flush-to-zero.** With 16 levels per block, deterministic rounding biases gradients and tiny values flush to zero (Nemotron 3 kept Mamba output projections in MXFP8 for this reason).
- **Determinism & reproducibility.** Low precision plus non-deterministic reductions complicate debugging of instabilities — and in RL, BF16 rounding makes trainer and sampler disagree.

## Solutions & Current Best Practices
**bf16 remains the safe pretraining default** (fp32 master weights + bf16 compute, fp32 accumulation, no loss scaling). For **fp16** (older hardware), use **dynamic loss scaling**. For **fp8** (Hopper+), use **Transformer Engine** with per-tensor/blockwise scaling, keep optimizer states and sensitive ops in higher precision, and validate stability carefully. For **fp4** (Blackwell+), use **NVFP4** with 2D weight scaling, Hadamard transforms on Wgrad, stochastic rounding, and BF16 for the most sensitive layers. Always **accumulate gradients/reductions in fp32**, and keep **LayerNorm/softmax/router** computations in higher precision. For **RL post-training**, consider **FP16** (more mantissa) to keep rollout and trainer log-probs consistent.

## Lab Perspectives
- **Google** co-developed **bfloat16** for TPUs and uses it pervasively; Ironwood (TPU v7) is specified at ~4.6 PF FP8 per chip.
- **NVIDIA** drives fp8 via Transformer Engine and now **NVFP4**: Nemotron 3 Super (120B-A12B) and Ultra were pretrained in NVFP4; Rubin is marketed at 50 PF NVFP4.
- **DeepSeek** demonstrated **production fp8 pretraining** for V3 with fine-grained (tile/block) quantization, then UE8M0 FP8 scales (V3.1) aimed at domestic chips and **FP4 QAT** for routed experts in V4.
- **Moonshot / OpenAI** push 4-bit into post-training and release: Kimi K2 Thinking uses INT4 weight-only QAT; gpt-oss ships MoE weights in MXFP4.
- **Meta/OpenAI/Anthropic** do not disclose current pretraining precision; bf16 with fp8 for the largest matmuls was the 2024-era norm.

## Latest Developments (2023–Nov 2026)
**FP8 training went mainstream** for frontier models (DeepSeek-V3's detailed fp8 recipe is a key reference). **Blackwell** hardware adds fp4 and microscaling (MX) formats; research explores **fp4 / sub-8-bit training** with block scaling and stochastic rounding. **Precision scaling laws** (Kumar et al. 2024) characterize the loss penalty of low-precision training/inference as a function of bits, model size, and tokens — informing how aggressively to quantize.

**Mid-2025 → Nov 2026 update:**
- **Longest public FP4 run (2025-09, v2 2026-03).** NVIDIA trains a 12B hybrid Mamba-Transformer on **10T tokens in NVFP4** with loss and accuracy close to FP8; MXFP4 needed ~36% more tokens to match ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)).
- **NVFP4 at production scale (2025-12 → 2026-04).** Nemotron 3 Super (120B / 12B active) is pretrained in NVFP4 on ~25T tokens, with a reported <1% loss gap vs BF16; Mamba output projections stayed in MXFP8 because of flush-to-zero ([arXiv:2512.20856](https://arxiv.org/abs/2512.20856), [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)). Checkpoints were released in NVFP4, FP8 and BF16.
- **Hardware-targeted FP8 scales (2025-08).** DeepSeek-V3.1's **UE8M0 FP8** (exponent-only, power-of-two block scales) was described as "tailored for next-generation domestically built chips" — export controls now shape number formats.
- **FP4 QAT for experts (2026-04).** DeepSeek-V4 applies FP4 quantization-aware training to routed-expert weights, and its sparse-attention indexer reportedly computes in FP4; experts dominate parameter memory, so this is where 4-bit pays most.
- **4-bit post-training and release (2025-08 → 2026-07).** gpt-oss MoE weights are post-trained into MXFP4 (~4.25 bits/param); Kimi K2 Thinking applies INT4 weight-only QAT to MoE components during post-training (~2× generation speed, all benchmarks reported at INT4); Kimi K3 weights ship as MXFP4.
- **FP16 returns for RL (2025-10).** BF16's 7-bit mantissa makes rollout-engine and trainer log-probs diverge; FP16 RL largely removes the training–inference mismatch without importance-sampling patches ([arXiv:2510.26788](https://arxiv.org/abs/2510.26788)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"bf16 vs fp16 for training — which and why?"** bf16: fp32-range exponent → stable, no loss scaling; fp16: more precision but needs loss scaling.
- **"Why keep an fp32 master copy of weights?"** Small updates would be lost to rounding in 16-bit; master weights preserve them.
- **"What is loss scaling and when is it needed?"** Scale loss to keep small fp16 gradients above underflow; dynamic scaling backs off on overflow.
- **"What makes fp8 training hard?"** Per-tensor dynamic-range scaling, outliers, and keeping sensitive ops higher precision.
- **"Why does NVFP4 train better than MXFP4, and what else does an FP4 recipe need?"** 16- vs 32-element blocks and E4M3 vs power-of-two scales (+ per-tensor FP32 scale) give finer range tracking; add Hadamard transforms on Wgrad, stochastic rounding on gradients, 2D weight scaling, and BF16 for sensitive layers.
- **"Why might you use FP16 rather than BF16 for RL?"** The policy-gradient ratio needs sampler and trainer to agree on log-probs; more mantissa bits reduce mismatch.

## Open Problems
How low *training* precision can go (below fp4, or fp4 for all tensors including attention and optimizer states) without quality loss is unresolved. Principled per-tensor vs per-block scaling, outlier handling, which layers must stay high-precision and why, and the precise precision-scaling-law tradeoffs at frontier scale (e.g., whether FP4 penalties grow with token count) are active research, as is co-design of formats (MX/NVFP4/UE8M0) with optimizer numerics, non-NVIDIA hardware, and RL training–inference consistency.

## References
- Micikevicius, P. et al. (2017). *Mixed Precision Training.* arXiv:1710.03740.
- Kalamkar, D. et al. (2019). *A Study of BFLOAT16.* arXiv:1905.12322.
- Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning.* arXiv:2209.05433.
- DeepSeek-AI (2024). *DeepSeek-V3 (FP8 training).* arXiv:2412.19437.
- Kumar, T. et al. (2024). *Scaling Laws for Precision.* arXiv:2411.04330.
- Rouhani, B. et al. (2023). *Microscaling Data Formats for Deep Learning.* arXiv:2310.10537.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
- NVIDIA (2025). *Nemotron 3 (hybrid Mamba-Transformer MoE).* arXiv:2512.20856.
- Sea AI Lab (2025). *Defeating the Training-Inference Mismatch via FP16.* arXiv:2510.26788.
