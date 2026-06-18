# Mixed Precision and Low-Precision Training

> **Last Updated:** 2026-06-18
> **Related Files:** [Training Stability](02_training_stability.md) · [Compute & Memory Efficiency](05_compute_and_memory_efficiency.md) · [Quantization Techniques](../06_efficiency/02_quantization_techniques.md)
> **Key Papers:** Micikevicius et al. 2017 Mixed Precision Training ([arXiv:1710.03740](https://arxiv.org/abs/1710.03740)) · Kalamkar et al. 2019 bfloat16 ([arXiv:1905.12322](https://arxiv.org/abs/1905.12322)) · Micikevicius et al. 2022 FP8 Formats ([arXiv:2209.05433](https://arxiv.org/abs/2209.05433)) · DeepSeek-AI 2024 DeepSeek-V3 (FP8 training) ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))

## Overview
Training in reduced numerical precision is one of the highest-leverage efficiency techniques: it roughly halves (fp16/bf16) or quarters (fp8) memory and doubles/quadruples tensor-core throughput versus fp32, with little to no quality loss when done carefully. The journey fp32 → fp16/bf16 → fp8 (and research into fp4) tracks GPU hardware generations (Volta → Ampere → Hopper → Blackwell). This file covers the *training-time* numerics; **inference quantization** (GPTQ/AWQ/int4) is in [Quantization Techniques](../06_efficiency/02_quantization_techniques.md).

## Core Concepts
**Floating-point formats.** A float = sign + exponent (range) + mantissa (precision).
- **fp32**: 1/8/23 — baseline.
- **fp16**: 1/5/10 — high precision but *narrow exponent* (max ~65504); prone to overflow/underflow → needs loss scaling.
- **bf16**: 1/8/7 — *same exponent range as fp32* (huge dynamic range) but only 7 mantissa bits. Trades precision for range; the **default for LLM pretraining** because range matters more than precision for stability, and it needs no loss scaling.
- **fp8**: E4M3 (1/4/3, more precision) and E5M2 (1/5/2, more range). Hopper/Blackwell tensor cores support fp8 matmul.

**Mixed-precision training (the classic recipe).** Keep an **fp32 master copy** of weights; do forward/backward in fp16/bf16; accumulate matmuls in fp32. For **fp16**, apply **loss scaling**: multiply the loss by a large factor $S$ before backprop so small gradients don't underflow to zero, then unscale before the optimizer step (dynamic loss scaling adjusts $S$ on overflow). **bf16** typically skips loss scaling thanks to its range.

**FP8 training.** Cast matmul inputs to fp8 with **per-tensor (or finer) scaling factors** that track each tensor's dynamic range (delayed scaling uses a history of amax). NVIDIA's **Transformer Engine** automates this. Sensitive components (optimizer states, master weights, certain reductions, sometimes attention) stay in higher precision. fp8 gives ~2× throughput over bf16 on H100.

## Key Challenges
- **Overflow/underflow (fp16).** Narrow range causes NaNs/inf or silent gradient underflow; loss scaling is required and fiddly.
- **Precision loss (bf16/fp8).** Few mantissa bits cause rounding errors that can accumulate (e.g., in large reductions, LayerNorm, softmax).
- **FP8 stability.** Outliers and shifting dynamic ranges make per-tensor scaling delicate; some layers/operations must remain higher precision.
- **Determinism & reproducibility.** Low precision plus non-deterministic reductions complicate debugging of instabilities.

## Solutions & Current Best Practices
**bf16 is the pretraining default** (fp32 master weights + bf16 compute, fp32 accumulation, no loss scaling). For **fp16** (older hardware), use **dynamic loss scaling**. For **fp8** (Hopper+), use **Transformer Engine** with per-tensor/blockwise scaling, keep optimizer states and sensitive ops in higher precision, and validate stability carefully. Always **accumulate gradients/reductions in fp32**, and keep **LayerNorm/softmax/router** computations in higher precision. Stochastic rounding can help low-bit accumulation.

## Lab Perspectives
- **Google** co-developed **bfloat16** for TPUs and uses it pervasively.
- **NVIDIA** drives fp8 via Transformer Engine; Hopper/Blackwell make fp8 (and fp4) first-class.
- **DeepSeek** demonstrated **production fp8 pretraining** for V3 (a major efficiency milestone under H800 constraints), with fine-grained (tile/block) quantization and selective high-precision.
- **Meta/OpenAI/Anthropic** train primarily in bf16, increasingly experimenting with fp8 for the largest runs.

## Latest Developments (2023–2026)
**FP8 training went mainstream** for frontier models (DeepSeek-V3's detailed fp8 recipe is a key reference). **Blackwell** hardware adds fp4 and microscaling (MX) formats; research explores **fp4 / sub-8-bit training** with block scaling and stochastic rounding. **Precision scaling laws** (Kumar et al. 2024) characterize the loss penalty of low-precision training/inference as a function of bits, model size, and tokens — informing how aggressively to quantize.

## Interview Angles
> 💡 **What labs actually ask:**
- **"bf16 vs fp16 for training — which and why?"** bf16: fp32-range exponent → stable, no loss scaling; fp16: more precision but needs loss scaling.
- **"Why keep an fp32 master copy of weights?"** Small updates would be lost to rounding in 16-bit; master weights preserve them.
- **"What is loss scaling and when is it needed?"** Scale loss to keep small fp16 gradients above underflow; dynamic scaling backs off on overflow.
- **"What makes fp8 training hard?"** Per-tensor dynamic-range scaling, outliers, and keeping sensitive ops higher precision.

## Open Problems
How low can *training* precision go (fp4 and below) without quality loss is unresolved — inference quantizes more aggressively than training tolerates. Principled per-tensor vs per-block scaling, outlier handling, and the precise precision-scaling-law tradeoffs at frontier scale are active research, as is co-design of formats (MX/fp4) with optimizer numerics.

## References
- Micikevicius, P. et al. (2017). *Mixed Precision Training.* arXiv:1710.03740.
- Kalamkar, D. et al. (2019). *A Study of BFLOAT16.* arXiv:1905.12322.
- Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning.* arXiv:2209.05433.
- DeepSeek-AI (2024). *DeepSeek-V3 (FP8 training).* arXiv:2412.19437.
- Kumar, T. et al. (2024). *Scaling Laws for Precision.* arXiv:2411.04330.
