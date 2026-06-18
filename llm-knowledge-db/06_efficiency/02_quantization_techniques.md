# Quantization Techniques

> **Last Updated:** 2026-06-18
> **Related Files:** [Mixed Precision & Low-Precision Training](../02_pretraining/04_mixed_precision_and_quantization.md) · [LoRA & Variants](01_LoRA_and_variants.md) · [Speculative Decoding](04_speculative_decoding.md)
> **Key Papers:** Dettmers et al. 2022 LLM.int8() ([arXiv:2208.07339](https://arxiv.org/abs/2208.07339)) · Frantar et al. 2022 GPTQ ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) · Lin et al. 2023 AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) · Xiao et al. 2022 SmoothQuant ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438))

## Overview
Quantization reduces the numerical precision of model weights (and sometimes activations and KV cache) from 16-bit down to 8, 4, or even fewer bits, shrinking memory footprint and bandwidth — the dominant costs of LLM **inference**. Because autoregressive decoding is **memory-bandwidth-bound** (each token requires reading all weights from HBM), halving weight precision can nearly double throughput and lets large models fit on fewer/smaller GPUs. This file covers **inference quantization** (PTQ/QAT, GPTQ/AWQ/SmoothQuant); **training** precision (fp8/bf16) is in [Mixed Precision](../02_pretraining/04_mixed_precision_and_quantization.md).

## Core Concepts
**The mechanics.** Map a high-precision range to a low-bit grid: $x_q = \text{round}(x/s) + z$ with scale $s$ and zero-point $z$. **Symmetric** (z=0) vs **asymmetric**; **per-tensor** vs **per-channel** vs **per-group** (e.g., groups of 128 weights share a scale — finer granularity, better accuracy). Dequantize for compute, or use low-bit kernels.

**PTQ vs QAT.**
- **Post-Training Quantization (PTQ):** quantize an already-trained model, typically with a small **calibration set** to set scales — fast, no retraining, the standard for LLMs.
- **Quantization-Aware Training (QAT):** simulate quantization during training/fine-tuning so the model adapts to it — higher accuracy at low bits but expensive; used for aggressive (≤4-bit) targets.

**The outlier problem.** LLM activations contain **systematic outlier channels** with huge magnitudes that dominate the quantization range and destroy accuracy if naively quantized. Most successful methods are strategies for handling outliers:
- **LLM.int8()** (Dettmers): keep outlier dimensions in fp16, quantize the rest to int8 (mixed-precision decomposition). Accurate but with overhead.
- **SmoothQuant** (Xiao): migrate the quantization difficulty from activations to weights via a per-channel scaling that "smooths" activation outliers — enables W8A8 (8-bit weights *and* activations).
- **GPTQ** (Frantar): one-shot **weight-only** PTQ using approximate second-order (Hessian) information to quantize weights column-by-column while compensating for error — excellent 4-bit weight quality, the workhorse for W4 inference.
- **AWQ** (Lin): **Activation-aware Weight Quantization** — protect the ~1% of weights that are *salient* (those multiplying large-activation channels) by per-channel scaling, no backprop needed; fast, hardware-friendly, strong 4-bit results.

**Weight-only vs weight+activation.** **Weight-only** (W4A16: 4-bit weights, 16-bit activations — GPTQ/AWQ) is the most common because decoding is bandwidth-bound by *weights*; great memory/throughput wins with minimal accuracy loss. **Weight+activation** (W8A8 — SmoothQuant) speeds up *compute* (int8 tensor cores) and helps compute-bound (prefill/large-batch) regimes. **KV-cache quantization** (int8/fp8/int4) cuts the dominant long-context memory cost.

## Key Challenges
- **Activation outliers.** The central obstacle to low-bit activation quantization.
- **Accuracy at ≤4 bits.** Below 4-bit (3-bit, 2-bit, 1.58-bit), accuracy degrades sharply without QAT or clever schemes.
- **Hardware support.** Speedups require kernels/datatypes the hardware supports (int8/fp8 tensor cores, int4 via custom kernels like Marlin); arbitrary bit-widths may not accelerate.
- **Calibration sensitivity.** PTQ quality depends on representative calibration data and group size.
- **Task-dependent degradation.** Reasoning/long-context tasks are more sensitive to quantization than simple ones.

## Solutions & Current Best Practices
**Default for serving large models:** **weight-only 4-bit PTQ (AWQ or GPTQ)** with group size 128, often near-lossless on most tasks; **W8A8 (SmoothQuant)** or **fp8** for compute-bound/large-batch serving; **KV-cache quantization** (fp8/int8) for long context. For aggressive low-bit, use **QAT** or LoRA-on-quantized (**QLoRA**, see [LoRA & Variants](01_LoRA_and_variants.md)). Validate on **reasoning/long-context** evals, not just perplexity. Use optimized kernels (Marlin, Machete) to realize speedups. Inference engines (vLLM, TensorRT-LLM, llama.cpp/GGUF) ship these built-in.

## Lab Perspectives
- **NVIDIA** drives fp8/int8 (TensorRT-LLM, Transformer Engine) and Blackwell fp4.
- **Academia (UW/MIT/IST)** produced LLM.int8(), GPTQ, AWQ, SmoothQuant — the core methods.
- **DeepSeek/Microsoft** explored low-bit *training* (fp8) and extreme quantization (BitNet **1.58-bit** ternary weights — Microsoft, a QAT-from-scratch approach).
- **Open community** (llama.cpp/GGUF, bitsandbytes) made k-bit quantization ubiquitous for local inference.

## Latest Developments (2023–2026)
**4-bit weight-only became the default** for serving large open models with minimal loss. **fp8 inference** (and Blackwell **fp4**/microscaling) matured. **KV-cache quantization** grew essential for long context. **BitNet b1.58** (ternary {-1,0,1} weights, trained QAT) demonstrated extreme low-bit training is viable, hinting at hardware co-design. **Precision scaling laws** (Kumar et al. 2024) quantified the accuracy cost of low-bit inference vs model size and training tokens — notably, **more-trained models are more sensitive to PTQ**.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does quantization speed up LLM inference?"** Decoding is memory-bandwidth-bound; fewer bits/weight → less HBM traffic → higher throughput.
- **"PTQ vs QAT?"** Post-hoc + calibration (fast, standard) vs train-aware (accurate at low bits, costly).
- **"What's the outlier problem and how do GPTQ/AWQ/SmoothQuant handle it?"** Huge activation channels; GPTQ Hessian-compensated weight quant, AWQ protects salient weights, SmoothQuant migrates difficulty to weights.
- **"Weight-only vs weight+activation quantization — when each?"** W-only (W4A16) for bandwidth-bound decoding; W8A8 for compute-bound prefill/large-batch.
- **"What about the KV cache?"** Quantize it (fp8/int8) — dominant long-context memory cost.

## Open Problems
Robust **sub-4-bit** quantization without QAT, fully solving **activation/KV outliers**, and **hardware-datatype co-design** (fp4, ternary) are open. The accuracy-vs-bits-vs-training-tokens frontier (why heavily-trained models resist PTQ) and quantization's disproportionate impact on **reasoning/long-context** remain active research.

## References
- Dettmers, T. et al. (2022). *LLM.int8().* arXiv:2208.07339.
- Frantar, E. et al. (2022). *GPTQ.* arXiv:2210.17323.
- Lin, J. et al. (2023). *AWQ.* arXiv:2306.00978.
- Xiao, G. et al. (2022). *SmoothQuant.* arXiv:2211.10438.
- Ma, S. et al. (2024). *BitNet b1.58.* arXiv:2402.17764.
