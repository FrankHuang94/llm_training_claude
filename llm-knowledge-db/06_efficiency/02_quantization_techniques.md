# Quantization Techniques

> **Last Updated:** 2026-11-04
> **Related Files:** [Mixed Precision & Low-Precision Training](../02_pretraining/04_mixed_precision_and_quantization.md) · [LoRA & Variants](01_LoRA_and_variants.md) · [Speculative Decoding](04_speculative_decoding.md)
> **Key Papers:** Dettmers et al. 2022 LLM.int8() ([arXiv:2208.07339](https://arxiv.org/abs/2208.07339)) · Frantar et al. 2022 GPTQ ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) · Lin et al. 2023 AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) · Xiao et al. 2022 SmoothQuant ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438))

## Overview
Quantization reduces the numerical precision of model weights (and sometimes activations and KV cache) from 16-bit down to 8, 4, or even fewer bits, shrinking memory footprint and bandwidth — the dominant costs of LLM **inference**. Because autoregressive decoding is **memory-bandwidth-bound** (each token requires reading all model weights — or, for MoE, all *active* expert weights — from HBM), halving weight precision can nearly double throughput and lets large models fit on fewer/smaller GPUs. Since 2025 the frontier of open models has shifted from community post-training quantization toward **lab-shipped, quantization-aware 4-bit checkpoints** (MXFP4, INT4, FP4). This file covers **inference quantization** (PTQ/QAT, GPTQ/AWQ/SmoothQuant, block formats); **training** precision (fp8/bf16/NVFP4) is in [Mixed Precision](../02_pretraining/04_mixed_precision_and_quantization.md).

## Core Concepts
**The mechanics.** Map a high-precision range to a low-bit grid: $x_q = \text{round}(x/s) + z$ with scale $s$ and zero-point $z$. **Symmetric** (z=0) vs **asymmetric**; **per-tensor** vs **per-channel** vs **per-group** (e.g., groups of 128 weights share a scale — finer granularity, better accuracy). Dequantize for compute, or use low-bit kernels.

**Block floating-point (microscaling) formats.** Hardware-native per-group quantization: **MXFP4** stores E2M1 elements in 32-element blocks with a shared power-of-two (E8M0) scale (~4.25 bits/param effective); **NVFP4** uses 16-element blocks with an E4M3 scale plus a per-tensor FP32 scale, tracking local range more finely. Blackwell/Rubin tensor cores compute on these directly, so 4-bit is now a *compute* format, not just a storage format.

**PTQ vs QAT.**
- **Post-Training Quantization (PTQ):** quantize an already-trained model, typically with a small **calibration set** to set scales — fast, no retraining, the standard for community LLM quantization.
- **Quantization-Aware Training (QAT):** simulate quantization during training/fine-tuning so the model adapts to it — higher accuracy at low bits but expensive; used for aggressive (≤4-bit) targets and now by labs for their released MoE experts.

**The outlier problem.** LLM activations contain **systematic outlier channels** with huge magnitudes that dominate the quantization range and destroy accuracy if naively quantized. Most successful methods are strategies for handling outliers:
- **LLM.int8()** (Dettmers): keep outlier dimensions in fp16, quantize the rest to int8 (mixed-precision decomposition). Accurate but with overhead.
- **SmoothQuant** (Xiao): migrate the quantization difficulty from activations to weights via a per-channel scaling that "smooths" activation outliers — enables W8A8 (8-bit weights *and* activations).
- **GPTQ** (Frantar): one-shot **weight-only** PTQ using approximate second-order (Hessian) information to quantize weights column-by-column while compensating for error — excellent 4-bit weight quality, the workhorse for W4 inference.
- **AWQ** (Lin): **Activation-aware Weight Quantization** — protect the ~1% of weights that are *salient* (those multiplying large-activation channels) by per-channel scaling, no backprop needed; fast, hardware-friendly, strong 4-bit results.
- **Rotations (Hadamard transforms)** spread outliers across channels before quantizing; the same trick underpins NVFP4 training.

**Weight-only vs weight+activation.** **Weight-only** (W4A16: 4-bit weights, 16-bit activations — GPTQ/AWQ) is the most common because decoding is bandwidth-bound by *weights*; great memory/throughput wins with minimal accuracy loss. **Weight+activation** (W8A8 — SmoothQuant; FP8; W4A4 in NVFP4) speeds up *compute* and helps compute-bound (prefill/large-batch) regimes. **KV-cache quantization** (int8/fp8/int4) cuts the dominant long-context memory cost.

## Key Challenges
- **Activation outliers.** The central obstacle to low-bit activation quantization.
- **Accuracy at ≤4 bits.** Below 4-bit (3-bit, 2-bit, 1.58-bit), accuracy degrades sharply without QAT or clever schemes.
- **Hardware support.** Speedups require kernels/datatypes the hardware supports (int8/fp8 tensor cores, MXFP4/NVFP4 on Blackwell, int4 via custom kernels like Marlin); arbitrary bit-widths may not accelerate.
- **Calibration sensitivity.** PTQ quality depends on representative calibration data and group size.
- **Task-dependent degradation.** Reasoning/long-context tasks are more sensitive to quantization than simple ones.
- **Recurrent state precision.** Linear-attention/SSM hybrids carry a compressed state across the whole sequence; MiniMax reported linear attention was sensitive to low-precision state storage.

## Solutions & Current Best Practices
**Default for serving large models:** use the **lab's QAT low-bit checkpoint when one exists** (MXFP4/INT4/FP4 experts); otherwise **weight-only 4-bit PTQ (AWQ or GPTQ)** with group size 128, often near-lossless on most tasks; **W8A8 (SmoothQuant)**, **fp8**, or **NVFP4** for compute-bound/large-batch serving on supporting hardware; **KV-cache quantization** (fp8/int8) for long context. For aggressive low-bit, use **QAT** or LoRA-on-quantized (**QLoRA**, see [LoRA & Variants](01_LoRA_and_variants.md)). Keep attention, router, and recurrent-state paths at higher precision. Validate on **reasoning/long-context** evals, not just perplexity. Use optimized kernels (Marlin, Machete) to realize speedups. Inference engines (vLLM, SGLang, TensorRT-LLM, llama.cpp/GGUF) ship these built-in.

## Lab Perspectives
- **NVIDIA** drives fp8/int8 (TensorRT-LLM, Transformer Engine) and **NVFP4** (Blackwell/Rubin), releasing Nemotron 3 Super checkpoints in NVFP4, FP8 and BF16.
- **Academia (UW/MIT/IST)** produced LLM.int8(), GPTQ, AWQ, SmoothQuant — the core methods.
- **OpenAI** shipped gpt-oss with MoE weights in **MXFP4**; **Moonshot** shipped Kimi K2 Thinking with **INT4 QAT** and Kimi K3 in MXFP4; **DeepSeek** applies **FP4 QAT** to V4's routed experts.
- **DeepSeek/Microsoft** explored low-bit *training* (fp8) and extreme quantization (BitNet **1.58-bit** ternary weights — Microsoft, a QAT-from-scratch approach).
- **Open community** (llama.cpp/GGUF, bitsandbytes) made k-bit quantization ubiquitous for local inference.

## Latest Developments (2023–Nov 2026)
**4-bit weight-only became the default** for serving large open models with minimal loss. **fp8 inference** (and Blackwell **fp4**/microscaling) matured. **KV-cache quantization** grew essential for long context. **BitNet b1.58** (ternary {-1,0,1} weights, trained QAT) demonstrated extreme low-bit training is viable, hinting at hardware co-design. **Precision scaling laws** (Kumar et al. 2024) quantified the accuracy cost of low-bit inference vs model size and training tokens — notably, **more-trained models are more sensitive to PTQ**.

**Mid-2025 → Nov 2026 update:**
- **MXFP4 as a release format (2025-08).** gpt-oss-120b's MoE weights are post-trained into MXFP4 (~4.25 bits/param), so a 116.8B-total model fits on a single 80 GB GPU.
- **INT4 QAT for a 1T reasoning model (2025-11).** Kimi K2 Thinking applies INT4 weight-only QAT to MoE components during post-training, giving native INT4 inference at ~2× generation speed with all benchmarks reported at INT4; LMSYS published an open reproduction of the INT4-QAT recipe (2026-01).
- **FP4 QAT for experts (2026-04).** DeepSeek-V4 trains routed-expert weights with FP4 QAT (its sparse-attention indexer reportedly also computes in FP4); Kimi K3 (2026-07) ships its 2.8T weights as MXFP4 safetensors (~1.56 TB). Quantizing experts — most of the parameters but a small share of per-token FLOPs — is the highest-leverage target.
- **NVFP4 end to end (2025-09 → 2026-03).** NVIDIA's NVFP4 recipe (16-element blocks, E4M3 + FP32 scales, Hadamard transforms) works for 10T-token *pretraining* ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)); Rubin is quoted at 50 PF NVFP4 for inference, making W4A4 FP serving mainstream on new hardware.
- **Hardware-specific scale formats (2025-08).** DeepSeek's UE8M0 FP8 (power-of-two scales) targets domestic Chinese accelerators — quantization formats now follow supply chains.
- **KV-cache compression beyond scalar quantization (2026).** Hybrid quantize-or-evict policies (ARKV, ~4× less memory at ~97% accuracy, [arXiv:2603.08727](https://arxiv.org/abs/2603.08727)) and KV transform coding (KVTC, ICLR 2026: PCA + adaptive quantization + entropy coding, up to 20×) report large gains (abstract-level claims).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does quantization speed up LLM inference?"** Decoding is memory-bandwidth-bound; fewer bits/weight → less HBM traffic → higher throughput.
- **"PTQ vs QAT?"** Post-hoc + calibration (fast, standard) vs train-aware (accurate at low bits, costly).
- **"What's the outlier problem and how do GPTQ/AWQ/SmoothQuant handle it?"** Huge activation channels; GPTQ Hessian-compensated weight quant, AWQ protects salient weights, SmoothQuant migrates difficulty to weights.
- **"Weight-only vs weight+activation quantization — when each?"** W-only (W4A16) for bandwidth-bound decoding; W8A8 for compute-bound prefill/large-batch.
- **"What about the KV cache?"** Quantize it (fp8/int8) — dominant long-context memory cost.
- **"Why do labs now ship QAT 4-bit MoE experts instead of leaving quantization to users?"** Experts dominate memory but not FLOPs; QAT lets the model adapt to 4-bit (critical for long reasoning, where PTQ errors compound), and benchmarks can be reported at the shipped precision.
- **"MXFP4 vs NVFP4?"** 32- vs 16-element blocks; power-of-two E8M0 vs E4M3 (+FP32 per-tensor) scales — NVFP4 tracks local range more tightly at slightly higher scale overhead.

## Open Problems
Robust **sub-4-bit** quantization without QAT, fully solving **activation/KV outliers**, and **hardware-datatype co-design** (fp4, ternary) are open. The accuracy-vs-bits-vs-training-tokens frontier (why heavily-trained models resist PTQ), quantization's disproportionate impact on **reasoning/long-context**, low-precision storage of linear-attention/SSM state, and fair comparison when the released checkpoint *is* the quantized model remain active research.

## References
- Dettmers, T. et al. (2022). *LLM.int8().* arXiv:2208.07339.
- Frantar, E. et al. (2022). *GPTQ.* arXiv:2210.17323.
- Lin, J. et al. (2023). *AWQ.* arXiv:2306.00978.
- Xiao, G. et al. (2022). *SmoothQuant.* arXiv:2211.10438.
- Ma, S. et al. (2024). *BitNet b1.58.* arXiv:2402.17764.
- Rouhani, B. et al. (2023). *Microscaling Data Formats for Deep Learning.* arXiv:2310.10537.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
