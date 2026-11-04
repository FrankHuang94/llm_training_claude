# Context Window Extension

> **Last Updated:** 2026-11-04
> **Related Files:** [Positional Encoding](../00_foundations/03_positional_encoding.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)
> **Key Papers:** Chen et al. 2023 Position Interpolation ([arXiv:2306.15595](https://arxiv.org/abs/2306.15595)) · Peng et al. 2023 YaRN ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071)) · Xiao et al. 2023 StreamingLLM ([arXiv:2309.17453](https://arxiv.org/abs/2309.17453)) · Liu et al. 2023 Lost in the Middle ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) · DeepSeek-AI 2025 DeepSeek Sparse Attention ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556))

## Overview
Context windows have grown from 2K (GPT-3) to 128K–1M+ tokens, enabling whole-codebase reasoning, long-document analysis, and many-shot prompting. By late 2026 **~1M tokens is the de facto frontier spec**: Gemini has offered 1–2M since 1.5, Claude Opus 4.6 introduced 1M (beta), GPT-5.4 and GPT-6 Astra ship ~1.05M, and open models (DeepSeek-V4, Kimi K3, MiniMax-M3, MiMo-V2.5, Nemotron 3) ship 1M natively. The competition has shifted from nominal length to **cost per token at 1M** (FLOPs and KV cache) and **effective** use of that context. Extending context is a multi-part problem: the **positional encoding** must generalize beyond training length, the **attention cost** ($O(n^2)$) must be tamed, the **KV cache** memory must be managed, and the model must actually *use* distant information well (the "lost in the middle" problem). This file focuses on the *techniques* for extension; the *frontier* of memory architectures and ultra-long context is in [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md).

## Core Concepts
**Positional extension (the RoPE-stretching family).** Naively running a RoPE model past its training length fails. Fixes (see [Positional Encoding](../00_foundations/03_positional_encoding.md) for the math):
- **Position Interpolation (PI):** linearly downscale positions into the trained range, then briefly fine-tune — simple and effective.
- **NTK-aware / "theta" scaling:** adjust the RoPE base frequency to interpolate unevenly across dimensions (high-freq dims need different treatment than low-freq).
- **YaRN:** NTK-by-parts + attention-temperature correction — extends 4K→128K with minimal fine-tuning; widely used.
- **LongRoPE:** searches non-uniform per-dimension rescalings to reach 2M+ tokens.
Long-context recipes also **anneal on long documents** during a dedicated training phase. Hybrid designs sidestep part of the problem: in Kimi K3 the global MLA layers reportedly use no positional encoding (NoPE), with position carried by the recurrent KDA layers.

**Attention cost & streaming.**
- **FlashAttention** removes the $O(n^2)$ *memory*; **ring/sequence parallelism** distributes long sequences across devices (see [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md)).
- **Sparse/local patterns** (sliding window, dilated, global+local) reduce compute; **StreamingLLM** showed that keeping a few initial "**attention sink**" tokens + a recent window enables (near-)infinite streaming generation without retraining (the model dumps excess attention onto the first tokens).
- **Learned sparse attention** (trainable sparsity): NSA (2025), MoBA (top-$k$ over KV blocks), and **DeepSeek Sparse Attention** — a cheap "lightning indexer" scores past tokens and full attention runs only over the top-$k$; DeepSeek-V4 adds KV compression before selection (CSA) plus a heavily-compressed dense branch (HCA).
- **Hybrid linear attention:** ~3 linear/delta-rule layers (Gated DeltaNet, KDA) per full-attention layer give a constant-size state for most layers, cutting KV cache by ~75%.

**KV-cache management.** The KV cache grows linearly with context × batch × layers and dominates long-context memory. Mitigations: **GQA/MLA** (fewer KV heads / latent KV), **KV quantization** (fp8/int8/int4), **KV eviction/compression** (H2O, SnapKV — keep "heavy hitter" tokens; transform coding), **architectural KV reduction** (hybrids, compressed sparse attention), and **PagedAttention** (virtual-memory paging, prefix sharing).

**Retrieval as an alternative.** **RAG** sidesteps long context: retrieve the relevant chunks and feed only those, trading recall for cost. The long-context-vs-RAG tradeoff is a recurring design question; agentic **context engineering** (just-in-time retrieval via tools, compaction, sub-agents with clean windows) is a third option.

## Key Challenges
- **Lost in the middle (Liu et al. 2023).** Models use information at the **beginning and end** of long contexts far better than the **middle** — a U-shaped accuracy curve; long context ≠ uniform usage.
- **Extension–quality tradeoff.** Aggressive positional stretching can degrade short-context quality and reasoning.
- **Effective vs nominal context.** A model "supporting" 1M tokens may not *reason* reliably over all of it (retrieval-style "needle in a haystack" passes ≠ multi-hop reasoning over the full context).
- **Efficiency vs multi-hop reasoning.** Hybrid/linear attention matched full attention on standard benchmarks but lost on multi-hop reasoning in MiniMax's experiments, which is why MiniMax-M2 reverted to full attention.
- **Cost.** Long-context prefill is compute-heavy and KV memory is large; serving 1M-token contexts is expensive.
- **Evaluation.** Needle-in-a-haystack is necessary but insufficient; harder long-reasoning benchmarks (RULER, LongBench, ∞-Bench, MRCR) reveal degradation.

## Solutions & Current Best Practices
**RoPE theta-scaling / YaRN + a long-document annealing phase** for positional extension; **FlashAttention + sequence/ring parallelism** for compute; **GQA/MLA + KV quantization + paging** for memory. At 1M scale, adopt an **architectural** fix: learned top-$k$ sparse attention over (compressed) KV (DSA/CSA, MSA) or a ~3:1 linear/full hybrid (Gated DeltaNet, KDA), trained dense at short context first and switched to sparse in later long-context stages. Use **output-gated attention** or **StreamingLLM-style sinks** for stability and unbounded streaming. Evaluate with **RULER/∞-Bench/MRCR** and multi-hop tasks, not just needle tests. Decide **long-context vs RAG** by task: RAG for large, sparse knowledge bases (cheaper, updatable); long context for dense cross-document reasoning. Frontier systems combine long context, retrieval, and compaction.

## Lab Perspectives
- **Google DeepMind:** long-context pioneer (Gemini 1.5 at 1–2M tokens) with strong needle-test results; architecture proprietary. Research on test-time memory (Titans, ATLAS, Nested Learning/HOPE) targets beyond-window memory.
- **Anthropic:** Claude's long context (100K→1M; Opus 4.6 brought 1M to Opus in beta, scoring 76% on MRCR v2 8-needle at 1M vs 18.5% for Sonnet 4.5); emphasizes reliable use and context engineering (compaction, sub-agents).
- **OpenAI:** 128K–1M (GPT-4.1), ~1.05M for GPT-5.4 and GPT-6 Astra; productized long context.
- **DeepSeek:** **MLA** for KV efficiency; **NSA** → **DSA** (V3.2) → **CSA/HCA** (V4, 1M default) for trainable long-context efficiency; DSA adopted by GLM-5.
- **Moonshot / Qwen / MiniMax:** MoBA and KDA hybrids (Kimi Linear, K3 at 1M); Gated DeltaNet hybrids (Qwen3-Next, Qwen3.5); MiniMax went from lightning-attention hybrid (M1) to full attention (M2) to block-sparse MSA (M3, 1M).
- **Meta/Mistral:** RoPE-scaling (Llama 3.1 128K) and sliding-window (Mistral) approaches in open models.

## Latest Developments (2023–Nov 2026)
**1M–2M token** context became available at the frontier (Gemini), with research pushing toward 10M+. **Trainable sparse attention** (NSA, MoBA) targets long-context efficiency end-to-end. **KV compression/eviction** (SnapKV, H2O) and **MLA** addressed the memory wall. Harder **long-context reasoning** benchmarks (RULER, ∞-Bench) exposed that nominal ≫ effective context, refocusing work on *usable* long context and hybrid **long-context + retrieval + memory** systems (see [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)).

**Mid-2025 → Nov 2026 update:**
- **DeepSeek Sparse Attention (2025-09/12).** An FP8 lightning indexer (few heads, ReLU) selects top-$k$ tokens for MLA; trained by a dense warm-up that KL-aligns the indexer to the main attention distribution, then sparse continued training ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556)); GLM-5 adopted it at 744B ([arXiv:2602.15763](https://arxiv.org/abs/2602.15763)).
- **Compressed sparse attention at 1M (2026-04).** DeepSeek-V4's CSA compresses each $m$-token KV block into one entry before indexer top-$k$ selection, while HCA runs dense attention over a far more compressed KV; at 1M context V4-Pro reportedly needs ~27% of V3.2's per-token FLOPs and ~10% of its KV cache, with sparse attention switched on at a 64K stage after dense 4K→16K training (report listed as arXiv:2606.19348; ID/date inconsistent — verify).
- **Hybrid linear attention goes mainstream (2025-09/10).** Qwen3-Next mixes Gated DeltaNet and gated attention 3:1 (>10× prefill throughput vs Qwen3-32B beyond 32K); Kimi Linear's channel-wise-gated KDA, interleaved 3:1 with MLA, reports up to 75% less KV cache and up to 6× decode throughput at 1M and claims to beat full attention under fair comparison ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)).
- **Block-sparse attention for 1M (2026-06).** MiniMax-M3's MSA scores 128-token blocks and keeps the top-16 blocks per GQA group, claiming 9× prefill / 15× decode speedups over M2 at 1M.
- **Gating fixes sinks and helps long context (2025-05).** A head-specific sigmoid gate after SDPA removes the attention sink and improves RULER by >10 points ([arXiv:2505.06708](https://arxiv.org/abs/2505.06708)).
- **Context as test-time learning (2025-05 → 2025-12).** TTT-E2E treats long context as continual learning — a sliding-window Transformer takes gradient steps on the context at test time, matching full-attention scaling with constant per-token latency and 2.7× speedup at 128K ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)); ATLAS reports +80% on BABILong at 10M tokens ([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)).
- **KV compression (2026).** KV transform coding (KVTC, ICLR 2026; up to 20×) and quantize-or-evict policies (ARKV, [arXiv:2603.08727](https://arxiv.org/abs/2603.08727)) report large savings (abstract-level); DSA-style indexers add an "indexer K cache" that complicates prefill/decode KV transfer.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How do you extend a 4K-trained model to 128K?"** RoPE PI/NTK/YaRN theta-scaling + long-document fine-tuning; manage KV with GQA/MLA + quantization.
- **"What is 'lost in the middle'?"** U-shaped utilization — models use context ends better than the middle; nominal length ≠ uniform usage.
- **"Long context vs RAG — when each?"** RAG for large sparse knowledge (cheap, updatable); long context for dense cross-document reasoning; often combine.
- **"How does StreamingLLM enable infinite generation?"** Keep initial attention-sink tokens + recent window; offload excess attention to sinks.
- **"Biggest cost of long context at inference?"** KV cache memory (and prefill compute) → quantize/page/compress KV, GQA/MLA.
- **"Make 1M context affordable: learned sparse attention or a linear hybrid?"** Sparse (DSA/CSA): keeps full-attention expressivity on selected tokens, but KV must still be stored (compressed) and the indexer is $O(L^2)$-cheap; hybrid (GDN/KDA 3:1): constant state for most layers and big KV savings, but MiniMax found weaker multi-hop reasoning and low-precision state sensitivity. Validate on multi-hop long-context evals, not perplexity.

## Open Problems
Making **effective** context match **nominal** context — reliable multi-hop reasoning over millions of tokens, not just retrieval — is unsolved. Eliminating "lost in the middle," efficient/trainable long-context attention without multi-hop quality loss (sparse vs hybrid vs compressed), and the right architecture for **persistent memory** beyond a fixed window (test-time training, memory layers, vs RAG and agentic context engineering) remain open (see [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)).

## References
- Chen, S. et al. (2023). *Position Interpolation.* arXiv:2306.15595.
- Peng, B. et al. (2023). *YaRN.* arXiv:2309.00071.
- Xiao, G. et al. (2023). *StreamingLLM.* arXiv:2309.17453.
- Liu, N. et al. (2023). *Lost in the Middle.* arXiv:2307.03172.
- Yuan, J. et al. (2025). *Native Sparse Attention (NSA).* arXiv:2502.11089.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
