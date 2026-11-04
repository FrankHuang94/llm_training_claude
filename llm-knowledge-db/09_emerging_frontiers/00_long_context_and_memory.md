# Long Context and Memory

> **Last Updated:** 2026-11-04
> **Related Files:** [Context Window Extension](../06_efficiency/06_context_window_extension.md) · [Non-Transformer Architectures](04_neuromorphic_and_non_transformer.md) · [Diffusion & Hybrid LMs](06_diffusion_and_hybrid_language_models.md) · [Agentic Frameworks](../04_reasoning_and_agents/04_agentic_frameworks.md)
> **Key Papers:** Liu et al. 2023 Lost in the Middle ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) · Packer et al. 2023 MemGPT ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)) · Behrouz et al. 2024 Titans ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) · Gemini Team 2024 (1M+ context) ([arXiv:2403.05530](https://arxiv.org/abs/2403.05530)) · Kimi Team 2025 Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692))

## Overview
The distinction between **context** (information in the current window) and **memory** (information persisted across windows/sessions) is becoming central as models tackle book-length documents, hour-long videos, multi-day agentic tasks, and personalized assistance. Context windows have reached 1–10M tokens — and by 2026 **~1M tokens is the default spec** for both closed frontier models and open flagships — but two problems remain: models **don't use long context uniformly** ("lost in the middle", "context rot"), and a fixed window — however large — is not true **persistent memory**. This frontier explores both pushing context length *and* architecting external/parametric memory so models can accumulate and recall information over time. It bridges efficiency engineering (KV management, sparse/hybrid attention) and architectural research (memory modules, SSMs, test-time-trained memory).

## Core Concepts
**Context vs memory.**
- **Context:** the transformer's attention window — fast, fully accessible, but quadratic-cost and ephemeral (cleared each call).
- **Memory:** mechanisms to store and retrieve information *beyond* the window: external (retrieval/databases — RAG, vector stores), or *internal/parametric* (weights or recurrent state that carry information forward).

**Lost in the middle (Liu et al.).** Even within a supported window, retrieval/usage accuracy is **U-shaped** in position — strong at the start and end, weak in the middle. Long nominal context ≠ uniform usable context; this motivates better positional methods, training on long documents, and retrieval reranking.

**External memory architectures.**
- **MemGPT:** treats the LLM like an OS with virtual memory — a fixed "main context" plus "external storage," with the model issuing function calls to **page** information in/out, enabling effectively unbounded memory.
- **RAG / vector memory:** store embeddings of past interactions/documents; retrieve top-k relevant chunks per query (see [Hallucination](../05_alignment_and_safety/01_hallucination_causes_and_mitigations.md) for grounding).
- **Agent memory:** short-term scratchpads + long-term episodic/semantic stores + learned skills (see [Agentic Frameworks](../04_reasoning_and_agents/04_agentic_frameworks.md)). **Context engineering** (compaction, structured note-taking, sub-agents with clean windows, just-in-time retrieval via tools) is now the practical discipline for agents.

**Internal/learned memory (a 2024–2026 frontier).**
- **Titans** (Behrouz et al.): augment attention with a **neural long-term memory** module that *learns to memorize at test time* (surprise-based gating), combining the precision of attention with a compressive recurrent memory — scaling to very long sequences. Successors (ATLAS, HOPE) and **TTT-E2E** treat long context as *test-time learning* (see [Test-Time Training](03_test_time_training.md)).
- **Compressive / recurrent memory** (Infini-attention, RMT, RWKV/SSM states, Gated DeltaNet/KDA): compress past context into a bounded state carried forward — sub-quadratic but lossy.

**KV-cache as memory.** The KV cache *is* the transformer's working memory; managing it (quantization, eviction of non-"heavy-hitter" tokens, paging, prefix caching, MLA latent compression, learned top-k sparse selection) is the practical face of long-context memory (see [Context Window Extension](../06_efficiency/06_context_window_extension.md)). For a full-attention model the KV footprint is $2\,n_{\text{layers}}\,n_{kv}\,d_{\text{head}}\,L$ elements, linear in context $L$ — the term that 2025–26 hybrid and compressed-sparse designs attack.

## Key Challenges
- **Effective ≪ nominal context.** Models can't reliably reason over all of a 1M-token window (lost-in-the-middle; multi-hop degradation; "context rot" as irrelevant tokens accumulate).
- **Cost.** Long-context prefill compute and KV memory are large; persistent memory adds storage/retrieval overhead.
- **Memory management policy.** *What* to store, retrieve, compress, and forget — and when — is an unsolved control problem (the "memory hierarchy" question); 2026 work starts to *learn* it with RL.
- **Catastrophic interference.** Parametric memory updates risk overwriting old knowledge (links to [Continual Learning](01_continual_learning.md)).
- **Retrieval quality.** RAG is only as good as retrieval; errors propagate.
- **Evaluation.** Needle tests are insufficient; multi-hop, aggregation, and long-horizon memory benchmarks (RULER, ∞-Bench, LongMemEval, MRCR multi-needle) remain immature.

## Solutions & Current Best Practices
**Hybrid: long context + retrieval + structured memory.** Use long context for dense, in-task reasoning; **RAG/vector memory** for large external knowledge; **MemGPT-style paging**, compaction, and agent note-taking for cross-session persistence. Manage KV with **MLA/GQA + quantization + eviction/paging**; at 1M scale, open models make context affordable with **~3:1 linear:full hybrids** (Gated DeltaNet/KDA + full/MLA layers) or **learned top-k sparse attention over (compressed) KV** (DSA → CSA/HCA). Train on **long documents** and evaluate with **multi-hop** long-context benchmarks (not just needle). Emerging **learned-memory** modules (Titans/ATLAS/HOPE), **test-time-trained weights** (TTT-E2E), and **SSM/attention hybrids** (see [Non-Transformer](04_neuromorphic_and_non_transformer.md), [Diffusion & Hybrid LMs](06_diffusion_and_hybrid_language_models.md)) target sub-quadratic, persistent memory.

## Lab Perspectives
- **Google DeepMind:** pioneered 1–2M-token production context (Gemini 1.5 onward) and research toward 10M+ (ATLAS, Nested Learning/HOPE); no longer alone at 1M.
- **OpenAI/Anthropic:** moved from 128K–200K to **~1M** windows (Claude Opus 4.6 1M beta in 2026-02 with context compaction; GPT-5.4 and GPT-6 Astra ~1.05M), emphasizing *reliable* use (multi-needle MRCR) and agent-level memory (compaction, parallel sub-agents).
- **Chinese open-weight labs (DeepSeek, Moonshot, Qwen, MiniMax):** treat 1M context as an architecture problem — DSA/CSA/HCA (DeepSeek), KDA+MLA hybrids (Kimi), Gated DeltaNet hybrids (Qwen), block-sparse MSA (MiniMax M3).
- **Academia/startups:** MemGPT (Letta), Titans, Infini-attention, RMT, TTT-E2E — external and learned-memory architectures; RL-trained memory tools (AgeMem, Memory-R1).
- **SSM labs / NVIDIA:** Mamba-family hybrids (Nemotron 3, 1M context) as the long-context substrate.

## Latest Developments (2023–Nov 2026)
**1–2M token** context shipped (Gemini); research toward 10M+. **Learned test-time memory** (Titans) and **compressive recurrent** memory advanced. **Agent/persistent memory** systems (MemGPT/Letta, memory features in assistants) matured for personalization and long-horizon agents. New **long-context reasoning** benchmarks (RULER, ∞-Bench, LongMemEval) exposed the effective-vs-nominal gap and refocused work on *usable* memory. Debate: scale context vs build explicit memory vs switch architectures (SSMs).

**Mid-2025 → Nov 2026 update:**
- **1M becomes the default spec, and the race shifts to KV/FLOP cost (2025-09 → 2026-10).** DeepSeek-V4 (2026-04), Kimi K3 (2026-07), MiniMax M3 (~2026-06), MiMo-V2.5 and Nemotron 3 all ship 1M context; Claude Opus 4.6 (2026-02) was the first Opus at 1M (beta), scoring 76% on MRCR v2 8-needle at 1M vs 18.5% for Sonnet 4.5 (vendor) — evidence that *usable* context, not nominal length, is the competitive axis.
- **Hybrid linear attention (2025-09/10).** Qwen3-Next interleaves Gated DeltaNet and gated softmax attention 3:1 (>10× prefill throughput vs Qwen3-32B beyond 32K); **Kimi Linear** (KDA, channel-wise gated delta rule, 3:1 with MLA) reports up to 75% less KV cache and up to 6× decode throughput at 1M ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)). MiniMax's M2 reverted to full attention citing multi-hop losses — a key counterpoint.
- **Learned sparse attention over compressed KV (2025-09 → 2026-04).** DSA (lightning indexer + top-k, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)) evolved into DeepSeek-V4's CSA (compress each $m$-token block, then top-k) + HCA (dense attention over heavily compressed KV); reviewers report ~27% of V3.2's per-token FLOPs and ~10% of its KV at 1M (report listed as arXiv:2606.19348; ID/date inconsistent, verify).
- **Long context as test-time learning (2025-05 → 2025-12).** ATLAS optimizes memory over a sliding window of past tokens (Omega rule), reporting +80% on BABILong at 10M tokens ([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)); **TTT-E2E** takes next-token gradient steps on the context with a meta-learned init, scaling like full attention at 3B with constant per-token latency and 2.7× speedup at 128K ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)).
- **Gated attention fixes sinks and helps length generalization (2025-05).** A head-wise sigmoid gate after SDPA cuts first-token attention from 46.7% to 4.8% and gains >10 points on RULER; shipped in Qwen3-Next ([arXiv:2505.06708](https://arxiv.org/abs/2505.06708)).
- **Context engineering and learned memory for agents (2025-09 → 2026-01).** Anthropic's guidance frames context as a finite attention budget subject to "context rot" (compaction, note-taking, sub-agents, just-in-time retrieval); **AgeMem** makes store/retrieve/update/discard/summarize tool actions of the policy trained with step-wise GRPO ([arXiv:2601.01885](https://arxiv.org/abs/2601.01885)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Context vs memory — what's the difference?"** In-window attention (ephemeral, quadratic) vs persisted/retrievable information across sessions.
- **"What is 'lost in the middle' and how do you mitigate it?"** U-shaped positional usage; long-doc training, retrieval reranking, positional fixes.
- **"How would you give an agent long-term memory?"** MemGPT-style paging + vector store + episodic/semantic memory + compaction; manage what to store/retrieve/forget (increasingly learned with RL).
- **"Long context vs RAG vs learned memory — tradeoffs?"** Cost/recall/persistence; hybrids in practice.
- **"How do 2026 open models make 1M context affordable?"** ~3:1 linear (GDN/KDA) : full-attention hybrids with bounded state, or top-k sparse attention over compressed KV (DSA/CSA); know the KV-size formula and MiniMax's multi-hop counter-argument.

## Open Problems
Closing the **effective-vs-nominal** context gap (true multi-hop reasoning over millions of tokens), principled **memory-management** policies, **lossless-enough compressive memory**, parametric memory **without interference**, and reliable **long-horizon memory evaluation** are all unsolved. Whether hybrid/sparse attention loses something essential for long-chain reasoning (MiniMax's claim) or matches full attention under fair comparison (Kimi Linear's claim) is unsettled. Whether the future is "infinite context," explicit memory systems, test-time-trained weights, or new architectures is an open, high-stakes debate.

## References
- Liu, N. et al. (2023). *Lost in the Middle.* arXiv:2307.03172.
- Packer, C. et al. (2023). *MemGPT.* arXiv:2310.08560.
- Behrouz, A. et al. (2024). *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663.
- Munkhdalai, T. et al. (2024). *Infini-attention.* arXiv:2404.07143.
- Gemini Team (2024). *Gemini 1.5.* arXiv:2403.05530.
- Behrouz, A. et al. (2025). *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models.* arXiv:2505.06708.
- Stanford/NVIDIA et al. (2025). *End-to-End Test-Time Training for Long Context (TTT-E2E).* arXiv:2512.23675.
