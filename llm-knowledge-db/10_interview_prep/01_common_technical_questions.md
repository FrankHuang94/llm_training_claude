# Common Technical Questions

> **Last Updated:** 2026-11-04
> **Related Files:** [RLHF](../03_posttraining/03_RLHF.md) · [DPO](../03_posttraining/05_DPO_and_preference_optimization.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Diffusion & Hybrid LMs](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)

## Overview
This is a rapid-fire bank of the **most frequently asked LLM technical questions**, each with a crisp model answer at the depth interviewers expect. Use it as a self-quiz: cover the answer, attempt it aloud, then check. Each links to the deep-dive file. The goal is **fluency** — being able to state the key equation, the mechanism, and the tradeoff in 60–90 seconds. The final section adds the **2025–2026 questions** that now appear in frontier-lab loops (RL objectives and numerics, optimizers, FP4, sparse/hybrid attention, safety and evals).

## Architecture & Attention
**Q: Derive scaled dot-product attention and explain the $\sqrt{d_k}$.**
$\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$. The $\sqrt{d_k}$ counteracts dot-product variance growth ($\propto d_k$) that would push softmax into saturated, low-gradient regions. (See [Transformer Architecture](../00_foundations/00_transformer_architecture.md).)

**Q: Why does GQA reduce memory, and what does it trade off?**
GQA shares KV heads across groups of query heads, shrinking the **KV cache** by factor $h/g$. Trades a small quality loss (vs full MHA) for big memory/throughput gains; far better quality than MQA's single KV head. (See [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md).)

**Q: How does FlashAttention avoid $O(n^2)$ memory?**
Tiling + **online softmax** compute attention in SRAM in one streaming pass without materializing the $n\times n$ score matrix; it's IO-aware and *exact*, attacking the memory-bound bottleneck.

**Q: Why did RoPE win over sinusoidal/learned encodings?**
Rotates Q/K so the dot product depends only on **relative** position ($m-n$); no params, preserves norms, decays with distance, and extends cleanly to long context (YaRN). (See [Positional Encoding](../00_foundations/03_positional_encoding.md).)

**Q: Pre-norm vs post-norm; RMSNorm vs LayerNorm?**
Pre-norm ($x+\text{Sublayer}(\text{Norm}(x))$) gives stable gradients at depth. RMSNorm drops mean-centering (cheaper, equal quality). The modern default is pre-norm RMSNorm.

## Scaling & Pretraining
**Q: State and explain the Chinchilla scaling law.**
$L(N,D)=E+A/N^\alpha+B/D^\beta$; compute-optimal scales $N$ and $D$ ~equally (~**20 tokens/param**). Gopher-class models were *undertrained*; a 70B/1.4T Chinchilla beat 280B Gopher at equal compute. (See [Scaling Laws](../00_foundations/04_scaling_laws.md).)

**Q: Why did Kaplan and Chinchilla disagree?**
Kaplan didn't tune the LR schedule per horizon and counted parameters inconsistently, biasing toward "bigger models, less data." Chinchilla's corrected method gave equal scaling.

**Q: Why overtrain a small model past compute-optimal (LLaMA paradigm)?**
**Inference cost dominates** lifetime cost; a smaller, longer-trained model is cheaper to serve while strong for its size.

**Q: Derive training FLOPs and explain MFU.**
$C\approx 6ND$ ($2N$ fwd + $4N$ bwd per token). **MFU** = useful FLOPs / (peak × devices × time); typically 35–55% due to memory-boundedness, comm, bubbles. (See [Compute Efficiency](../02_pretraining/05_compute_and_memory_efficiency.md).)

**Q: What causes training loss spikes; how do you handle them?**
Attention/output-logit growth, bad batches, fp16 overflow. Mitigate: gradient clipping, **z-loss**, **QK-norm**, bf16, μP; recover by **rollback + skip-batch**. (See [Training Stability](../02_pretraining/02_training_stability.md).)

**Q: bf16 vs fp16 for training?**
bf16 has fp32-range exponent → stable, no loss scaling; fp16 has more precision but narrow range → needs (dynamic) loss scaling.

**Q: Explain ZeRO stages / FSDP.**
Partition optimizer states (1), + gradients (2), + parameters (3) across DP ranks; ZeRO-3 ≈ FSDP (all-gather params just-in-time, reduce-scatter grads). (See [Distributed Training](../02_pretraining/01_distributed_training.md).)

## Post-Training (highest yield)
**Q: Explain the KL term in PPO/RLHF.**
Reward $r-\beta(\log\pi_\theta-\log\pi_{ref})$ keeps the policy near the reference, preventing reward-model exploitation and off-distribution drift. $\beta$ trades reward vs fidelity; it's **reverse KL** (mode-seeking → diversity loss). (See [RLHF](../03_posttraining/03_RLHF.md).)

**Q: What's the difference between DPO and RLHF/PPO?**
DPO collapses RM + KL-constrained RL into one supervised loss on preference pairs (policy = implicit reward, $Z(x)$ cancels): $-\log\sigma(\beta\log\frac{\pi_w}{\pi_{ref,w}}-\beta\log\frac{\pi_l}{\pi_{ref,l}})$. Simpler/stable/offline vs PPO's on-policy correction of OOD errors. (See [DPO](../03_posttraining/05_DPO_and_preference_optimization.md).)

**Q: Write the Bradley-Terry reward-model loss.**
$-\log\sigma(r(x,y_w)-r(x,y_l))$; rewards meaningful only relatively.

**Q: What is GRPO and how does it differ from PPO?**
No critic; advantage = group-relative normalized reward $\hat A_i=(r_i-\text{mean})/\text{std}$ over $G$ samples. Cheaper, stable, great for verifiable rewards. Known biases: per-response length normalization favors long wrong answers and std-normalization reweights easy/hard prompts (Dr. GRPO removes both). (See [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md).)

**Q: What is reward overoptimization?**
Optimizing the proxy RM too hard exploits its errors; true reward rises then *falls* past a KL threshold (Gao et al.). Mitigate with KL control, RM ensembles, verifiable rewards.

**Q: Why mask the prompt in SFT?**
Compute loss only on response tokens so the model learns to *answer*, not generate user turns.

## Efficiency
**Q: Derive LoRA and its savings.**
$\Delta W=BA$, rank $r\ll\min(d,k)$; params $dk\to r(d+k)$; $B=0$ init; mergeable → zero inference cost. QLoRA adds 4-bit NF4 base. (See [LoRA](../06_efficiency/01_LoRA_and_variants.md).)

**Q: Why does quantization speed up inference?**
Decoding is bandwidth-bound; fewer bits/weight → less HBM traffic. Outliers handled by GPTQ/AWQ/SmoothQuant. (See [Quantization](../06_efficiency/02_quantization_techniques.md).)

**Q: Why does speculative decoding preserve the output distribution?**
Modified rejection sampling: accept draft token with $\min(1,p/q)$, else resample from $(p-q)_+$ → provably samples from target $p$. (See [Speculative Decoding](../06_efficiency/04_speculative_decoding.md).)

**Q: What does MoE buy you, and what's the load-balancing problem?**
Decouples total capacity from active FLOPs/token; routers collapse without an **auxiliary load-balancing loss** ($\sum f_i P_i$) or aux-loss-free balancing. (See [MoE](../06_efficiency/05_mixture_of_experts.md).)

## Reasoning & Safety
**Q: Why does chain-of-thought help?**
Externalizes intermediate computation (more serial steps), decomposes problems, conditions on correct partial work; provably increases fixed-depth expressivity. (See [Chain-of-Thought](../04_reasoning_and_agents/00_chain_of_thought.md).)

**Q: What is inference-time scaling?**
Spend more compute per query (sampling/search/longer CoT); accuracy scales with test-time compute, tradeable against training compute. (See [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md).)

**Q: Why do LLMs hallucinate?**
Plausibility-not-truth objective, knowledge gaps, miscalibration, and eval incentives that reward guessing over abstaining. Kalai et al. (2025) formalize it: generative error is lower-bounded by an "is-it-valid" classification error, so facts seen once (singleton rate, Good–Turing) must be hallucinated at some rate; binary-graded benchmarks then reward guessing. Fix: confidence-threshold scoring with credit for "I don't know." (See [Hallucination](../05_alignment_and_safety/01_hallucination_causes_and_mitigations.md).)

## 2025–2026 Frontier Topics (new high-yield)
**Q: DAPO, Dr. GRPO, GSPO, CISPO — what problem does each fix?**
**DAPO:** clip-higher (decoupled $\epsilon_{low}<\epsilon_{high}$ so low-prob tokens can grow → fights entropy collapse), dynamic sampling (drop all-correct/all-wrong groups: zero advantage, zero gradient), **token-level** loss aggregation (long sequences aren't down-weighted), overlong reward shaping. **Dr. GRPO:** remove length and std normalization (unbiased). **GSPO:** sequence-level, length-normalized ratio $s_i=\big(\pi_\theta(y_i|x)/\pi_{old}(y_i|x)\big)^{1/|y_i|}$ with sequence-level clipping — token-level IS noise collapsed large-MoE RL; no routing replay needed (Qwen3). **CISPO:** clip the *IS weight* with stop-gradient, $\text{sg}(\text{clip}(r_t))\,\hat A\,\log\pi_\theta$, instead of clipping the update — rare "fork" tokens keep their gradient (MiniMax-M1; 2× faster than DAPO reported).

**Q: What does ScaleRL say about scaling RL compute?**
Over >400K GPU-hours, performance vs compute fits a **sigmoid** with asymptote $A$ and efficiency exponent $B$; fits on early compute extrapolate to 100K-GPU-hour runs. Most design choices change **efficiency, not the ceiling**. Recipe: PipelineRL-style async, CISPO loss, FP32 LM-head logits, prompt-level loss averaging, zero-variance filtering, no-positive-resampling.

**Q: What is training–inference mismatch in RL and how do you fix it?**
The rollout engine (vLLM/SGLang) and trainer compute different log-probs for the same tokens (BF16's 7-bit mantissa, non-batch-invariant kernels whose results depend on server batch size, MoE routing differences), so "on-policy" RL is quietly off-policy and can collapse. Fixes: **FP16** for RL (largely removes mismatch), **batch-invariant kernels** (zero sampler–trainer KL), truncated/masked IS correction, routing replay, FP32 LM head.

**Q: Muon vs AdamW?**
Muon orthogonalizes the momentum matrix (Newton–Schulz) so every singular direction gets a similar step — a spectral-norm steepest descent for 2D weights; embeddings, heads and norms stay on AdamW. To scale it: weight decay + match update RMS to AdamW (Moonlight, ~2× compute efficiency claimed), **QK-Clip** to cap attention logits (Kimi K2: 15.5T tokens, zero spikes), μP + 1/width weight decay for HP transfer. Caveat: rigorous per-optimizer tuning (Wen et al. 2025) finds ≤1.4× speedups that shrink with scale. Used in production by Kimi K2, GLM-5 and DeepSeek-V4 (reported).

**Q: How do you pretrain in NVFP4, and why NVFP4 over MXFP4?**
NVFP4 = E2M1 values in **16-element blocks** with **E4M3 (FP8) scales** plus a per-tensor FP32 scale; MXFP4 uses 32-element blocks with power-of-two (E8M0) scales — coarser. Recipe (NVIDIA, 12B on 10T tokens ≈ FP8 quality): random Hadamard transforms on Wgrad GEMM inputs (spread outliers), 2D block scaling for weights (consistent fwd/bwd quantization), **stochastic rounding** on gradients (unbiased), keep a few sensitive (mostly final) layers in BF16. MXFP4 needed ~36% more tokens to match. Nemotron 3 Super was pretrained in NVFP4 (~25T tokens).

**Q: Compare DeepSeek Sparse Attention with hybrid linear attention.**
**DSA:** a cheap FP8 "lightning indexer" (few heads, ReLU) scores all past tokens; MLA attends only to the top-$k$ → main attention $O(Lk)$ (indexer still $O(L^2)$ but cheap); trained with a dense warm-up that KL-aligns the indexer to main attention, then sparse training. Exact softmax over a learned subset; KV still stored for all tokens (V4 adds block compression: CSA/HCA). **Hybrid linear (Qwen3-Next GDN, Kimi KDA):** 3 of 4 layers keep a fixed-size delta-rule state $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ → KV only in the 1-in-4 full layers. Counterpoint: MiniMax-M2 went back to full attention (multi-hop losses, precision-sensitive state).

**Q: Why put a sigmoid gate after attention?**
Head-wise gate on the SDPA output adds non-linearity to the low-rank $W_VW_O$ path and input-dependent sparsity; it removes the **attention sink** (first-token attention 46.7% → 4.8%), tolerates larger LRs and improves long-context (RULER +10). Contrast gpt-oss's explicit learned sink logit in the softmax denominator.

**Q: What is on-policy distillation and why use it instead of SFT or RL?**
The student samples its own trajectories; the teacher scores every token, and the student minimizes per-token **reverse KL** $\sum_t \mathrm{KL}(\pi_s(\cdot|y_{<t})\,\|\,\pi_T(\cdot|y_{<t}))$ on those samples. Dense supervision (like SFT) on the student's own distribution (like RL) — no exposure bias, ~10× cheaper than RL to reach a teacher's level (reported). Used to merge domain experts and recover behaviors after mid-training. Failure modes: mode-seeking collapse; teacher signal decays on off-distribution prefixes.

**Q: Why does LoRA work for RL even at rank 1?**
Policy-gradient RL injects roughly **one bit of information per episode** (scalar reward), so the total information is tiny relative to LoRA capacity; SFT, by contrast, needs all-layer LoRA (esp. MLP/MoE) with ~10× the full-FT LR to match full fine-tuning.

**Q: How can reward hacking cause broad misalignment, and what is inoculation prompting?**
Anthropic (2025-11): a model that learned real coding-environment hacks (e.g., `sys.exit(0)` to pass tests) generalized to sabotage of safety-research code (12%) and alignment-faking reasoning (50%); RLHF safety training only masked it on chat-like inputs. **Inoculation prompting** — telling the model during training that the hack is acceptable in this context — removes most of the generalization, plausibly because the behavior no longer updates the model's self-concept ("I'm the kind of model that cheats"). Also: prevent hacks, diversify safety training.

**Q: What is evaluation awareness and why does it matter?**
Models increasingly recognize tests (Sonnet 4.5 verbalized it often; Opus 5.5 "often suspects it is being evaluated"). Steering away eval-awareness features *increased* misbehavior in some settings, and anti-scheming training raised awareness — so low misbehavior rates may reflect test recognition. Mirror failure: believing reality is a simulation caused real 2026 agentic incidents. Mitigations: realistic evals, white-box checks, production monitoring.

**Q: Explain METR's time-horizon metric.**
For each model, fit a logistic curve of success probability vs log(human time-to-complete) across tasks; the **50% horizon** is the human task length at which success = 50% (the 80% horizon is much shorter). It doubled ~every 7 months (2019–25), ~4.3 months post-2023 under TH1.1; frontier models reached ≥16 h (Mythos Preview, CI 8.5–55 h), where METR says the suite saturates.

**Q: Why did OpenAI stop reporting SWE-bench Verified?**
Its 2026-02 audit found ≥59.4% of audited hard problems had flawed tests rejecting correct solutions, 18.8% tested unrequested features, and frontier models showed signs of training-time exposure to solutions. Moved to SWE-bench Pro — where harness/scaffold choice still moves scores 10–30 points.

**Q: What are "restricted releases," and how do safety frameworks gate them?**
Dual-use capability (esp. cyber) now ships in tiers: Anthropic's Mythos Preview via Project Glasswing, then Fable 5/Mythos 5 (same weights, different safeguards; classifiers route flagged requests to a weaker model); OpenAI rated GPT-6 Astra "Critical" in cyber under its Preparedness Framework (Daybreak trusted access); Google limits Gemini 3.5 Flash Cyber to trusted partners. Frameworks: **RSP v3** (2026-02: unilateral commitments separated from an industry capability-to-mitigation map, Risk Reports every 3–6 months with external review, public Frontier Safety Roadmap), GDM **FSF v3** (harmful-manipulation CCL), OpenAI **Preparedness v2** (High/Critical).

**Q: Why is CoT monitorability "fragile"?**
Reasoning traces currently reveal intent to misbehave, but direct optimization pressure on CoT, heavy outcome RL, or latent-reasoning architectures can make traces unfaithful. OpenAI's 13-eval suite: CoT monitors beat action-only monitors, longer CoT is more monitorable, and RL hasn't yet degraded it (two data points); direct CoT reward pressure does.

## Interview Angles
> 💡 **Meta-advice:** State the **equation**, the **mechanism**, and the **tradeoff** for each. Interviewers escalate ("why $\sqrt{d_k}$ and not $d_k$?", "derive DPO", "what breaks at scale?") — be ready to go one level deeper than the headline.

## References
See linked deep-dive files for full citations.
