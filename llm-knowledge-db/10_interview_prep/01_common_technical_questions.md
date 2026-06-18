# Common Technical Questions

> **Last Updated:** 2026-06-18
> **Related Files:** [RLHF](../03_posttraining/03_RLHF.md) · [DPO](../03_posttraining/05_DPO_and_preference_optimization.md) · [Attention Mechanisms](../00_foundations/01_attention_mechanisms.md) · [Scaling Laws](../00_foundations/04_scaling_laws.md)

## Overview
This is a rapid-fire bank of the **most frequently asked LLM technical questions**, each with a crisp model answer at the depth interviewers expect. Use it as a self-quiz: cover the answer, attempt it aloud, then check. Each links to the deep-dive file. The goal is **fluency** — being able to state the key equation, the mechanism, and the tradeoff in 60–90 seconds.

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
No critic; advantage = group-relative normalized reward $\hat A_i=(r_i-\text{mean})/\text{std}$ over $G$ samples. Cheaper, stable, great for verifiable rewards. (See [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md).)

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
Plausibility-not-truth objective, knowledge gaps, miscalibration, and eval incentives that reward guessing over abstaining. (See [Hallucination](../05_alignment_and_safety/01_hallucination_causes_and_mitigations.md).)

## Interview Angles
> 💡 **Meta-advice:** State the **equation**, the **mechanism**, and the **tradeoff** for each. Interviewers escalate ("why $\sqrt{d_k}$ and not $d_k$?", "derive DPO", "what breaks at scale?") — be ready to go one level deeper than the headline.

## References
See linked deep-dive files for full citations.
