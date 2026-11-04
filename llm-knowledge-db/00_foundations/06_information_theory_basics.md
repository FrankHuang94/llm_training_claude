# Information Theory Basics for LLMs

> **Last Updated:** 2026-11-04
> **Related Files:** [Scaling Laws](04_scaling_laws.md) · [Pretraining Objectives](../02_pretraining/00_pretraining_objectives.md) · [DPO & Preference Optimization](../03_posttraining/05_DPO_and_preference_optimization.md)
> **Key Papers:** Shannon 1948 "A Mathematical Theory of Communication" · Cover & Thomas, *Elements of Information Theory* · Schulman 2015 TRPO (KL trust region, [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)) · Rafailov et al. 2023 DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290))

## Overview
Information theory is the mathematical bedrock of language modeling. Training an LM by maximum likelihood is *exactly* minimizing cross-entropy between the data distribution and the model — equivalently, building the best possible lossless compressor of the data. Many quantities you will be quizzed on (perplexity, bits-per-byte, the KL penalty in RLHF, the entropy bonus in PPO, label smoothing) are direct applications of a handful of information-theoretic primitives. A frequently cited framing — "**compression is intelligence**" (and the practical observation that strong LLMs are state-of-the-art compressors) — rests entirely on this equivalence.

## Core Concepts
**Entropy.** The average information content of a distribution $p$:
$$H(p)=-\sum_x p(x)\log p(x)$$
It is the theoretical minimum expected code length (in bits if $\log_2$) to encode samples from $p$. For language, $H$ is the irreducible "$E$" floor in scaling laws.

**Cross-entropy and the LM objective.** Training minimizes
$$H(p,q)=-\sum_x p(x)\log q(x) = H(p)+D_{KL}(p\,\|\,q)$$
where $p$ is data and $q$ the model. Since $H(p)$ is fixed, minimizing cross-entropy minimizes $D_{KL}(p\|q)$ — pushing the model toward the data distribution. Per-token next-token loss is exactly this cross-entropy.

**KL divergence.** $D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}\ge0$, zero iff $p=q$. It is asymmetric and not a metric. KL appears everywhere in post-training: the RLHF reward includes $-\beta\,D_{KL}(\pi_\theta\|\pi_{ref})$ to keep the policy near the reference (see [RLHF](../03_posttraining/03_RLHF.md)); DPO's optimum is a KL-regularized reward maximizer; PPO/TRPO use KL trust regions. **Forward vs reverse KL** matters: reverse KL ($D_{KL}(q\|p)$) is mode-seeking (the RL/variational case), forward KL is mode-covering (the MLE case) — this explains why RLHF can reduce diversity.

**Perplexity.** The standard LM metric, the exponentiated per-token cross-entropy:
$$\text{PPL}=\exp\!\Big(-\frac1N\sum_{i=1}^N\log q(x_i\mid x_{<i})\Big)$$
Interpretable as the model's effective branching factor (average number of equally-likely next tokens). Lower is better. It is **tokenizer-dependent**, so cross-model PPL comparisons require identical tokenization.

**Bits-per-byte (BPB) / bits-per-character.** A tokenizer-independent alternative: normalize total negative log-likelihood by the number of *bytes* rather than tokens, enabling fair cross-tokenizer comparison and connecting directly to compression ratio.

**Mutual information & MDL.** Mutual information $I(X;Y)=H(X)-H(X|Y)$ underlies contrastive objectives (CLIP/InfoNCE) and representation learning. The **Minimum Description Length** principle frames learning as compression: the best model minimizes (model description + data-given-model description), a lens on regularization and the "grokking"/generalization debate.

## Key Challenges
- **Tokenizer dependence of PPL.** Makes naive comparisons misleading; BPB or shared tokenizers are required.
- **KL estimation in RL.** The KL penalty must be estimated from samples; biased/high-variance estimators (the $k_1,k_2,k_3$ estimators) affect stability.
- **Asymmetry pitfalls.** Choosing forward vs reverse KL changes whether you cover or collapse modes — a frequent source of confusion.
- **Loss ≠ capability.** Cross-entropy improvements don't map linearly to downstream skill (see [Scaling Laws](04_scaling_laws.md)).

## Solutions & Current Best Practices
Report **BPB** alongside perplexity for cross-model claims (essential when a lab changes tokenizer between versions, as Anthropic did with Claude Opus 4.7). In RLHF/GRPO, use low-variance KL estimators (John Schulman's $k_3$ estimator $\frac{r-1-\log r}{}$, unbiased and non-negative) and tune $\beta$ to balance reward and KL. **Label smoothing** is understood as adding a uniform-distribution cross-entropy term (penalizing over-confidence, regularizing). Entropy bonuses in RL encourage exploration by maximizing policy entropy; 2025–26 reasoning-RL practice prefers targeted entropy control (clip-higher, covariance-based clipping/penalties) over a global bonus. For distillation, **on-policy reverse KL** (student samples, teacher scores each token) is now a standard post-training stage.

## Lab Perspectives
Labs broadly agree on the theory but differ in framing. **OpenAI/DeepMind** lean on the compression-as-intelligence view (DeepMind's "Language Modeling Is Compression," 2023, showed LLMs beat gzip/PNG). **Anthropic** emphasizes KL-to-reference as a safety/conservatism lever in RLHF and CAI. **DeepSeek** publicized clean KL handling in GRPO. **Thinking Machines** framed on-policy distillation as a per-token reverse-KL reward and argued RL transmits only ~1 bit per episode. **OpenAI** (Kalai et al. 2025) recast hallucination as a statistical consequence of pretraining plus binary-graded evals. All report BPB-style normalized metrics internally.

## Latest Developments (2023–Nov 2026)
"Language Modeling Is Compression" (Delétang et al. 2023) empirically tied LM quality to lossless compression of arbitrary modalities. Debate continues on whether **lower loss ⇒ more general intelligence**. Information-theoretic analyses of in-context learning, the role of entropy in reasoning-model exploration (GRPO), and MDL accounts of generalization/grokking are active.

**Mid-2025 → Nov 2026 update:**
- **An entropy law for RL (2025-05; arXiv:2505.22617).** Without intervention, reasoning RL trades entropy for reward predictably, $R=-a\,e^{H}+b$, so the attainable reward is capped once entropy collapses. The per-step entropy change is driven by the covariance between a token's log-probability and its advantage; Clip-Cov (detach high-covariance tokens) and KL-Cov (penalize them) keep entropy alive. DAPO's decoupled clip-higher (arXiv:2503.14476) is the simpler cousin.
- **Reverse KL as a dense reward (2025-10, Thinking Machines; secondary).** On-policy distillation samples from the student and uses per-token $-\log\frac{\pi_\theta(x_t)}{\pi_T(x_t)}$ as reward — the mode-seeking $D_{KL}(\pi_\theta\|\pi_T)$ on the student's own distribution — reportedly reaching ~70% on AIME24 at roughly a tenth of RL's GPU-hours. Its mode-seeking failure modes motivated entropy-gated forward/reverse switching (arXiv:2603.07079) and truncated top-K reverse KL (arXiv:2603.25562).
- **Bits per episode (2025-09, "LoRA Without Regret"; secondary).** Policy-gradient RL with a scalar reward conveys on the order of 1 bit per episode, which is why even rank-1 LoRA matches full fine-tuning for RL while SFT needs far more capacity — an information-capacity argument for adapter sizing.
- **Likelihood ratios under numeric noise (2025-07 → 2025-10).** GSPO (arXiv:2507.18071) replaces token-level importance ratios with a length-normalized sequence-level ratio to cut variance in MoE RL. BF16's 7-bit mantissa makes rollout-engine and trainer log-probs diverge, silently turning "on-policy" RL off-policy; switching RL to FP16 largely removes the mismatch (arXiv:2510.26788), and batch-invariant kernels drive sampler–trainer KL to exactly zero (Thinking Machines; secondary).
- **Hallucination as a statistical lower bound (2025-09; arXiv:2509.04664).** Generative error is lower-bounded by the error of the binary "is-this-valid" classification problem; facts seen exactly once (the Good–Turing singleton rate) must yield hallucinations at a comparable rate. Binary-graded benchmarks then reward guessing over abstaining; the proposed fix is confidence-threshold scoring with credit for "I don't know".
- **Loss ≠ capability, again (2026-02; arXiv:2602.15014).** Among diffusion LMs, uniform-state diffusion has worse likelihood than masked diffusion yet does better on GSM8K — a clean example that likelihood bounds across model families do not rank downstream skill.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why is minimizing cross-entropy the same as minimizing KL to the data?"** Decompose $H(p,q)=H(p)+D_{KL}(p\|q)$.
- **"Define perplexity and its caveats."** Exp of mean NLL; branching-factor intuition; tokenizer dependence → use BPB.
- **"Forward vs reverse KL — which does RLHF use and why does it matter?"** Reverse KL, mode-seeking, diversity collapse.
- **"How is the KL penalty estimated in PPO/GRPO?"** Sampling-based $k_3$ estimator; bias/variance tradeoffs.
- **"Why is on-policy distillation reverse KL, and what can go wrong?"** Sampling from the student makes the expectation tractable and fixes exposure bias; reverse KL is mode-seeking, so the student can collapse onto one teacher mode and the teacher signal weakens on off-distribution prefixes.
- **"Why does a singleton rate lower-bound hallucinations?"** Good–Turing: the fraction of facts seen once estimates the probability mass of unseen facts the model cannot reliably distinguish from plausible fabrications.

## Open Problems
Whether the compression-intelligence equivalence extends to *reasoning* and *agency* (not just prediction) is contested. The precise relationship between cross-entropy reduction and emergent capability, and information-theoretic limits on sample efficiency and continual learning, remain open.

## References
- Shannon, C. (1948). *A Mathematical Theory of Communication.* Bell System Tech. J.
- Cover, T., Thomas, J. *Elements of Information Theory* (2nd ed.).
- Delétang, G. et al. (2023). *Language Modeling Is Compression.* arXiv:2309.10668.
- Schulman, J. (2020). *Approximating KL Divergence* (blog, the $k_1/k_2/k_3$ estimators).
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Cui, G. et al. (2025). *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* arXiv:2505.22617.
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- Kalai, A. T. et al. (2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
