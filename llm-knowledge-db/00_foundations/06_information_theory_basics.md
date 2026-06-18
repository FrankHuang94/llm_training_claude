# Information Theory Basics for LLMs

> **Last Updated:** 2026-06-18
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
Report **BPB** alongside perplexity for cross-model claims. In RLHF/GRPO, use low-variance KL estimators (John Schulman's $k_3$ estimator $\frac{r-1-\log r}{}$, unbiased and non-negative) and tune $\beta$ to balance reward and KL. **Label smoothing** is understood as adding a uniform-distribution cross-entropy term (penalizing over-confidence, regularizing). Entropy bonuses in RL encourage exploration by maximizing policy entropy.

## Lab Perspectives
Labs broadly agree on the theory but differ in framing. **OpenAI/DeepMind** lean on the compression-as-intelligence view (DeepMind's "Language Modeling Is Compression," 2023, showed LLMs beat gzip/PNG). **Anthropic** emphasizes KL-to-reference as a safety/conservatism lever in RLHF and CAI. **DeepSeek** publicized clean KL handling in GRPO. All report BPB-style normalized metrics internally.

## Latest Developments (2023–2026)
"Language Modeling Is Compression" (Delétang et al. 2023) empirically tied LM quality to lossless compression of arbitrary modalities. Debate continues on whether **lower loss ⇒ more general intelligence**. Information-theoretic analyses of in-context learning, the role of entropy in reasoning-model exploration (GRPO), and MDL accounts of generalization/grokking are active.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why is minimizing cross-entropy the same as minimizing KL to the data?"** Decompose $H(p,q)=H(p)+D_{KL}(p\|q)$.
- **"Define perplexity and its caveats."** Exp of mean NLL; branching-factor intuition; tokenizer dependence → use BPB.
- **"Forward vs reverse KL — which does RLHF use and why does it matter?"** Reverse KL, mode-seeking, diversity collapse.
- **"How is the KL penalty estimated in PPO/GRPO?"** Sampling-based $k_3$ estimator; bias/variance tradeoffs.

## Open Problems
Whether the compression-intelligence equivalence extends to *reasoning* and *agency* (not just prediction) is contested. The precise relationship between cross-entropy reduction and emergent capability, and information-theoretic limits on sample efficiency and continual learning, remain open.

## References
- Shannon, C. (1948). *A Mathematical Theory of Communication.* Bell System Tech. J.
- Cover, T., Thomas, J. *Elements of Information Theory* (2nd ed.).
- Delétang, G. et al. (2023). *Language Modeling Is Compression.* arXiv:2309.10668.
- Schulman, J. (2020). *Approximating KL Divergence* (blog, the $k_1/k_2/k_3$ estimators).
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
