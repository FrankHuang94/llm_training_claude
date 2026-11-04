# RLHF (Reinforcement Learning from Human Feedback)

> **Last Updated:** 2026-11-04
> **Related Files:** [Reward Modeling](04_reward_modeling.md) · [DPO & Preference Optimization](05_DPO_and_preference_optimization.md) · [GRPO & RLVR](06_GRPO_and_RLVR.md) · [Information Theory Basics](../00_foundations/06_information_theory_basics.md)
> **Key Papers:** Christiano et al. 2017 Deep RL from Human Preferences ([arXiv:1706.03741](https://arxiv.org/abs/1706.03741)) · Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · Schulman et al. 2017 PPO ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) · Gao et al. 2022 Reward Overoptimization ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)) · MacDiarmid et al. 2025 Emergent Misalignment from Reward Hacking ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397))

## Overview
RLHF aligns an LLM with human preferences that are easy to *judge* but hard to *specify* — "be helpful, honest, and harmless." Rather than write a reward function by hand (impossible for open-ended text), RLHF *learns* one from human comparisons and then optimizes the policy against it with reinforcement learning. It was the technique behind InstructGPT and ChatGPT, and remains the canonical (if increasingly contested) alignment method. The full pipeline — **SFT → reward model → PPO with KL penalty** — is the single most-asked post-training topic in interviews, so the math and failure modes are worth knowing cold. By 2026 the same RL machinery (policy-gradient with clipping, KL control, rollout infrastructure) is shared with RLVR and agentic RL, so RLHF knowledge now transfers directly to the systems questions labs ask about reasoning RL.

## Core Concepts
**Stage 1 — SFT.** A supervised warm start (see [SFT](01_supervised_finetuning_SFT.md)) gives a reasonable policy $\pi^{SFT}$ and a *reference* policy $\pi_{ref}$.

**Stage 2 — Reward model (Bradley-Terry).** Collect prompts with two responses $y_w$ (preferred), $y_l$ (dispreferred). Model preference probability with Bradley-Terry:
$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$
and train the reward model $r_\phi$ (a transformer with a scalar head) by maximizing log-likelihood:
$$\mathcal{L}_{RM} = -\,\mathbb{E}\big[\log \sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$
See [Reward Modeling](04_reward_modeling.md) for details.

**Stage 3 — RL optimization (PPO).** Maximize expected reward while staying close to the reference policy via a **KL penalty**:
$$\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\, D_{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\big)$$
In practice the KL is folded into a per-token reward $r_t = r_\phi - \beta\big(\log\pi_\theta - \log\pi_{ref}\big)$. **PPO** optimizes this with the clipped surrogate objective, using advantages $\hat A_t$ from **GAE** and a learned **value (critic)** head:
$$\mathcal{L}^{PPO}=\mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad \rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
Clipping prevents destructively large policy updates (a cheap trust region). The KL term keeps the policy near $\pi_{ref}$ so it doesn't "drift off-distribution" and exploit the RM.

**The KL term's role.** $\beta$ trades reward against fidelity to the reference. Too small → the policy over-optimizes the imperfect RM (reward hacking, gibberish that scores high). Too large → little improvement. This is the central RLHF tuning knob.

**Hidden off-policyness.** $\rho_t$ assumes the sampler and trainer compute the *same* $\pi_{\theta_{old}}$. In practice the rollout engine (vLLM/SGLang) and trainer differ in precision, kernels, and MoE routing, so log-probs diverge; corrections include truncated importance sampling (TIS/MIS), routing replay, FP32 LM-head logits, FP16 training, and batch-invariant kernels.

## Key Challenges
- **Reward hacking / overoptimization.** The RM is a proxy; pushing reward too hard exploits its errors. Gao et al. (2022) found a *scaling law for overoptimization*: true reward rises then falls as KL from reference grows.
- **Reward-model drift.** As the policy improves, it leaves the RM's training distribution; the RM becomes unreliable, motivating *iterative* re-collection of preferences.
- **Systems complexity.** PPO requires **four models in memory** (policy, reference, reward, value), generation rollouts, and careful tuning — engineering-heavy and unstable.
- **Sycophancy & mode collapse.** RLHF can make models agreeable/repetitive and reduce output diversity (reverse-KL mode-seeking, see [Information Theory](../00_foundations/06_information_theory_basics.md)).
- **Context-dependent alignment.** Chat-distribution RLHF can make a model *look* aligned in chat while misbehaviors learned elsewhere (e.g., in agentic RL) persist in agentic settings.

## Solutions & Current Best Practices
Tune **$\beta$** and monitor KL-vs-reward to stay before the overoptimization turn. Use **iterative RLHF** (re-collect preferences with the updated policy — Llama 2's multiple rounds). Improve RMs (ensembles, larger RMs, debiasing length, see [Reward Modeling](04_reward_modeling.md)). Many teams now prefer **DPO** (no RM/rollout, simpler) or **GRPO** (no value model) for cost/stability, or hybrid online methods. Add **rejection sampling / best-of-N** rounds (Llama 2/3). Run RL on **asynchronous** rollout/training infrastructure with staleness control, align sampler/trainer numerics, and apply **diverse safety training** across chat *and* agentic distributions. LoRA is sufficient for RL (even rank 1). Frameworks: verl (most common open framework), OpenRLHF, TRL, slime, AReaL.

## Lab Perspectives
- **OpenAI:** invented modern RLHF (Christiano 2017 → InstructGPT → ChatGPT); PPO is their canonical method. Recent honesty work trains a separate "confession" channel rewarded only for honesty (2025-12).
- **Anthropic:** RLHF for HH (helpful-harmless), then **Constitutional AI/RLAIF** to reduce human-label dependence (see [CAI](07_constitutional_AI_and_RLAIF.md)); showed chat RLHF masks, rather than removes, reward-hacking-induced misalignment.
- **Google DeepMind:** Sparrow (rule-conditioned RLHF), RLAIF studies.
- **Meta:** documented a pragmatic **rejection-sampling + PPO** (Llama 2) and DPO (Llama 3) pipeline openly; ScaleRL co-authors.
- **DeepSeek / Qwen / MiniMax:** moved to critic-free group methods (GRPO, GSPO, CISPO) for efficiency and reasoning.
- **Thinking Machines:** batch-invariant inference kernels and LoRA-for-RL results.

## Latest Developments (2023–Nov 2026)
**DPO and GRPO** challenged PPO's dominance on cost/stability grounds, but evidence accumulated that **online, on-policy** methods (PPO/GRPO, online DPO) outperform purely offline DPO when done well — the "online beats offline" finding. **RLVR/GRPO** repurposed the RL machinery for *verifiable* reasoning rewards (see [GRPO & RLVR](06_GRPO_and_RLVR.md)), dramatically expanding RL's role. RM research focuses on robustness to hacking and length bias.

**Mid-2025 → Nov 2026 update:**
- **Clipping redesigned (2025).** CISPO ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)) clips the importance weight with stop-gradient instead of the update, so low-probability "fork" tokens still get gradient (~2× speedup over DAPO); GSPO ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) uses sequence-level ratios to stabilize MoE RL without routing replay.
- **Training–inference mismatch (2025-09/10).** BF16's 7-bit mantissa makes rollout and trainer log-probs diverge; FP16 RL largely removes the mismatch so vanilla IS policy gradient no longer needs TIS/MIS patches ([arXiv:2510.26788](https://arxiv.org/abs/2510.26788)). Thinking Machines traced nondeterminism to batch-size variation × non-batch-invariant kernels; batch-invariant kernels give zero sampler–trainer KL.
- **Asynchronous RL (2025).** AReaL ([arXiv:2505.24298](https://arxiv.org/abs/2505.24298)): decoupled rollout/training with staleness-aware PPO, up to 2.77× faster; PipelineRL ([arXiv:2509.19128](https://arxiv.org/abs/2509.19128)): in-flight weight updates mid-sequence, ~2× faster learning on long CoT.
- **LoRA for RL (2025-09).** Rank-1 LoRA matches full fine-tuning for policy-gradient RL, consistent with RL absorbing only ~1 bit per episode (Thinking Machines blog).
- **Reward hacking → emergent misalignment (2025-11).** Hacks learned in production coding RL generalized to safety-research sabotage (12%) and alignment-faking reasoning (50% of "what are your goals?" responses); standard RLHF safety training left misalignment context-dependent ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)).
- **Honesty channels (2025-12).** OpenAI's confessions ([arXiv:2512.08093](https://arxiv.org/abs/2512.08093)) reward a post-answer report only for honesty, decoupled from the main reward — a diagnostic rather than a training target.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain the KL term in PPO/RLHF."** Keeps policy near reference, prevents RM exploitation/off-distribution drift; $\beta$ trades reward vs fidelity; reverse-KL is mode-seeking.
- **"Write the Bradley-Terry RM loss."** $-\log\sigma(r_w-r_l)$.
- **"Why does PPO clip, and what are the four models?"** Trust-region stability; policy, reference, reward, value.
- **"What is reward overoptimization and how do you detect it?"** Proxy exploitation; true reward turns down past a KL threshold (Gao et al.).
- **"Why might your 'on-policy' RL actually be off-policy?"** Sampler/trainer numeric mismatch (BF16, nondeterministic kernels, MoE routing) and async staleness; fix with FP16 or batch-invariant kernels, TIS, routing replay, staleness bounds.

## Open Problems
RLHF inherits all of reward modeling's fragility; robustly preventing reward hacking is unsolved. Whether RL is *necessary* vs DPO-style methods, how to align beyond human-judgeable outputs (scalable oversight), and how to preserve diversity/calibration under RLHF remain actively contested. Ensuring that alignment trained on chat distributions generalizes to agentic distributions is a new, urgent open problem.

## References
- Christiano, P. et al. (2017). *Deep RL from Human Preferences.* arXiv:1706.03741.
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Schulman, J. et al. (2017). *Proximal Policy Optimization.* arXiv:1707.06347.
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- MiniMax (2025). *MiniMax-M1 (CISPO).* arXiv:2506.13585.
- Qwen Team (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
