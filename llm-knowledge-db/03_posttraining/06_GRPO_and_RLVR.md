# GRPO and RLVR (Reasoning RL)

> **Last Updated:** 2026-06-18
> **Related Files:** [RLHF](03_RLHF.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [DeepSeek Roadmap](../08_lab_roadmaps/04_deepseek_roadmap.md)
> **Key Papers:** Shao et al. 2024 DeepSeekMath/GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · Lambert et al. 2024 Tülu 3 / RLVR ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) · Schulman et al. 2017 PPO ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))

## Overview
GRPO (Group Relative Policy Optimization) and RLVR (Reinforcement Learning with Verifiable Rewards) are the algorithmic core of the 2024–2026 reasoning revolution. The central idea is to apply RL not against a fragile learned reward model, but against an **objective verifier** — a math grader, code unit tests, a symbolic checker — that returns a reliable, non-hackable reward. This lets models *learn to reason* by trial and error: generate long chains of thought, get rewarded for correct final answers, and reinforce the reasoning that works. DeepSeek-R1 stunned the field by showing that **pure RL with verifiable rewards**, with no reasoning-specific supervised data, induces emergent long chain-of-thought, self-verification, and "aha moments." This is the most active post-training area today.

## Core Concepts
**RLVR (verifiable rewards).** The reward is computed by a *verifier* for domains with checkable answers: `reward = 1` if the final answer matches ground truth (math) or passes all unit tests (code), else `0` (sometimes with format rewards). Because the verifier is correct by construction, there is **no reward model to hack** — the dominant failure mode of RLHF is largely removed, enabling stable, large-scale RL.

**GRPO (the algorithm).** PPO needs a separate **value/critic** network to estimate the baseline for advantages — expensive (doubles model memory) and itself hard to train for long sequences. GRPO eliminates the critic by using a **group-relative baseline**: for each prompt, sample a *group* of $G$ outputs $\{o_1,\dots,o_G\}$, score each with the verifier to get rewards $\{r_i\}$, and compute the advantage by **normalizing within the group**:
$$\hat A_i=\frac{r_i-\text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}$$
The group mean *is* the baseline (an empirical, unbiased estimate of expected reward for that prompt). The policy is then updated with a PPO-style clipped objective using these advantages, plus a KL penalty to the reference:
$$\mathcal{L}_{GRPO}=\mathbb{E}\Big[\tfrac{1}{G}\sum_i \min\big(\rho_i \hat A_i,\ \text{clip}(\rho_i,1\!-\!\epsilon,1\!+\!\epsilon)\hat A_i\big)\Big]-\beta\,D_{KL}(\pi_\theta\|\pi_{ref})$$
This is cheaper (no value net), naturally suited to outcome rewards (advantage shared across all tokens of a sample), and stable.

**R1-Zero vs R1.** **R1-Zero** applied GRPO+RLVR directly to a base model with *no SFT* — and long CoT, reflection, and self-correction *emerged* purely from RL, demonstrating reasoning can be learned without demonstrations (though output readability/language-mixing suffered). **R1** added a small **cold-start long-CoT SFT** before RL (and a final RLHF stage for general alignment) to fix readability and broaden capability — the production recipe.

**Why RLVR enables long reasoning.** The verifiable reward only cares about *correctness*, so the model is free to use however many "thinking tokens" it needs; RL discovers that *more deliberate, longer* reasoning yields higher reward, and **test-time compute scales up emergently** (see [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md)).

## Key Challenges
- **Verifiable-domain limitation.** RLVR needs a checkable reward — natural for math/code/logic, hard for open-ended writing, helpfulness, or safety.
- **Reward sparsity & credit assignment.** A single 0/1 outcome reward for a long trajectory gives coarse signal; assigning credit to specific reasoning steps is unsolved (PRMs help but can be hacked).
- **Length / "overthinking."** Models can game length, produce repetitive or unnecessarily long chains, or mix languages.
- **Exploration & collapse.** Entropy can collapse, killing exploration; KL/entropy control and sampling diversity matter.
- **Compute cost.** Generating many long rollouts per prompt is expensive (inference-heavy RL).

## Solutions & Current Best Practices
**GRPO (or PPO/variants) + RLVR** on math/code/logic with **outcome verifiers**, optionally with a **cold-start long-CoT SFT** (R1-style) and **format/length rewards**. Use **rejection sampling** of correct traces to also do SFT/distillation. Mix verifiable RL with a general RLHF/preference stage for non-verifiable behavior. Many 2025 variants refine GRPO: **decoupled clipping / dynamic sampling (DAPO)**, **token-level loss normalization**, removing the std normalization or KL term (Dr. GRPO), and length penalties to curb overthinking. Distill the resulting reasoning into smaller dense models (DeepSeek-R1-Distill).

## Lab Perspectives
- **DeepSeek:** originated GRPO (DeepSeekMath) and the R1 pure-RL reasoning paradigm; deliberately used **rule-based verifiers over neural PRMs** to avoid reward hacking.
- **OpenAI:** o1/o3 are RL-on-reasoning models (methods undisclosed but conceptually RLVR-aligned); pioneered the "RL + inference-time compute" scaling story.
- **AllenAI (Tülu 3):** open RLVR recipe, formalizing "RL with verifiable rewards" as a named method.
- **Google DeepMind:** verifiable reasoning via formal systems (AlphaProof/AlphaGeometry) and RL; Gemini "thinking" models.
- **Qwen/others:** widely adopted GRPO for open reasoning models (QwQ, Qwen-Math).

## Latest Developments (2023–2026)
An explosion of GRPO variants in 2025 (DAPO, Dr. GRPO, GSPO, VAPO) targeting stability, length control, and exploration. Debate over whether RL **elicits latent** reasoning vs **teaches new** reasoning (the "RL only sharpens the base model" critique vs "RL discovers new behaviors"). Extension of RLVR beyond math/code to **agentic/tool-use** tasks with execution-based rewards, and to **semi-verifiable** domains via LLM judges. Reasoning-model post-training compute now rivals pretraining.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How does GRPO differ from PPO?"** No critic; group-relative normalized advantage as baseline; cheaper, stable for outcome rewards.
- **"Write the GRPO advantage."** $\hat A_i=(r_i-\text{mean})/\text{std}$ over the sampled group.
- **"Why are verifiable rewards a big deal?"** Non-hackable, reliable signal → stable large-scale RL → emergent long CoT.
- **"What did R1-Zero demonstrate?"** Reasoning (long CoT, self-correction) emerges from pure RL with no SFT.
- **"Limits of RLVR?"** Needs checkable rewards; sparse credit assignment; hard for open-ended tasks.

## Open Problems
Whether RL *creates* new capabilities or only *amplifies* latent ones in the base model is hotly debated. Credit assignment for long reasoning (robust process rewards without hacking), extending verifiable RL to open-ended/agentic domains, controlling overthinking, and the true compute-optimal balance of RL vs distillation vs inference-time search are all open.

## References
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Lambert, N. et al. (2024). *Tülu 3 (RLVR).* arXiv:2411.15124.
- Yu, Q. et al. (2025). *DAPO.* arXiv:2503.14476.
- Liu, Z. et al. (2025). *Understanding R1-Zero-Like Training (Dr. GRPO).* arXiv:2503.20783.
