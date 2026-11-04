# GRPO and RLVR (Reasoning RL)

> **Last Updated:** 2026-11-04
> **Related Files:** [RLHF](03_RLHF.md) · [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [Agentic Coding & Computer Use](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md) · [DeepSeek Roadmap](../08_lab_roadmaps/04_deepseek_roadmap.md)
> **Key Papers:** Shao et al. 2024 DeepSeekMath/GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · Lambert et al. 2024 Tülu 3 / RLVR ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) · Yu et al. 2025 DAPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) · ScaleRL 2025 ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786))

## Overview
GRPO (Group Relative Policy Optimization) and RLVR (Reinforcement Learning with Verifiable Rewards) are the algorithmic core of the 2024–2026 reasoning revolution. The central idea is to apply RL not against a fragile learned reward model, but against an **objective verifier** — a math grader, code unit tests, a symbolic checker — that returns a reliable, hard-to-hack reward. This lets models *learn to reason* by trial and error: generate long chains of thought, get rewarded for correct final answers, and reinforce the reasoning that works. DeepSeek-R1 stunned the field by showing that **pure RL with verifiable rewards**, with no reasoning-specific supervised data, induces emergent long chain-of-thought, self-verification, and "aha moments." Through 2025–26 the field converged on a family of GRPO refinements, learned to predict RL scaling from small runs, and extended RLVR to long-horizon agentic environments — the main capability-scaling axis of late 2026.

## Core Concepts
**RLVR (verifiable rewards).** The reward is computed by a *verifier* for domains with checkable answers: `reward = 1` if the final answer matches ground truth (math) or passes all unit tests (code), else `0` (sometimes with format rewards). Because the verifier is correct by construction, the *learned-RM* failure mode of RLHF is largely removed, enabling stable, large-scale RL — though environment exploits (e.g., `sys.exit(0)` to fake passing tests) remain.

**GRPO (the algorithm).** PPO needs a separate **value/critic** network to estimate the baseline for advantages — expensive (doubles model memory) and itself hard to train for long sequences. GRPO eliminates the critic by using a **group-relative baseline**: for each prompt, sample a *group* of $G$ outputs $\{o_1,\dots,o_G\}$, score each with the verifier to get rewards $\{r_i\}$, and compute the advantage by **normalizing within the group**:
$$\hat A_i=\frac{r_i-\text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}$$
The group mean *is* the baseline (an empirical, unbiased estimate of expected reward for that prompt). The policy is then updated with a PPO-style clipped objective using these advantages, plus a KL penalty to the reference:
$$\mathcal{L}_{GRPO}=\mathbb{E}\Big[\tfrac{1}{G}\sum_i \min\big(\rho_i \hat A_i,\ \text{clip}(\rho_i,1\!-\!\epsilon,1\!+\!\epsilon)\hat A_i\big)\Big]-\beta\,D_{KL}(\pi_\theta\|\pi_{ref})$$
This is cheaper (no value net), naturally suited to outcome rewards (advantage shared across all tokens of a sample), and stable.

**The 2025 refinements.** *Dr. GRPO* drops the per-response length normalization and the std division (both bias toward long wrong answers and toward too-easy/too-hard prompts). *DAPO* decouples clipping ($\epsilon_{low}<\epsilon_{high}$, "Clip-Higher") against entropy collapse, drops all-correct/all-wrong groups (dynamic sampling, since $\hat A_i\equiv 0$), uses token-level loss aggregation, and shapes overlong outputs. *CISPO* clips the importance weight itself and stops its gradient, $\mathcal{L}=\mathbb{E}\big[\text{sg}(\text{clip}(\rho_t,1-\epsilon_{low},1+\epsilon_{high}))\,\hat A_t\log\pi_\theta(a_t)\big]$, so rare "fork" tokens keep gradient. *GSPO* replaces token ratios with a length-normalized sequence ratio $s_i=\big(\pi_\theta(o_i|x)/\pi_{\theta_{old}}(o_i|x)\big)^{1/|o_i|}$ and clips at sequence level, removing token-level IS noise that destabilizes MoE RL.

**R1-Zero vs R1.** **R1-Zero** applied GRPO+RLVR directly to a base model with *no SFT* — and long CoT, reflection, and self-correction *emerged* purely from RL, demonstrating reasoning can be learned without demonstrations (though output readability/language-mixing suffered). **R1** added a small **cold-start long-CoT SFT** before RL (and a final RLHF stage for general alignment) to fix readability and broaden capability — the production recipe.

**Why RLVR enables long reasoning.** The verifiable reward only cares about *correctness*, so the model is free to use however many "thinking tokens" it needs; RL discovers that *more deliberate, longer* reasoning yields higher reward, and **test-time compute scales up emergently** (see [Inference-Time Scaling](../04_reasoning_and_agents/02_inference_time_scaling.md)).

**Predictable RL scaling.** ScaleRL fits pass rate vs RL compute $C$ with a saturating sigmoid, $R(C)=R_0+\frac{A-R_0}{1+(C_{mid}/C)^{B}}$, where $A$ is the asymptotic ceiling and $B$ the efficiency exponent; fits on early compute extrapolate to much larger runs.

## Key Challenges
- **Verifiable-domain limitation.** RLVR needs a checkable reward — natural for math/code/logic, hard for open-ended writing, helpfulness, or safety.
- **Reward sparsity & credit assignment.** A single 0/1 outcome reward for a long trajectory gives coarse signal; assigning credit to specific reasoning steps is unsolved (PRMs help but can be hacked).
- **Length / "overthinking."** Models can game length, produce repetitive or unnecessarily long chains, or mix languages.
- **Exploration & entropy collapse.** Without intervention entropy is traded for reward predictably, $R=-a\,e^{H}+b$, driven by the covariance between log-probability and advantage; once entropy is spent, the reward ceiling is fixed.
- **Training–inference mismatch.** Rollout-engine vs trainer numerics (BF16, non-batch-invariant kernels, MoE routing) make nominally on-policy RL off-policy.
- **Compute cost.** Generating many long rollouts per prompt is expensive (inference-heavy RL); agentic rollouts add environment cost and stragglers.
- **Environment hacking.** Test-based and sandbox rewards can be exploited, and hacks can generalize to broader misalignment.

## Solutions & Current Best Practices
**GRPO-family objective + RLVR** on math/code/logic/agentic tasks with **outcome verifiers**, optionally with a **cold-start long-CoT SFT** (R1-style) and **format/length rewards**. The converged 2026 recipe (ScaleRL and peers): token-level or prompt-level loss averaging, asymmetric or IS-weight clipping (DAPO/CISPO) or sequence-level ratios for MoE (GSPO), zero-variance filtering, FP32 LM-head logits, and asynchronous PipelineRL-style training. Control entropy (Clip-Higher, Clip-Cov/KL-Cov, RL↔SFT alternation). Align sampler and trainer numerics (FP16 or batch-invariant kernels, TIS, routing replay). LoRA suffices for RL. Use **rejection sampling** of correct traces for SFT/distillation, and **on-policy distillation** to transfer to smaller models. Mix verifiable RL with rubric/preference RL for non-verifiable behavior; harden environments and apply **inoculation prompting**.

## Lab Perspectives
- **DeepSeek:** originated GRPO (DeepSeekMath) and the R1 pure-RL reasoning paradigm; deliberately used **rule-based verifiers over neural PRMs** in R1, then trained an LLM proof verifier as reward in DeepSeekMath-V2; V3.2-Speciale is claimed gold-level at IMO/ICPC/IOI 2025.
- **OpenAI:** o-series and GPT-5/6 thinking models are RL-on-reasoning (methods undisclosed); an experimental reasoning model announced 35/42 at IMO 2025 and an OpenAI system solved 12/12 at ICPC 2025 (unofficial).
- **AllenAI (Tülu 3):** open RLVR recipe, formalizing "RL with verifiable rewards" as a named method.
- **Google DeepMind:** Gemini Deep Think achieved officially graded IMO 2025 gold (35/42) in natural language within the time limit, without Lean formalization.
- **Qwen / MiniMax / ByteDance Seed:** GSPO (Qwen3), CISPO (MiniMax-M1), DAPO — the main open GRPO successors.
- **Meta et al.:** ScaleRL, the largest public study of RL compute scaling.

## Latest Developments (2023–Nov 2026)
An explosion of GRPO variants in 2025 (DAPO, Dr. GRPO, GSPO, VAPO) targeting stability, length control, and exploration. Debate over whether RL **elicits latent** reasoning vs **teaches new** reasoning (the "RL only sharpens the base model" critique vs "RL discovers new behaviors"). Extension of RLVR beyond math/code to **agentic/tool-use** tasks with execution-based rewards, and to **semi-verifiable** domains via LLM judges. Reasoning-model post-training compute now rivals pretraining.

**Mid-2025 → Nov 2026 update:**
- **Objective convergence (2025-03 → 07).** DAPO reached 50 on AIME24 from Qwen2.5-32B base ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)); CISPO reports ~2× speedup over DAPO ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)); GSPO fixes MoE RL collapse without routing replay ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)).
- **ScaleRL (2025-10).** >400K GPU-hours of ablations; sigmoid compute–performance fits extrapolate to 100K GPU-hour runs; most tricks move efficiency $B$, not ceiling $A$ ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786)). Follow-ups find larger models are more compute- and data-efficient in RL, and RL compute grows as a power law in reasoning depth ([arXiv:2605.06638](https://arxiv.org/abs/2605.06638)).
- **Entropy mechanism (2025-05).** Entropy change is driven by Cov(log-prob, advantage); Clip-Cov/KL-Cov restrain high-covariance tokens ([arXiv:2505.22617](https://arxiv.org/abs/2505.22617)); ComputerRL's Entropulse alternates RL and SFT in long agentic runs ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040)).
- **Numerics and infra (2025).** FP16 instead of BF16 largely removes rollout/trainer mismatch ([arXiv:2510.26788](https://arxiv.org/abs/2510.26788)); AReaL (up to 2.77×, [arXiv:2505.24298](https://arxiv.org/abs/2505.24298)), PipelineRL ([arXiv:2509.19128](https://arxiv.org/abs/2509.19128)), and slime (GLM-5 agentic RL, [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)) make RL asynchronous; verl remains the most common open framework.
- **Olympiad results (2025-07/09).** IMO 2025 gold in natural language end-to-end (Gemini Deep Think, officially graded; OpenAI, graded by former medalists); ICPC 2025: OpenAI 12/12, Gemini 2.5 Deep Think 10/12 (news coverage). IMO 2026 reportedly saw several systems at 42/42 (news only) — olympiad math is effectively saturated as a benchmark.
- **RLVR beyond exact match.** Rubrics as Rewards ([arXiv:2507.17746](https://arxiv.org/abs/2507.17746)) and self-verifying proof RL (DeepSeekMath-V2, [arXiv:2511.22570](https://arxiv.org/abs/2511.22570)).
- **Midtraining vs RL data (2026-03).** PRISM: RL makes sparse updates to ~5% of parameters; midtraining data, not the RL mix, mostly determines post-RL ceilings ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)) — evidence for the "RL sharpens" side of the elicit-vs-create debate.
- **Agentic RLVR (2025-08 → 2026).** Multi-turn RL on real repositories and virtual desktops, and parallel-agent RL (Kimi K2.5 PARL, [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)); environment scaling is now the bottleneck (see [Agentic Coding & Computer Use](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)).

## Interview Angles
> 💡 **What labs actually ask:**
- **"How does GRPO differ from PPO?"** No critic; group-relative normalized advantage as baseline; cheaper, stable for outcome rewards.
- **"Write the GRPO advantage."** $\hat A_i=(r_i-\text{mean})/\text{std}$ over the sampled group.
- **"Why are verifiable rewards a big deal?"** Low-noise, reliable signal → stable large-scale RL → emergent long CoT (but environments can still be hacked).
- **"What did R1-Zero demonstrate?"** Reasoning (long CoT, self-correction) emerges from pure RL with no SFT.
- **"Limits of RLVR?"** Needs checkable rewards; sparse credit assignment; hard for open-ended tasks.
- **"What biases does vanilla GRPO have, and how do DAPO/Dr. GRPO/GSPO fix them?"** Length normalization favors long wrong answers and std division reweights by difficulty (Dr. GRPO); symmetric clipping collapses entropy (Clip-Higher); zero-advantage groups waste compute (dynamic sampling); token-level IS noise destabilizes MoE (GSPO's sequence ratio).
- **"How would you decide whether to scale an RL run?"** Fit ScaleRL's sigmoid on early compute; compare asymptote $A$ vs efficiency $B$ across recipes before committing.

## Open Problems
Whether RL *creates* new capabilities or only *amplifies* latent ones in the base model is hotly debated. Credit assignment for long reasoning (robust process rewards without hacking), extending verifiable RL to open-ended/agentic domains, controlling overthinking, and the true compute-optimal balance of RL vs distillation vs inference-time search are all open. Scaling and hardening RL *environments*, and keeping RL from eroding CoT monitorability, are the new frontier problems.

## References
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Lambert, N. et al. (2024). *Tülu 3 (RLVR).* arXiv:2411.15124.
- Yu, Q. et al. (2025). *DAPO.* arXiv:2503.14476.
- Liu, Z. et al. (2025). *Understanding R1-Zero-Like Training (Dr. GRPO).* arXiv:2503.20783.
- Qwen Team (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- MiniMax (2025). *MiniMax-M1 (CISPO).* arXiv:2506.13585.
- Meta et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
- Cui, G. et al. (2025). *The Entropy Mechanism of RL for Reasoning LMs.* arXiv:2505.22617.
