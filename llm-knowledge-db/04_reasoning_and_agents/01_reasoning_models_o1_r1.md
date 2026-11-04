# Reasoning Models (o1 / R1)

> **Last Updated:** 2026-11-04
> **Related Files:** [Chain-of-Thought](00_chain_of_thought.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Inference-Time Scaling](02_inference_time_scaling.md) · [Agentic Coding & Computer Use](06_agentic_coding_and_computer_use.md)
> **Key Papers:** OpenAI 2024 "Learning to Reason with LLMs" (o1) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · Lightman et al. 2023 PRM ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) · Snell et al. 2024 Test-Time Compute ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314)) · DeepSeek-AI 2025 DeepSeekMath-V2 ([arXiv:2511.22570](https://arxiv.org/abs/2511.22570))

## Overview
"Reasoning models" (OpenAI o1/o3 and the GPT-5/GPT-6 thinking models, DeepSeek-R1 and its hybrid successors, Gemini "thinking"/Deep Think, Claude with extended → adaptive thinking, Qwen QwQ/Qwen3, Kimi K2 Thinking) are the defining LLM development of 2024–2026. They are trained — primarily via reinforcement learning on verifiable problems — to spend a large, variable amount of **inference-time compute** generating an internal chain of thought before answering. The result is a step-change on hard math, competition coding, and science (o1/o3 reaching expert-level on AIME, Codeforces, GPQA; R1 matching o1 openly; natural-language IMO 2025 gold). The paradigm shift is from "scale training compute" to "**also scale test-time compute**," and from "elicit reasoning by prompting" to "**train reasoning by RL**." By late 2026 "reasoning model" is no longer a separate product line: frontier models are **hybrid** (thinking on demand or always on), and the reasoning budget is a priced, user-facing control.

## Core Concepts
**Thinking tokens.** The model generates a long internal "reasoning" segment (often hidden from the user) — exploring approaches, checking work, backtracking — then a concise final answer. More thinking tokens generally means higher accuracy on hard problems: a new scaling axis (see [Inference-Time Scaling](02_inference_time_scaling.md)).

**Training via RL on verifiable rewards.** The dominant recipe (explicit in R1, inferred for o1): take a strong base model and run large-scale **RL (GRPO/PPO) with verifiable rewards** (math answer-checking, code unit tests) so the model *learns* to produce reasoning that leads to correct answers (see [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)). Long CoT, self-verification, and "**aha moments**" (the model realizing and correcting an error) **emerge** from this optimization — R1-Zero showed this happens even with *no* reasoning SFT.

**The R1 recipe (open blueprint).** (1) Optional **cold-start SFT** on a small set of high-quality long-CoT examples (for readability); (2) **large-scale RLVR** on math/code/logic; (3) **rejection sampling** of the RL model's best traces to build broad SFT data; (4) a final **RLHF** stage for general helpfulness/safety. DeepSeek then **distilled** R1's reasoning into smaller dense models (Qwen/Llama-based) that inherited strong reasoning cheaply.

**PRM vs ORM and search.** **Outcome reward models** score final answers; **process reward models** (Lightman et al.) score each step, enabling **verifier-guided search** (best-of-N, beam, MCTS over reasoning steps). o1-style training and inference can combine learned/verifiable rewards with search. Notably, DeepSeek-R1 reported that **rule-based outcome rewards** worked better than neural PRMs for RL (PRMs got hacked), though PRMs remain useful for inference-time verification. For proofs, where answers aren't checkable, DeepSeekMath-V2 trains an **LLM verifier** and rewards the generator for finding and fixing its own errors.

**MCTS for reasoning.** Tree search (à la AlphaGo) over reasoning steps with a value model was an influential hypothesis for o1; in practice, simpler long-CoT RL (R1) proved highly effective, and the role of explicit MCTS at training time is debated.

**Hybrid thinking and effort control.** One checkpoint now serves several reasoning budgets: a router choosing between fast and thinking models (GPT-5), explicit Think/Non-Think modes (DeepSeek-V3.1 → V4), or an **effort** parameter with adaptive thinking (Claude Opus 4.5 → 5.5). Accuracy is reported alongside tokens-per-task.

## Key Challenges
- **Verifiable-domain dependence.** The clean RL signal exists for math/code/logic; extending to open-ended reasoning (law, medicine, writing) is hard.
- **Overthinking & cost.** Models can burn excessive tokens on easy problems; latency and inference cost rise sharply.
- **Faithfulness & safety of hidden CoT.** If reasoning is hidden/unfaithful, monitoring is harder; obfuscated reasoning could hide misbehavior (an active safety concern).
- **Reward hacking / language mixing.** R1-Zero produced unreadable, language-mixed chains; rewards must be shaped for usability.
- **Generalization.** Does math/code RL transfer to general reasoning? Partly yes, but the boundaries are unclear.
- **Benchmark saturation.** Olympiad math, ARC-AGI-2, and GPQA-style tests saturated within ~12–18 months; interactive novel-environment tests (ARC-AGI-3) remain hard.

## Solutions & Current Best Practices
**RLVR (GRPO-family) on verifiable tasks + cold-start CoT SFT + final RLHF**, with **rejection sampling** to bootstrap data and **distillation** (increasingly on-policy) to make small reasoning models. **Adaptive thinking** (reasoning effort proportional to difficulty) curbs overthinking; train one hybrid model with explicit modes or effort levels rather than separate reasoning and chat models. **Inference-time search** (self-consistency, best-of-N with verifiers) complements training. Keep CoT **monitorable** where possible for safety. Distillation from a strong reasoner is, per DeepSeek, often more compute-efficient than RL-ing a small model directly. Invest in **midtraining** data: it largely sets post-RL ceilings.

## Lab Perspectives
- **OpenAI:** o1 (2024) launched the paradigm; o3/o4 pushed frontier math/science/agentic coding; GPT-5 (2025-08) unified fast and thinking models behind a real-time router; GPT-6 Astra (2026-09) attributes gains to scaled pretraining plus long-horizon RL; emphasizes CoT monitoring and hides raw CoT.
- **DeepSeek:** R1 gave the field an **open, reproducible** reasoning recipe (GRPO + RLVR); V3.1 merged Think/Non-Think into one model; V3.2-Speciale claimed gold-level IMO/ICPC/IOI 2025; V4 (2026-04) keeps both modes; no "R2" has been released.
- **Google DeepMind:** Gemini "thinking" and **Deep Think**, which won officially graded IMO 2025 gold in natural language; Gemini 3.1 Pro reported 77.1% on ARC-AGI-2 (2026-02).
- **Anthropic:** Claude "extended thinking" → effort parameter (Opus 4.5) → adaptive thinking (Opus 4.6) → always-on thinking with "preserved thinking" (Opus 5.5); research on CoT faithfulness/monitorability.
- **Qwen / Moonshot / others:** open reasoning models (QwQ, Qwen3, Kimi K2 Thinking with INT4 QAT during post-training); Qwen3.8 open weights ship with thinking always on.

## Latest Developments (2023–Nov 2026)
Reasoning RL became the frontier's main lever; **post-training compute now rivals pretraining**. Explosion of **GRPO variants** (DAPO, Dr. GRPO). **Reasoning distillation** made small models strong. Extension to **agentic/tool-use** reasoning with execution rewards. Debate intensified over **"does RL create new reasoning or elicit latent ability?"** and over CoT **faithfulness/monitoring** as a safety pillar. Inference-time-compute scaling laws (Snell et al.) formalized the train-vs-test compute tradeoff.

**Mid-2025 → Nov 2026 update:**
- **Olympiad gold in natural language (2025-07/09).** Gemini Deep Think scored 35/42 at IMO 2025, officially graded, within 4.5 hours and without Lean (AlphaProof in 2024 needed formalization and days of compute); OpenAI's experimental model also announced 35/42 (graded by former medalists). At ICPC 2025 an OpenAI system solved 12/12 and Gemini 2.5 Deep Think 10/12 (news coverage). IMO 2026 reportedly saw multiple 42/42 scores (news only).
- **Self-verifying reasoners (2025-11).** DeepSeekMath-V2 reports IMO 2025 gold-level (5/6) and 118/120 on Putnam 2024 (self-reported) by training a proof verifier as reward ([arXiv:2511.22570](https://arxiv.org/abs/2511.22570)).
- **Hybrid models replace separate reasoners (2025-08 → 2026).** GPT-5's router (whose launch-day malfunction caused a backlash), DeepSeek-V3.1/V4 Think/Non-Think modes, and Claude's adaptive thinking; Opus 5.5 (2026-09) removes the no-thinking mode entirely.
- **Effort as a priced control (2025-11 → 2026).** Opus 4.5 at medium effort matched Sonnet 4.5's best SWE-bench Verified score with 76% fewer output tokens; later releases added `xhigh` (Opus 4.7) and effort on small models (Haiku 5.5); vendors now report tokens per task alongside accuracy.
- **RL scaling science (2025-10 → 2026-03).** ScaleRL makes RL gains predictable from early compute ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786)); PRISM shows midtraining data, more than the RL mix, determines post-RL GPQA ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)).
- **Saturation and new tests (2026).** ARC-AGI-2 is effectively saturated (frontier API models listed at 90–95% by aggregators), while ARC-AGI-3 (interactive, goal-inference) launched 2026-03 with every frontier model under 1%.
- **Distillation defenses (2026).** Account-bound "preserved thinking," distillation-detecting classifiers, and restricted model tiers limit the reasoning traces available to outside distillers.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How are o1/R1 trained, conceptually?"** RL (GRPO/PPO) with verifiable rewards on a strong base → emergent long CoT; R1 adds cold-start SFT + final RLHF.
- **"What did R1-Zero show?"** Reasoning emerges from pure RL, no SFT — long CoT, self-correction, 'aha moments.'
- **"Why is test-time compute a new scaling axis?"** Accuracy scales with reasoning tokens / search; trade train compute for inference compute.
- **"PRM vs ORM for reasoning, and the hacking caveat?"** Process gives denser signal/search; but neural PRMs can be hacked in RL → R1 used rule-based outcome rewards.
- **"How would you train one model with both thinking and non-thinking modes?"** Mode tokens in the chat template, mixed SFT data for both modes, RL with length/effort-conditioned rewards, and evaluation of the accuracy–tokens frontier per mode.

## Open Problems
Whether RL *creates* novel reasoning or *amplifies latent* capability is unresolved. Extending verifiable-reward reasoning to open-ended domains, guaranteeing **faithful, monitorable** CoT, controlling overthinking, and the true compute-optimal mix of RL training, distillation, and inference-time search are the field's central open questions. With olympiad math saturated, how to measure reasoning on genuinely novel, interactive problems (ARC-AGI-3-style) is open.

## References
- OpenAI (2024). *Learning to Reason with LLMs (o1).*
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050.
- Snell, C. et al. (2024). *Scaling LLM Test-Time Compute Optimally.* arXiv:2408.03314.
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- DeepSeek-AI (2025). *DeepSeekMath-V2.* arXiv:2511.22570.
- Meta et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
