# Reasoning Models (o1 / R1)

> **Last Updated:** 2026-06-18
> **Related Files:** [Chain-of-Thought](00_chain_of_thought.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Inference-Time Scaling](02_inference_time_scaling.md)
> **Key Papers:** OpenAI 2024 "Learning to Reason with LLMs" (o1) · DeepSeek-AI 2025 DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) · Lightman et al. 2023 PRM ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) · Snell et al. 2024 Test-Time Compute ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314))

## Overview
"Reasoning models" (OpenAI o1/o3, DeepSeek-R1, Gemini "thinking," Claude with extended thinking, Qwen QwQ) are the defining LLM development of 2024–2026. They are trained — primarily via reinforcement learning on verifiable problems — to spend a large, variable amount of **inference-time compute** generating an internal chain of thought before answering. The result is a step-change on hard math, competition coding, and science (o1/o3 reaching expert-level on AIME, Codeforces, GPQA; R1 matching o1 openly). The paradigm shift is from "scale training compute" to "**also scale test-time compute**," and from "elicit reasoning by prompting" to "**train reasoning by RL**." This is the hottest interview topic in 2026.

## Core Concepts
**Thinking tokens.** The model generates a long internal "reasoning" segment (often hidden from the user) — exploring approaches, checking work, backtracking — then a concise final answer. More thinking tokens generally means higher accuracy on hard problems: a new scaling axis (see [Inference-Time Scaling](02_inference_time_scaling.md)).

**Training via RL on verifiable rewards.** The dominant recipe (explicit in R1, inferred for o1): take a strong base model and run large-scale **RL (GRPO/PPO) with verifiable rewards** (math answer-checking, code unit tests) so the model *learns* to produce reasoning that leads to correct answers (see [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)). Long CoT, self-verification, and "**aha moments**" (the model realizing and correcting an error) **emerge** from this optimization — R1-Zero showed this happens even with *no* reasoning SFT.

**The R1 recipe (open blueprint).** (1) Optional **cold-start SFT** on a small set of high-quality long-CoT examples (for readability); (2) **large-scale RLVR** on math/code/logic; (3) **rejection sampling** of the RL model's best traces to build broad SFT data; (4) a final **RLHF** stage for general helpfulness/safety. DeepSeek then **distilled** R1's reasoning into smaller dense models (Qwen/Llama-based) that inherited strong reasoning cheaply.

**PRM vs ORM and search.** **Outcome reward models** score final answers; **process reward models** (Lightman et al.) score each step, enabling **verifier-guided search** (best-of-N, beam, MCTS over reasoning steps). o1-style training and inference can combine learned/verifiable rewards with search. Notably, DeepSeek-R1 reported that **rule-based outcome rewards** worked better than neural PRMs for RL (PRMs got hacked), though PRMs remain useful for inference-time verification.

**MCTS for reasoning.** Tree search (à la AlphaGo) over reasoning steps with a value model was an influential hypothesis for o1; in practice, simpler long-CoT RL (R1) proved highly effective, and the role of explicit MCTS at training time is debated.

## Key Challenges
- **Verifiable-domain dependence.** The clean RL signal exists for math/code/logic; extending to open-ended reasoning (law, medicine, writing) is hard.
- **Overthinking & cost.** Models can burn excessive tokens on easy problems; latency and inference cost rise sharply.
- **Faithfulness & safety of hidden CoT.** If reasoning is hidden/unfaithful, monitoring is harder; obfuscated reasoning could hide misbehavior (an active safety concern).
- **Reward hacking / language mixing.** R1-Zero produced unreadable, language-mixed chains; rewards must be shaped for usability.
- **Generalization.** Does math/code RL transfer to general reasoning? Partly yes, but the boundaries are unclear.

## Solutions & Current Best Practices
**RLVR (GRPO) on verifiable tasks + cold-start CoT SFT + final RLHF**, with **rejection sampling** to bootstrap data and **distillation** to make small reasoning models. **Adaptive thinking** (reasoning effort proportional to difficulty) curbs overthinking. **Inference-time search** (self-consistency, best-of-N with verifiers) complements training. Keep CoT **monitorable** where possible for safety. Distillation from a strong reasoner is, per DeepSeek, often more compute-efficient than RL-ing a small model directly.

## Lab Perspectives
- **OpenAI:** o1 (2024) launched the paradigm; o3/o4 pushed frontier math/science/agentic coding; emphasizes inference-time-compute scaling laws and CoT monitoring for safety; hides raw CoT.
- **DeepSeek:** R1 gave the field an **open, reproducible** reasoning recipe (GRPO + RLVR), shocking the market; published R1-Zero's pure-RL emergence.
- **Google DeepMind:** Gemini "thinking"/Flash-Thinking; deep formal-reasoning lineage (AlphaProof/AlphaGeometry silver-medal IMO).
- **Anthropic:** Claude "extended thinking" with visible reasoning; research on CoT faithfulness/monitorability.
- **Qwen/others:** open reasoning models (QwQ) via GRPO.

## Latest Developments (2023–2026)
Reasoning RL became the frontier's main lever; **post-training compute now rivals pretraining**. Explosion of **GRPO variants** (DAPO, Dr. GRPO). **Reasoning distillation** made small models strong. Extension to **agentic/tool-use** reasoning with execution rewards. Debate intensified over **"does RL create new reasoning or elicit latent ability?"** and over CoT **faithfulness/monitoring** as a safety pillar. Inference-time-compute scaling laws (Snell et al.) formalized the train-vs-test compute tradeoff.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How are o1/R1 trained, conceptually?"** RL (GRPO/PPO) with verifiable rewards on a strong base → emergent long CoT; R1 adds cold-start SFT + final RLHF.
- **"What did R1-Zero show?"** Reasoning emerges from pure RL, no SFT — long CoT, self-correction, 'aha moments.'
- **"Why is test-time compute a new scaling axis?"** Accuracy scales with reasoning tokens / search; trade train compute for inference compute.
- **"PRM vs ORM for reasoning, and the hacking caveat?"** Process gives denser signal/search; but neural PRMs can be hacked in RL → R1 used rule-based outcome rewards.

## Open Problems
Whether RL *creates* novel reasoning or *amplifies latent* capability is unresolved. Extending verifiable-reward reasoning to open-ended domains, guaranteeing **faithful, monitorable** CoT, controlling overthinking, and the true compute-optimal mix of RL training, distillation, and inference-time search are the field's central open questions.

## References
- OpenAI (2024). *Learning to Reason with LLMs (o1).*
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050.
- Snell, C. et al. (2024). *Scaling LLM Test-Time Compute Optimally.* arXiv:2408.03314.
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
