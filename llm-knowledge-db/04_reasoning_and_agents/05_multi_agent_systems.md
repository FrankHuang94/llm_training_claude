# Multi-Agent Systems

> **Last Updated:** 2026-06-18
> **Related Files:** [Agentic Frameworks](04_agentic_frameworks.md) · [Tool Use & Function Calling](03_tool_use_and_function_calling.md) · [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)
> **Key Papers:** Du et al. 2023 Multiagent Debate ([arXiv:2305.14325](https://arxiv.org/abs/2305.14325)) · Wang et al. 2024 Mixture-of-Agents ([arXiv:2406.04692](https://arxiv.org/abs/2406.04692)) · Wu et al. 2023 AutoGen ([arXiv:2308.08155](https://arxiv.org/abs/2308.08155)) · Irving et al. 2018 AI Safety via Debate ([arXiv:1805.00899](https://arxiv.org/abs/1805.00899))

## Overview
Multi-agent systems compose several LLM instances — often with distinct roles, prompts, or even different base models — that interact to solve a task that a single agent handles less well. The motivations are diverse: **error correction** through cross-examination (debate), **specialization** (planner/coder/critic/tester division of labor), **ensembling** for quality (mixture-of-agents), and **scalable oversight** (using AI agents to check each other). Multi-agent designs can outperform single agents on reasoning and complex software tasks, but they multiply cost and introduce coordination failure modes. The paradigm also has deep ties to alignment research (Debate as a scalable-oversight proposal).

## Core Concepts
**Multi-agent debate.** Multiple model instances independently answer, then **read each other's answers and reasoning and revise** over several rounds, converging toward a consensus. Du et al. (2023) showed this improves factuality and reasoning by surfacing and correcting individual errors. It operationalizes "many minds catch more mistakes."

**Role-based / cooperative agents.** Assign specialized roles — e.g., **planner, executor, critic, verifier** — that pass messages (AutoGen's conversable agents, CrewAI's role crews, MetaGPT's software-company metaphor with PM/architect/engineer roles). Specialization + a critic loop improves complex, structured tasks (especially coding).

**Mixture-of-Agents (MoA).** Wang et al. (2024): layer multiple LLMs so that "proposer" models generate candidate responses and "aggregator" models synthesize them, iterating across layers. Leverages the finding that LLMs improve when shown other models' outputs; an open MoA stack rivaled GPT-4-class quality by combining several open models.

**Debate for oversight (Irving et al.).** A *safety* proposal: two agents argue opposing sides of a question before a (possibly weaker) judge; if honest argumentation is easier to defend than dishonest, debate lets a limited judge supervise more capable agents — a route to scalable oversight (see [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)).

**Orchestration.** A controller (or graph/state machine) routes messages, decides turn-taking, and terminates. Topologies range from flat (all-to-all debate) to hierarchical (manager → workers) to pipelines.

## Key Challenges
- **Cost multiplication.** $k$ agents over $r$ rounds means $k\times r$ (or more) model calls — often the same accuracy is achievable by one strong model thinking longer (single-agent inference-time scaling is a strong baseline).
- **Error propagation & groupthink.** Agents can reinforce shared mistakes or sycophantically converge on a wrong consensus; diversity is essential and hard to guarantee.
- **Coordination failures.** Miscommunication, role confusion, infinite loops, and dropped context across agents.
- **Diminishing returns.** Benefits often saturate after 2–3 agents/rounds; gains may not justify cost.
- **Evaluation & attribution.** Hard to measure and to attribute credit/blame across agents.

## Solutions & Current Best Practices
Use multi-agent designs where **diversity and verification** add real value: cross-checking for factuality, specialized pipelines for software (planner+coder+tester with execution feedback), and aggregation (MoA) when combining complementary models. Ensure **genuine diversity** (different prompts/models/temperatures) to avoid groupthink. Add a **verifier/critic with ground-truth feedback** (tests, tools) rather than pure self-evaluation. Cap rounds (returns saturate). Always **benchmark against a single strong agent with equal compute** — many reported multi-agent wins disappear under that control. Prefer structured orchestration (graphs) for reliability.

## Lab Perspectives
- **Anthropic:** researches **Debate** as scalable oversight; uses multi-agent patterns in research workflows (e.g., multi-agent research systems).
- **OpenAI/Google DeepMind:** explore debate and critique-model approaches for oversight; DeepMind ran debate experiments empirically.
- **Microsoft:** AutoGen (conversable multi-agent framework) and MetaGPT-style role agents.
- **Open community:** CrewAI, AutoGen, MoA stacks; popular for agentic products despite cost concerns.

## Latest Developments (2023–2026)
Empirical scrutiny tempered early hype: several studies found a **single strong reasoning model with more inference compute matches or beats** elaborate multi-agent setups on many benchmarks, shifting interest toward (a) **specialized, verifier-grounded** multi-agent pipelines (coding) and (b) multi-agent as a **scalable-oversight** mechanism rather than a raw capability booster. **Debate** gained renewed attention as frontier models approach human-level judging. Multi-agent **RL training** (training agents to cooperate/argue, not just prompt them) is an emerging direction.

## Interview Angles
> 💡 **What labs actually ask:**
- **"When does multi-agent debate help, and when is it just expensive?"** Helps via diverse error-correction/verification; often matched by single-agent inference-time scaling at equal compute — always control for that.
- **"Explain Debate as scalable oversight (Irving et al.)."** Two agents argue; a weaker judge decides; relies on honesty being easier to defend.
- **"What is Mixture-of-Agents?"** Layered propose-and-aggregate across multiple LLMs to synthesize better answers.
- **"Main failure modes of multi-agent systems?"** Cost blow-up, groupthink/error propagation, coordination loops, saturating returns.

## Open Problems
Whether multi-agent collaboration yields capabilities **beyond** a single strong model given equal compute is genuinely contested. Guaranteeing productive diversity (no groupthink), reliable coordination at scale, and whether **Debate** actually incentivizes honesty over persuasive deception (a load-bearing assumption for oversight) are key open questions.

## References
- Du, Y. et al. (2023). *Improving Factuality and Reasoning via Multiagent Debate.* arXiv:2305.14325.
- Wang, J. et al. (2024). *Mixture-of-Agents.* arXiv:2406.04692.
- Wu, Q. et al. (2023). *AutoGen.* arXiv:2308.08155.
- Irving, G. et al. (2018). *AI Safety via Debate.* arXiv:1805.00899.
- Hong, S. et al. (2023). *MetaGPT.* arXiv:2308.00352.
