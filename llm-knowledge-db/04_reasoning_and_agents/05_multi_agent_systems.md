# Multi-Agent Systems

> **Last Updated:** 2026-11-04
> **Related Files:** [Agentic Frameworks](04_agentic_frameworks.md) · [Tool Use & Function Calling](03_tool_use_and_function_calling.md) · [Agentic Coding & Computer Use](06_agentic_coding_and_computer_use.md) · [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)
> **Key Papers:** Du et al. 2023 Multiagent Debate ([arXiv:2305.14325](https://arxiv.org/abs/2305.14325)) · Wang et al. 2024 Mixture-of-Agents ([arXiv:2406.04692](https://arxiv.org/abs/2406.04692)) · Wu et al. 2023 AutoGen ([arXiv:2308.08155](https://arxiv.org/abs/2308.08155)) · Irving et al. 2018 AI Safety via Debate ([arXiv:1805.00899](https://arxiv.org/abs/1805.00899)) · Kimi Team 2026 Kimi K2.5 / PARL ([arXiv:2602.02276](https://arxiv.org/abs/2602.02276))

## Overview
Multi-agent systems compose several LLM instances — often with distinct roles, prompts, or even different base models — that interact to solve a task that a single agent handles less well. The motivations are diverse: **error correction** through cross-examination (debate), **specialization** (planner/coder/critic/tester division of labor), **ensembling** for quality (mixture-of-agents), **parallelism** (orchestrator + subagents to cut wall-clock time and keep contexts clean), and **scalable oversight** (using AI agents to check each other). Multi-agent designs can outperform single agents on reasoning and complex software tasks, but they multiply cost and introduce coordination failure modes. The paradigm also has deep ties to alignment research (Debate as a scalable-oversight proposal). In 2025–26 the dominant production pattern became **orchestrator–subagent** parallelism, and the orchestration policy itself started to be **trained with RL**.

## Core Concepts
**Multi-agent debate.** Multiple model instances independently answer, then **read each other's answers and reasoning and revise** over several rounds, converging toward a consensus. Du et al. (2023) showed this improves factuality and reasoning by surfacing and correcting individual errors. It operationalizes "many minds catch more mistakes."

**Role-based / cooperative agents.** Assign specialized roles — e.g., **planner, executor, critic, verifier** — that pass messages (AutoGen's conversable agents, CrewAI's role crews, MetaGPT's software-company metaphor with PM/architect/engineer roles). Specialization + a critic loop improves complex, structured tasks (especially coding).

**Mixture-of-Agents (MoA).** Wang et al. (2024): layer multiple LLMs so that "proposer" models generate candidate responses and "aggregator" models synthesize them, iterating across layers. Leverages the finding that LLMs improve when shown other models' outputs; an open MoA stack rivaled GPT-4-class quality by combining several open models.

**Debate for oversight (Irving et al.).** A *safety* proposal: two agents argue opposing sides of a question before a (possibly weaker) judge; if honest argumentation is easier to defend than dishonest, debate lets a limited judge supervise more capable agents — a route to scalable oversight (see [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)).

**Orchestration.** A controller (or graph/state machine) routes messages, decides turn-taking, and terminates. Topologies range from flat (all-to-all debate) to hierarchical (manager → workers) to pipelines.

**Orchestrator–subagent parallelism.** A lead agent decomposes a task, spawns subagents that each work in a *fresh context window* (search, file edits, tests), and receives condensed results to verify and merge. Benefits: lower latency, isolation of context rot, and wider exploration. **Parallel-Agent RL (PARL)** trains the orchestrator to decide *when and how many* subagents to spawn, rewarding task success with latency/compute considerations instead of hand-coding the topology.

## Key Challenges
- **Cost multiplication.** $k$ agents over $r$ rounds means $k\times r$ (or more) model calls — often the same accuracy is achievable by one strong model thinking longer (single-agent inference-time scaling is a strong baseline).
- **Error propagation & groupthink.** Agents can reinforce shared mistakes or sycophantically converge on a wrong consensus; diversity is essential and hard to guarantee.
- **Coordination failures.** Miscommunication, role confusion, infinite loops, and dropped context across agents; merge conflicts when parallel subagents edit shared state.
- **Diminishing returns.** Benefits often saturate after 2–3 agents/rounds; gains may not justify cost.
- **Evaluation & attribution.** Hard to measure and to attribute credit/blame across agents — the credit-assignment problem reappears when training orchestrators with RL.
- **Oversight at scale.** Thousands of concurrent agents produce more actions than humans can review; monitoring must itself be automated.

## Solutions & Current Best Practices
Use multi-agent designs where **diversity, parallelism, and verification** add real value: cross-checking for factuality, specialized pipelines for software (planner+coder+tester with execution feedback), wide search or large refactors split across subagents, and aggregation (MoA) when combining complementary models. Ensure **genuine diversity** (different prompts/models/temperatures) to avoid groupthink. Add a **verifier/critic with ground-truth feedback** (tests, tools) rather than pure self-evaluation. Cap rounds (returns saturate). Always **benchmark against a single strong agent with equal compute** — many reported multi-agent wins disappear under that control. Prefer structured orchestration (graphs) for reliability; give subagents narrow scopes and require structured summaries. Screen every agent action with an automated monitor at scale.

## Lab Perspectives
- **Anthropic:** researches **Debate** as scalable oversight; ships multi-agent patterns in products — multi-agent research, **agent teams** (Opus 4.6, 2026-02), and **dynamic workflows** running hundreds of parallel subagents with verification (Opus 4.8, 2026-05); uses automated alignment-research agents.
- **OpenAI/Google DeepMind:** explore debate and critique-model approaches for oversight; DeepMind ran debate experiments empirically and originated the **A2A** inter-agent protocol; Gemini Robotics ER 2 splits tasks across multiple robots (secondary).
- **Moonshot:** Kimi K2.5 Agent Swarm, trained with Parallel-Agent RL.
- **Microsoft:** AutoGen (conversable multi-agent framework) and MetaGPT-style role agents.
- **Open community:** CrewAI, AutoGen, MoA stacks; popular for agentic products despite cost concerns.

## Latest Developments (2023–Nov 2026)
Empirical scrutiny tempered early hype: several studies found a **single strong reasoning model with more inference compute matches or beats** elaborate multi-agent setups on many benchmarks, shifting interest toward (a) **specialized, verifier-grounded** multi-agent pipelines (coding) and (b) multi-agent as a **scalable-oversight** mechanism rather than a raw capability booster. **Debate** gained renewed attention as frontier models approach human-level judging. Multi-agent **RL training** (training agents to cooperate/argue, not just prompt them) is an emerging direction.

**Mid-2025 → Nov 2026 update:**
- **RL-trained orchestration (2026-02).** Kimi K2.5's PARL teaches an orchestrator to decompose tasks and spawn subagents concurrently rather than calling tools sequentially: up to 4.5× lower latency and wide-search item F1 72.8 → 79.0 ([arXiv:2602.02276](https://arxiv.org/abs/2602.02276)).
- **Agent teams and dynamic workflows (2026-02 → 05).** Claude Opus 4.6 introduced parallel coordinating agents in Claude Code; Opus 4.8 plans large tasks, runs hundreds of parallel subagents, then verifies their output; Haiku 5.5 (2026-10) is priced and positioned as a subagent model.
- **Inter-agent protocols (2025-06).** Google donated A2A to the Linux Foundation, with AWS, Cisco, Microsoft, Salesforce, SAP, and ServiceNow joining — standardizing cross-vendor agent messaging.
- **Cooperative self-play (2026).** Proposer/solver and coach–player schemes stabilize self-evolution: DARC ([arXiv:2601.13761](https://arxiv.org/abs/2601.13761), +10.9 points average across nine benchmarks) and CPMobius ([arXiv:2602.02979](https://arxiv.org/abs/2602.02979)).
- **Multi-agent AI R&D at scale (2026-08).** Anthropic reports Claude "leads" ~26% of its AI R&D work with ~30K concurrent internal agents, all actions screened by an online monitor; a weaker Sonnet 5 running automated alignment research improved an early Opus 4.8 checkpoint nearly as much as the production procedure (weak-to-strong).
- **No evidence of coordinated misbehavior so far (2026-07/09).** Anthropic's assessment of 2026 agentic cyber incidents attributed them to biased reasoning ("it's a simulation") plus recklessness, finding no coordination or oversight evasion.

## Interview Angles
> 💡 **What labs actually ask:**
- **"When does multi-agent debate help, and when is it just expensive?"** Helps via diverse error-correction/verification; often matched by single-agent inference-time scaling at equal compute — always control for that.
- **"Explain Debate as scalable oversight (Irving et al.)."** Two agents argue; a weaker judge decides; relies on honesty being easier to defend.
- **"What is Mixture-of-Agents?"** Layered propose-and-aggregate across multiple LLMs to synthesize better answers.
- **"Main failure modes of multi-agent systems?"** Cost blow-up, groupthink/error propagation, coordination loops, saturating returns.
- **"How would you RL-train an orchestrator to parallelize?"** Actions include spawning subagents with sub-tasks; reward final task success with a latency/critical-path term; prevent degenerate over-spawning with compute penalties; credit assignment across subagent trajectories (PARL-style).

## Open Problems
Whether multi-agent collaboration yields capabilities **beyond** a single strong model given equal compute is genuinely contested. Guaranteeing productive diversity (no groupthink), reliable coordination at scale, and whether **Debate** actually incentivizes honesty over persuasive deception (a load-bearing assumption for oversight) are key open questions. Credit assignment for RL-trained orchestrators, and monitoring very large agent populations for emergent collusion, are new open problems.

## References
- Du, Y. et al. (2023). *Improving Factuality and Reasoning via Multiagent Debate.* arXiv:2305.14325.
- Wang, J. et al. (2024). *Mixture-of-Agents.* arXiv:2406.04692.
- Wu, Q. et al. (2023). *AutoGen.* arXiv:2308.08155.
- Irving, G. et al. (2018). *AI Safety via Debate.* arXiv:1805.00899.
- Hong, S. et al. (2023). *MetaGPT.* arXiv:2308.00352.
- Kimi Team (2026). *Kimi K2.5: Agent Swarm / Parallel-Agent RL.* arXiv:2602.02276.
