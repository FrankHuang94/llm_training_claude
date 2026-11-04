# Agentic Frameworks

> **Last Updated:** 2026-11-04
> **Related Files:** [Tool Use & Function Calling](03_tool_use_and_function_calling.md) · [Multi-Agent Systems](05_multi_agent_systems.md) · [Agentic Coding & Computer Use](06_agentic_coding_and_computer_use.md) · [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)
> **Key Papers:** Yao et al. 2023 ReAct ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) · Shinn et al. 2023 Reflexion ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) · Wang et al. 2023 Voyager ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) · Yao et al. 2024 τ-bench ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)) · Kwa et al. 2025 METR Time Horizons ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499))

## Overview
An **agent** is an LLM placed in a loop where it perceives a state, reasons, takes actions (tool calls), observes results, and iterates toward a goal with minimal human intervention. Agentic frameworks (LangChain/LangGraph, AutoGen, CrewAI, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, smolagents) provide the scaffolding — the control loop, memory, tool integration, planning, and orchestration — that turns a chat model into a goal-directed system. The 2023 "AutoGPT/BabyAGI" wave overpromised; the 2024–2026 reality is more disciplined: agents now work well in **constrained, verifiable, tool-rich domains** (coding, research, customer support) and are a primary commercial focus across labs. By late 2026 frontier agents sustain coherent work for many hours (vendors report sessions of 18–30+ hours), METR's 50% time horizon for frontier models exceeds ~16 hours, and the harness has thinned to a general loop around a model whose agentic skill lives mostly in the weights.

## Core Concepts
**The agent loop.** Core components: (1) **planning** (decompose the goal, decide next action), (2) **action/tool execution**, (3) **observation/feedback** ingestion, (4) **memory** (state across steps), iterated until termination. ReAct is the minimal loop; richer frameworks add explicit planners and state machines.

**Planning patterns.** *ReAct* (interleaved reason-act), *Plan-and-Execute* (make a full plan, then execute steps — better for long horizons), *Tree/graph search* over actions, and *hierarchical* planning (high-level planner + low-level executors). LangGraph and similar model the agent as a **graph/state machine** for controllability.

**Memory.** Agents need memory beyond the context window: **short-term** (scratchpad, conversation), **long-term** (vector-store retrieval of past experiences), and **episodic/procedural** (learned skills). MemGPT-style "virtual context management" pages information in/out (see [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)). In 2026, memory operations are increasingly *learned* tool actions trained with RL.

**Context engineering.** Long runs exceed any context window, and quality degrades as context fills ("context rot"). The standard toolkit: **compaction** (summarize and restart the window), **structured note-taking** to files, **just-in-time retrieval** through tools, and **subagents** with clean contexts that return condensed results.

**Reflection / self-correction.** *Reflexion* (Shinn et al.): after a failed attempt, the agent **verbally reflects** on what went wrong, stores the reflection in memory, and retries — learning within a task without weight updates. A powerful pattern for improving reliability.

**Skill acquisition.** *Voyager* (Minecraft): an agent that writes, tests, and stores reusable code "skills" in a growing library, exhibiting open-ended self-improvement — an influential demonstration of lifelong agentic learning. Agent Skills (SKILL.md folders, 2025) productize the same idea for general agents.

**Time horizons.** METR measures the human task duration $t$ at which an agent succeeds 50% of the time by fitting $P(\text{success})=\sigma\big(\alpha-\beta\log t\big)$ over a suite of timed tasks; the horizon is where the fit crosses 0.5.

## Key Challenges
- **Compounding errors / brittleness.** Long-horizon tasks fail because per-step error rates compound; a 95%-reliable step over 20 steps ≈ 36% success. Reliability is the central blocker.
- **Planning & long-horizon coherence.** Models struggle to plan many steps ahead, recover from off-track states, and maintain goal coherence.
- **Memory management.** Deciding what to store/retrieve/forget; context bloat vs information loss.
- **Evaluation.** Agent benchmarks are noisy, environment-dependent, and quickly saturate or contaminate; reproducibility is hard. Vendor scaffolds can score 10–30 points above standardized ones.
- **Cost & latency.** Multi-step loops with large models are expensive and slow.
- **Safety.** Autonomous actions amplify risks (unsafe operations, prompt injection, runaway loops); 2026 brought real incidents from agents with unintended internet access.

## Solutions & Current Best Practices
**Constrain the domain and use verifiable feedback** (code execution, environment rewards) — agents shine where actions are checkable. Prefer **explicit, controllable orchestration** (state machines/graphs, LangGraph) over fully autonomous "do anything" loops for products; for frontier coding agents, use a **thin general harness** (bash, file edit, search, subagents) around an RL-trained model. Add **reflection/retry**, **structured memory + retrieval**, **compaction**, and **guardrails** (permissions, sandboxes, human-in-the-loop for risky actions, online monitors that screen actions before execution). Increasingly, **train** agents end-to-end with RL on execution rewards, which yields more robust behavior. Evaluate on realistic, versioned benchmarks (SWE-bench Pro, Terminal-Bench, τ-bench, WebArena, GAIA, OSWorld) with standardized harnesses.

## Lab Perspectives
- **OpenAI:** Agents SDK, Operator/computer-use, "Deep Research" agent, Codex; GPT-5.4 native computer use; GPT-6 Astra gains attributed partly to long-horizon RL.
- **Anthropic:** Claude Code and the **Claude Agent SDK** (2025-09, the Claude Code harness opened for general agents), MCP and Agent Skills, context compaction and agent teams (2026-02); Haiku 5.5 is positioned as a subagent/compaction model; emphasis on agent safety and monitoring.
- **Google DeepMind:** Gemini agents, Project Mariner (web), **Antigravity** (2025-11, an agent-first IDE that also runs Claude and gpt-oss agents), Gemini Spark persistent agent (announced 2026-05).
- **Open ecosystem:** LangChain/LangGraph, Microsoft AutoGen, CrewAI, HuggingFace smolagents provide the framework layer; open agent models (Qwen-Agent, GLM, Kimi).

## Latest Developments (2023–Nov 2026)
The field moved from **prompt-orchestrated** agents (LangChain-style) toward **RL-trained** agentic models where capability lives in the weights, with frameworks providing thinner orchestration. **Coding agents** (SWE-bench scores rising sharply) and **deep-research / web agents** became flagship products. **Computer-use** (GUI control via screenshots) emerged. Standardization via **MCP**. Strong focus on **reliability** (reducing compounding error) and **long-horizon** competence as the key frontier.

**Mid-2025 → Nov 2026 update:**
- **Horizons keep doubling (2025-03 → 2026).** METR's 50% time horizon doubled ~every 7 months over 2019–2025 ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499)); the TH1.1 update (2026-01) puts post-2023 doubling at ~4.3 months (secondary summaries). An early Claude Mythos Preview snapshot measured "at least 16 hrs (95% CI 8.5–55 hrs)"; METR says its suite is near saturation above ~16 h.
- **Hours-long autonomy claims (2025-09 → 2026-09).** Sonnet 4.5 stayed focused for >30 hours on multi-step tasks; a tester ran Opus 5.5 autonomously for >18 hours (vendor-reported).
- **Harnesses become SDKs (2025-09 → 11).** Claude Agent SDK opened the Claude Code harness; Google Antigravity shipped as a multi-vendor agent IDE alongside Gemini 3.
- **Context compaction and memory (2025-09 → 2026-02).** Anthropic's context-engineering guidance (compaction, notes, subagents) became product features (compaction beta in Opus 4.6); AgeMem trains memory operations as RL-learned tool actions ([arXiv:2601.01885](https://arxiv.org/abs/2601.01885)); Memory-R1 learns ADD/UPDATE/DELETE/NOOP from ~150 QA pairs.
- **Environment scaling for agentic RL (2026).** SWE-Master ([arXiv:2602.03411](https://arxiv.org/abs/2602.03411)), KAT-Coder-V2.5 with harness randomization ([arXiv:2607.05471](https://arxiv.org/abs/2607.05471)), and CodeMidas ([arXiv:2609.22068](https://arxiv.org/abs/2609.22068)) synthesize verifiable environments — now the main bottleneck.
- **Benchmark churn (2026).** OpenAI stopped reporting SWE-bench Verified (2026-02) after finding ≥59.4% of audited hard problems had flawed tests plus contamination; Terminal-Bench went 2.0 → 4.0 in about a year.
- **Agents at scale need monitors (2026-08).** Anthropic reports ~30K concurrent internal agents, with 100% of actions passing an online monitor before execution (0.002% of >1B decisions blocked in Aug 2026).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Describe the components of an LLM agent."** Planning, tool/action execution, observation, memory, in a loop with a termination condition.
- **"Why do long-horizon agents fail, and how do you improve reliability?"** Compounding per-step errors; verifiable feedback, reflection/retry, constrained scope, better planning, RL training.
- **"What is Reflexion?"** Verbal self-reflection on failures stored in memory to improve subsequent attempts without weight updates.
- **"Prompt-orchestration vs trained agents — tradeoffs?"** Frameworks are flexible but brittle; RL-trained agents are more robust but costly to build.
- **"How do you keep a 20-hour agent run coherent with a finite context?"** Compaction, external notes/files, just-in-time retrieval, subagents with clean contexts, periodic self-verification; measure with time-horizon-style evals.

## Open Problems
**Reliable long-horizon autonomy** is the defining unsolved problem — robust planning, error recovery, and goal coherence over many steps. Effective open-ended **memory/continual learning**, trustworthy **agent evaluation** (as horizon suites saturate), and **safe autonomy** (preventing harmful actions under adversarial conditions) are all active and largely unsolved.

## References
- Yao, S. et al. (2023). *ReAct.* arXiv:2210.03629.
- Shinn, N. et al. (2023). *Reflexion.* arXiv:2303.11366.
- Wang, G. et al. (2023). *Voyager.* arXiv:2305.16291.
- Yao, S. et al. (2024). *τ-bench.* arXiv:2406.12045.
- Jimenez, C. et al. (2023). *SWE-bench.* arXiv:2310.06770.
- Kwa, T. et al. (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- AgeMem authors (2026). *AgeMem.* arXiv:2601.01885.
