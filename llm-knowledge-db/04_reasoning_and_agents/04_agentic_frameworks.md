# Agentic Frameworks

> **Last Updated:** 2026-06-18
> **Related Files:** [Tool Use & Function Calling](03_tool_use_and_function_calling.md) · [Multi-Agent Systems](05_multi_agent_systems.md) · [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)
> **Key Papers:** Yao et al. 2023 ReAct ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) · Shinn et al. 2023 Reflexion ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) · Wang et al. 2023 Voyager ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) · Yao et al. 2024 τ-bench ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045))

## Overview
An **agent** is an LLM placed in a loop where it perceives a state, reasons, takes actions (tool calls), observes results, and iterates toward a goal with minimal human intervention. Agentic frameworks (LangChain/LangGraph, AutoGen, CrewAI, LlamaIndex, OpenAI Agents SDK, smolagents) provide the scaffolding — the control loop, memory, tool integration, planning, and orchestration — that turns a chat model into a goal-directed system. The 2023 "AutoGPT/BabyAGI" wave overpromised; the 2024–2026 reality is more disciplined: agents now work well in **constrained, verifiable, tool-rich domains** (coding, research, customer support) and are a primary commercial focus across labs.

## Core Concepts
**The agent loop.** Core components: (1) **planning** (decompose the goal, decide next action), (2) **action/tool execution**, (3) **observation/feedback** ingestion, (4) **memory** (state across steps), iterated until termination. ReAct is the minimal loop; richer frameworks add explicit planners and state machines.

**Planning patterns.** *ReAct* (interleaved reason-act), *Plan-and-Execute* (make a full plan, then execute steps — better for long horizons), *Tree/graph search* over actions, and *hierarchical* planning (high-level planner + low-level executors). LangGraph and similar model the agent as a **graph/state machine** for controllability.

**Memory.** Agents need memory beyond the context window: **short-term** (scratchpad, conversation), **long-term** (vector-store retrieval of past experiences), and **episodic/procedural** (learned skills). MemGPT-style "virtual context management" pages information in/out (see [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)).

**Reflection / self-correction.** *Reflexion* (Shinn et al.): after a failed attempt, the agent **verbally reflects** on what went wrong, stores the reflection in memory, and retries — learning within a task without weight updates. A powerful pattern for improving reliability.

**Skill acquisition.** *Voyager* (Minecraft): an agent that writes, tests, and stores reusable code "skills" in a growing library, exhibiting open-ended self-improvement — an influential demonstration of lifelong agentic learning.

## Key Challenges
- **Compounding errors / brittleness.** Long-horizon tasks fail because per-step error rates compound; a 95%-reliable step over 20 steps ≈ 36% success. Reliability is the central blocker.
- **Planning & long-horizon coherence.** Models struggle to plan many steps ahead, recover from off-track states, and maintain goal coherence.
- **Memory management.** Deciding what to store/retrieve/forget; context bloat vs information loss.
- **Evaluation.** Agent benchmarks are noisy, environment-dependent, and quickly saturate or contaminate; reproducibility is hard.
- **Cost & latency.** Multi-step loops with large models are expensive and slow.
- **Safety.** Autonomous actions amplify risks (unsafe operations, prompt injection, runaway loops).

## Solutions & Current Best Practices
**Constrain the domain and use verifiable feedback** (code execution, environment rewards) — agents shine where actions are checkable. Prefer **explicit, controllable orchestration** (state machines/graphs, LangGraph) over fully autonomous "do anything" loops. Add **reflection/retry**, **structured memory + retrieval**, and **guardrails** (permissions, sandboxes, human-in-the-loop for risky actions). Increasingly, **train** agents end-to-end with RL on execution rewards (rather than relying on prompting + frameworks), which yields more robust behavior. Evaluate on realistic benchmarks (SWE-bench, τ-bench, WebArena, GAIA).

## Lab Perspectives
- **OpenAI:** Agents SDK, Operator/computer-use, "Deep Research" agent; o-series reasoning powers agentic coding.
- **Anthropic:** Claude as a strong agentic/coding model (Claude Code), MCP for tools, computer-use; emphasis on agent safety.
- **Google DeepMind:** Gemini agents, Project Mariner (web), AlphaCode-style search agents.
- **Open ecosystem:** LangChain/LangGraph, Microsoft AutoGen, CrewAI, HuggingFace smolagents provide the framework layer; many open agent models (Qwen-Agent).

## Latest Developments (2023–2026)
The field moved from **prompt-orchestrated** agents (LangChain-style) toward **RL-trained** agentic models where capability lives in the weights, with frameworks providing thinner orchestration. **Coding agents** (SWE-bench scores rising sharply) and **deep-research / web agents** became flagship products. **Computer-use** (GUI control via screenshots) emerged. Standardization via **MCP**. Strong focus on **reliability** (reducing compounding error) and **long-horizon** competence as the key frontier.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Describe the components of an LLM agent."** Planning, tool/action execution, observation, memory, in a loop with a termination condition.
- **"Why do long-horizon agents fail, and how do you improve reliability?"** Compounding per-step errors; verifiable feedback, reflection/retry, constrained scope, better planning, RL training.
- **"What is Reflexion?"** Verbal self-reflection on failures stored in memory to improve subsequent attempts without weight updates.
- **"Prompt-orchestration vs trained agents — tradeoffs?"** Frameworks are flexible but brittle; RL-trained agents are more robust but costly to build.

## Open Problems
**Reliable long-horizon autonomy** is the defining unsolved problem — robust planning, error recovery, and goal coherence over many steps. Effective open-ended **memory/continual learning**, trustworthy **agent evaluation**, and **safe autonomy** (preventing harmful actions under adversarial conditions) are all active and largely unsolved.

## References
- Yao, S. et al. (2023). *ReAct.* arXiv:2210.03629.
- Shinn, N. et al. (2023). *Reflexion.* arXiv:2303.11366.
- Wang, G. et al. (2023). *Voyager.* arXiv:2305.16291.
- Yao, S. et al. (2024). *τ-bench.* arXiv:2406.12045.
- Jimenez, C. et al. (2023). *SWE-bench.* arXiv:2310.06770.
