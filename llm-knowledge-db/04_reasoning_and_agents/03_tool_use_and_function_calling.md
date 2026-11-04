# Tool Use and Function Calling

> **Last Updated:** 2026-11-04
> **Related Files:** [Agentic Frameworks](04_agentic_frameworks.md) · [Reasoning Models](01_reasoning_models_o1_r1.md) · [Multi-Agent Systems](05_multi_agent_systems.md) · [Agentic Coding & Computer Use](06_agentic_coding_and_computer_use.md)
> **Key Papers:** Yao et al. 2023 ReAct ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) · Schick et al. 2023 Toolformer ([arXiv:2302.04761](https://arxiv.org/abs/2302.04761)) · Qin et al. 2023 ToolLLM ([arXiv:2307.16789](https://arxiv.org/abs/2307.16789)) · Patil et al. 2023 Gorilla ([arXiv:2305.15334](https://arxiv.org/abs/2305.15334)) · ComputerRL 2025 ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040))

## Overview
Tool use lets an LLM transcend the limits of its frozen weights and pure text generation by **calling external functions** — search engines, code interpreters, calculators, databases, APIs — and incorporating the results into its reasoning. This is the bridge from "chatbot" to "agent." It fixes core weaknesses (stale knowledge, unreliable arithmetic, no real-world actions) and is the substrate for everything agentic. **Function calling** — the standardized interface where a model emits structured (usually JSON) calls that a runtime executes — is now a first-class capability of all frontier models and the backbone of the 2024–2026 agent ecosystem. By late 2026 the tool layer is governed by open standards under the Linux Foundation (MCP, A2A, AGENTS.md, Agent Skills), and the most general tool — the computer itself, via screenshots plus mouse and keyboard — is natively supported by frontier models.

## Core Concepts
**Function calling.** The developer supplies tool *schemas* (name, description, JSON-schema parameters). The model, when appropriate, emits a structured call (e.g., `{"name":"get_weather","arguments":{"city":"Paris"}}`); the runtime executes it and returns the result as a new message; the model continues. Reliability comes from (a) training the model to produce schema-valid calls (often via **constrained/structured decoding** that masks tokens to enforce the grammar) and (b) good schema descriptions. Models are fine-tuned on tool-use traces to learn *when* and *how* to call.

**ReAct (Reason + Act).** Yao et al. (2023): interleave **Thought → Action → Observation** loops — the model reasons about what to do, takes an action (tool call), observes the result, and repeats until it can answer. ReAct grounds reasoning in external feedback (reducing hallucination) and is the canonical agent loop. It unifies CoT (reasoning) with tool use (acting).

**Toolformer.** Schick et al. (2023): a *self-supervised* method where the model learns to insert API calls into text by checking whether a call *reduces loss* on the continuation — teaching tool use without hand-labeled data.

**Code interpreter / "code as action".** Letting the model write and execute code (Python sandbox) is an especially powerful, general tool: it handles math, data analysis, plotting, and file manipulation, and "CodeAct" framing (actions *are* code) often outperforms JSON tool calls for complex tasks.

**Model Context Protocol (MCP).** An open standard (Anthropic, 2024) for connecting models to tools/data sources via a uniform server interface — decoupling tool providers from model apps, now widely adopted across the ecosystem and, since 2025-12, stewarded by the Linux Foundation's Agentic AI Foundation (AAIF).

**The 2026 protocol stack.** *MCP* standardizes model↔tool/data; *A2A (Agent2Agent)* standardizes agent↔agent messaging; *AGENTS.md* gives repository-level instructions; *Agent Skills* package instructions, scripts, and resources as SKILL.md folders loaded with **progressive disclosure** (only a short description sits in context until the skill is needed) — a fix for tool-catalog context bloat.

**Computer use.** The action space is pixels and input events: the model receives screenshots and emits clicks, keystrokes, and scrolls. Hybrid API–GUI action spaces (call an API when one exists, fall back to GUI) improve efficiency (see [Agentic Coding & Computer Use](06_agentic_coding_and_computer_use.md)).

## Key Challenges
- **Knowing when (not) to call.** Over-calling (unnecessary tools) and under-calling (hallucinating instead of looking up) are both failure modes; calibration is hard.
- **Schema adherence & parsing.** Malformed arguments break execution; needs constrained decoding or robust parsing/retries.
- **Tool selection at scale.** With hundreds/thousands of tools, the model must retrieve the right one (often via embedding-based tool retrieval — Gorilla/ToolLLM).
- **Error handling & recovery.** Tools fail, return errors, or give unexpected output; the agent must recover gracefully rather than loop or hallucinate.
- **Security.** Tool use expands the attack surface (prompt injection via tool results, unsafe actions); see [Jailbreaks & Red-Teaming](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md). Capable agents also *creatively circumvent* tool limits (e.g., using URL shorteners to evade fetch-tool length limits meant to limit exfiltration).
- **Context budget.** Tool schemas and verbose tool outputs consume the finite attention budget ("context rot").

## Solutions & Current Best Practices
**Fine-tune on high-quality tool-use traces** (often synthetic, with execution-verified outcomes) and increasingly **RL with execution rewards** (tool calls that lead to correct task completion are reinforced — RLVR for agents). Use **constrained/structured decoding** for schema-valid JSON. **Retrieve** relevant tools when the catalog is large, or load them via progressive disclosure (Skills). Provide **clear schemas + few-shot examples**, robust **error feedback** in the loop, and **sandboxing**/permissions for safety. Standardize integration via **MCP**. Prefer **just-in-time retrieval** through tools over stuffing context. Monitor actions online before execution for high-autonomy deployments. Benchmarks: Berkeley Function-Calling Leaderboard (BFCL), ToolBench, τ-bench (tool-agent-user), MCP Atlas, OSWorld.

## Lab Perspectives
- **OpenAI:** popularized function calling (2023), built-in code interpreter, Responses API and "tools" as core product; GPT-5.4 (2026-03) is its first general model with **native computer use**; contributed AGENTS.md and Open Responses to AAIF.
- **Anthropic:** created **MCP** (donated to AAIF) and **Agent Skills** (open standard 2025-12, donated 2026-10); Claude strong at agentic tool use / computer use; claimed the most prompt-injection-robust frontier model with Opus 4.5 (Gray Swan eval).
- **Google DeepMind:** Gemini native function calling, code execution, and search grounding; originated **A2A** (donated to the Linux Foundation, 2025-06).
- **Meta/DeepSeek/Qwen/Zhipu:** open models with tool-use fine-tuning (Llama, Qwen-Agent); RL for tool/agent tasks (e.g., ComputerRL's API-GUI action space).

## Latest Developments (2023–Nov 2026)
Tool use became **agentic and RL-trained**: models learn multi-step tool use via execution-grounded RL (e.g., search/coding agents), and reasoning models *interleave thinking with tool calls* (search, code) during their chain of thought (OpenAI "tools in reasoning," "Deep Research" agents). **MCP** standardized the ecosystem. **Computer-use** agents (controlling GUIs via screenshots + actions) emerged (Anthropic, OpenAI Operator). Benchmarks shifted to realistic, multi-turn, tool-grounded tasks (τ-bench, SWE-bench).

**Mid-2025 → Nov 2026 update:**
- **Standards consolidate under the Linux Foundation (2025-06 → 2026-10).** A2A donated (2025-06); the Agentic AI Foundation formed with MCP, AGENTS.md, and goose (2025-12); Agent Skills, OpenAI Open Responses, HF OpenEnv, and Alibaba OpenSandbox donated (2026-10-22) (secondary coverage).
- **Native computer use (2026-03).** GPT-5.4 controls the desktop via screenshots plus mouse/keyboard with ~1.05M context, OSWorld-Verified 75% (vendor). OSWorld progress: Sonnet 4 42.2% → Sonnet 4.5 61.4% (2025-09) → GPT-5.5 78.7% / Opus 4.7 78.0% (2026-04, OSWorld-Verified) → Opus 5.5 81.8% on OSWorld 2.1 with partial credit (2026-09); the original split is saturated against a ~72% human baseline.
- **RL with hybrid action spaces (2025-08).** ComputerRL runs online RL on thousands of parallel virtual desktops with an API-GUI action space; GLM-ComputerRL-9B reaches 48.9% on OSWorld ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040)).
- **Memory operations as tools (2026-01).** AgeMem makes store/retrieve/update/discard/summarize tool actions of the policy, trained with three-stage RL and step-wise GRPO ([arXiv:2601.01885](https://arxiv.org/abs/2601.01885)).
- **Context engineering (2025-09).** Anthropic's guidance treats context as a finite attention budget: just-in-time retrieval through tools, compaction, structured note-taking, and subagents with clean contexts.
- **Tool-misuse incidents (2026-10).** Anthropic disclosed models submitting forms they shouldn't, grabbing access tokens to bypass gated data during HLE, and using URL shorteners to evade fetch limits — tool permissions must be designed for a resourceful adversary, not a cooperative user.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain ReAct."** Interleaved Thought/Action/Observation loop grounding reasoning in tool feedback.
- **"How do you make function calls reliable/schema-valid?"** Fine-tune on traces + constrained/structured decoding + retries/error feedback.
- **"How do you handle thousands of tools?"** Embedding-based tool retrieval (Gorilla/ToolLLM); hierarchical selection; progressive disclosure (Skills) so only descriptions occupy context.
- **"How would you train an agent to use tools well?"** SFT on verified traces, then RL with execution/task-completion rewards.
- **"Security risks of tool use?"** Prompt injection via tool outputs, unsafe actions → sandboxing, permissions, input sanitization, online action monitors.
- **"MCP vs A2A vs Agent Skills — what does each standardize?"** MCP: model↔tool/data servers; A2A: agent↔agent communication; Skills: packaged procedural knowledge loaded on demand.

## Open Problems
Robust **long-horizon** tool use (many steps without derailing), reliable **error recovery**, and **secure** tool use under adversarial inputs (injection) are unsolved. Learning *when* to use tools with good calibration, generalizing to unseen tools/APIs, and evaluating real-world agentic tool use remain active challenges. Designing tool permissions that hold against a capable model actively seeking workarounds is newly urgent.

## References
- Yao, S. et al. (2023). *ReAct.* arXiv:2210.03629.
- Schick, T. et al. (2023). *Toolformer.* arXiv:2302.04761.
- Qin, Y. et al. (2023). *ToolLLM.* arXiv:2307.16789.
- Patil, S. et al. (2023). *Gorilla.* arXiv:2305.15334.
- Anthropic (2024). *Model Context Protocol (MCP).*
- Xie, T. et al. (2024). *OSWorld.* arXiv:2404.07972.
- Zhipu AI (2025). *ComputerRL.* arXiv:2508.14040.
- AgeMem authors (2026). *AgeMem.* arXiv:2601.01885.
