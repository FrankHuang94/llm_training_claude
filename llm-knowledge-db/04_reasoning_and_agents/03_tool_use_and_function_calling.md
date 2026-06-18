# Tool Use and Function Calling

> **Last Updated:** 2026-06-18
> **Related Files:** [Agentic Frameworks](04_agentic_frameworks.md) · [Reasoning Models](01_reasoning_models_o1_r1.md) · [Multi-Agent Systems](05_multi_agent_systems.md)
> **Key Papers:** Yao et al. 2023 ReAct ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) · Schick et al. 2023 Toolformer ([arXiv:2302.04761](https://arxiv.org/abs/2302.04761)) · Qin et al. 2023 ToolLLM ([arXiv:2307.16789](https://arxiv.org/abs/2307.16789)) · Patil et al. 2023 Gorilla ([arXiv:2305.15334](https://arxiv.org/abs/2305.15334))

## Overview
Tool use lets an LLM transcend the limits of its frozen weights and pure text generation by **calling external functions** — search engines, code interpreters, calculators, databases, APIs — and incorporating the results into its reasoning. This is the bridge from "chatbot" to "agent." It fixes core weaknesses (stale knowledge, unreliable arithmetic, no real-world actions) and is the substrate for everything agentic. **Function calling** — the standardized interface where a model emits structured (usually JSON) calls that a runtime executes — is now a first-class capability of all frontier models and the backbone of the 2024–2026 agent ecosystem.

## Core Concepts
**Function calling.** The developer supplies tool *schemas* (name, description, JSON-schema parameters). The model, when appropriate, emits a structured call (e.g., `{"name":"get_weather","arguments":{"city":"Paris"}}`); the runtime executes it and returns the result as a new message; the model continues. Reliability comes from (a) training the model to produce schema-valid calls (often via **constrained/structured decoding** that masks tokens to enforce the grammar) and (b) good schema descriptions. Models are fine-tuned on tool-use traces to learn *when* and *how* to call.

**ReAct (Reason + Act).** Yao et al. (2023): interleave **Thought → Action → Observation** loops — the model reasons about what to do, takes an action (tool call), observes the result, and repeats until it can answer. ReAct grounds reasoning in external feedback (reducing hallucination) and is the canonical agent loop. It unifies CoT (reasoning) with tool use (acting).

**Toolformer.** Schick et al. (2023): a *self-supervised* method where the model learns to insert API calls into text by checking whether a call *reduces loss* on the continuation — teaching tool use without hand-labeled data.

**Code interpreter / "code as action".** Letting the model write and execute code (Python sandbox) is an especially powerful, general tool: it handles math, data analysis, plotting, and file manipulation, and "CodeAct" framing (actions *are* code) often outperforms JSON tool calls for complex tasks.

**Model Context Protocol (MCP).** An open standard (Anthropic, 2024) for connecting models to tools/data sources via a uniform server interface — decoupling tool providers from model apps, now widely adopted across the ecosystem.

## Key Challenges
- **Knowing when (not) to call.** Over-calling (unnecessary tools) and under-calling (hallucinating instead of looking up) are both failure modes; calibration is hard.
- **Schema adherence & parsing.** Malformed arguments break execution; needs constrained decoding or robust parsing/retries.
- **Tool selection at scale.** With hundreds/thousands of tools, the model must retrieve the right one (often via embedding-based tool retrieval — Gorilla/ToolLLM).
- **Error handling & recovery.** Tools fail, return errors, or give unexpected output; the agent must recover gracefully rather than loop or hallucinate.
- **Security.** Tool use expands the attack surface (prompt injection via tool results, unsafe actions); see [Jailbreaks & Red-Teaming](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md).

## Solutions & Current Best Practices
**Fine-tune on high-quality tool-use traces** (often synthetic, with execution-verified outcomes) and increasingly **RL with execution rewards** (tool calls that lead to correct task completion are reinforced — RLVR for agents). Use **constrained/structured decoding** for schema-valid JSON. **Retrieve** relevant tools when the catalog is large. Provide **clear schemas + few-shot examples**, robust **error feedback** in the loop, and **sandboxing**/permissions for safety. Standardize integration via **MCP**. Benchmarks: Berkeley Function-Calling Leaderboard (BFCL), ToolBench, τ-bench (tool-agent-user).

## Lab Perspectives
- **OpenAI:** popularized function calling (2023), built-in code interpreter, Responses API and "tools" as core product; o-series integrates tool use into reasoning.
- **Anthropic:** created **MCP** (open tool standard); Claude strong at agentic tool use / computer use; emphasizes safe tool-use.
- **Google DeepMind:** Gemini native function calling, code execution, and search grounding.
- **Meta/DeepSeek/Qwen:** open models with tool-use fine-tuning (Llama, Qwen-Agent); RL for tool/agent tasks.

## Latest Developments (2023–2026)
Tool use became **agentic and RL-trained**: models learn multi-step tool use via execution-grounded RL (e.g., search/coding agents), and reasoning models *interleave thinking with tool calls* (search, code) during their chain of thought (OpenAI "tools in reasoning," "Deep Research" agents). **MCP** standardized the ecosystem. **Computer-use** agents (controlling GUIs via screenshots + actions) emerged (Anthropic, OpenAI Operator). Benchmarks shifted to realistic, multi-turn, tool-grounded tasks (τ-bench, SWE-bench).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Explain ReAct."** Interleaved Thought/Action/Observation loop grounding reasoning in tool feedback.
- **"How do you make function calls reliable/schema-valid?"** Fine-tune on traces + constrained/structured decoding + retries/error feedback.
- **"How do you handle thousands of tools?"** Embedding-based tool retrieval (Gorilla/ToolLLM); hierarchical selection.
- **"How would you train an agent to use tools well?"** SFT on verified traces, then RL with execution/task-completion rewards.
- **"Security risks of tool use?"** Prompt injection via tool outputs, unsafe actions → sandboxing, permissions, input sanitization.

## Open Problems
Robust **long-horizon** tool use (many steps without derailing), reliable **error recovery**, and **secure** tool use under adversarial inputs (injection) are unsolved. Learning *when* to use tools with good calibration, generalizing to unseen tools/APIs, and evaluating real-world agentic tool use remain active challenges.

## References
- Yao, S. et al. (2023). *ReAct.* arXiv:2210.03629.
- Schick, T. et al. (2023). *Toolformer.* arXiv:2302.04761.
- Qin, Y. et al. (2023). *ToolLLM.* arXiv:2307.16789.
- Patil, S. et al. (2023). *Gorilla.* arXiv:2305.15334.
- Anthropic (2024). *Model Context Protocol (MCP).*
