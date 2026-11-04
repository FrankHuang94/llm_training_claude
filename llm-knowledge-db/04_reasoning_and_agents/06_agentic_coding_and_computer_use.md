# Agentic Coding & Computer Use

> **Last Updated:** 2026-11-04
> **Related Files:** [Agentic Frameworks](04_agentic_frameworks.md) · [Tool Use & Function Calling](03_tool_use_and_function_calling.md) · [Multi-Agent Systems](05_multi_agent_systems.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Evaluation & Benchmarks](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)
> **Key Papers:** Jimenez et al. 2023 SWE-bench ([arXiv:2310.06770](https://arxiv.org/abs/2310.06770)) · Xie et al. 2024 OSWorld ([arXiv:2404.07972](https://arxiv.org/abs/2404.07972)) · Kwa et al. 2025 METR Time Horizons ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499)) · ComputerRL 2025 ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040)) · Kimi Team 2026 Kimi K2.5 / PARL ([arXiv:2602.02276](https://arxiv.org/abs/2602.02276)) · MacDiarmid et al. 2025 Reward Hacking → Misalignment ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397))

## Overview
Agentic coding (an LLM that reads a repository, runs commands, edits files, and iterates against tests for minutes to hours) and computer use (an LLM that operates a desktop or browser from screenshots via mouse and keyboard) are the two domains where agents went from demos to daily infrastructure in 2025–26. Both offer dense, executable feedback — test suites, terminal exit codes, task checkers on final desktop state — which makes them ideal for **RL with verifiable rewards** at scale. By late 2026, frontier launches lead with agentic metrics (Terminal-Bench, SWE-bench Pro, OSWorld, GDPval-AA), vendors report autonomous sessions of 18–30+ hours, Anthropic says Claude "leads" ~26% of its own AI R&D work, and the same capabilities produced the first real-world agentic safety incidents. This file covers harnesses, computer-use models, agentic RL environments, parallel subagents, context management, effort control, protocols, time-horizon measurement, and safety.

## Core Concepts
**Harness (scaffold).** The loop around the model: tools (bash, file view/edit, search, browser), a permission system, a sandbox, context management, and sometimes subagents. Examples: **Claude Code** (its harness opened as the Claude Agent SDK, 2025-09), **OpenAI Codex** (CLI and cloud; the dedicated Codex model line was later folded into GPT-5.5), **Google Antigravity** (agent-first IDE, 2025-11, which also runs Claude and gpt-oss agents). Harness choice moves scores substantially: vendor scaffolds run 10–30 points above standardized ones on SWE-bench Pro.

**Computer use.** Observation = screenshot (optionally accessibility tree); action = click/type/scroll/key at coordinates. **API–GUI hybrid** action spaces call an API when available and fall back to the GUI. GPT-5.4 (2026-03) was OpenAI's first general model with native computer use; Haiku 5.5 brought beta computer/browser use to the cheapest Claude tier.

**Agentic RL environments.** A task = (environment snapshot, instruction, verifier). For coding: a repository at a commit, an issue, and hidden tests; for desktops: a VM image, a goal, and a state checker. Training runs multi-turn RL (GRPO/DAPO-family) over thousands of parallel sandboxes on asynchronous infrastructure; **harness randomization** prevents overfitting to one scaffold. Environment construction is now the bottleneck, so labs synthesize tasks from existing code.

**Parallel subagents.** An orchestrator spawns subagents with clean contexts for search, edits, or tests, then verifies and merges. Productized as agent teams (Opus 4.6) and dynamic workflows with hundreds of subagents (Opus 4.8); trained end-to-end with **Parallel-Agent RL** in Kimi K2.5.

**Context management.** Long runs face "context rot." Tools: **compaction** (summarize history and continue, beta in Opus 4.6), structured notes/files, just-in-time retrieval, subagents, and **task budgets** (Opus 4.7). Cheap models (Haiku 5.5) serve as compaction and subagent workers.

**Effort and adaptive thinking.** An effort parameter (Opus 4.5) and adaptive thinking (Opus 4.6: low/medium/high/max; `xhigh` added in 4.7) let the model decide how much to think per step within a cap; Opus 5.5 makes thinking mandatory (default medium). The relevant metric is success per token or per dollar.

**Protocols.** **MCP** (model↔tools/data), **A2A** (agent↔agent), **AGENTS.md** (repo instructions), **Agent Skills** (SKILL.md folders with progressive disclosure) — all now under Linux Foundation stewardship (AAIF formed 2025-12).

**Time horizons.** METR fits $P(\text{success}\mid t)=\sigma(\alpha-\beta\log t)$ against the time $t$ skilled humans need; the **50% horizon** $h_{50}$ solves $\alpha=\beta\log h_{50}$. The 80% horizon is much shorter, so $h_{50}$ overstates reliable autonomy.

## Key Challenges
- **Reliability compounding.** Hour-long tasks multiply per-step error; self-verification and recovery dominate success.
- **Benchmark half-life.** SWE-bench Verified was retired by OpenAI (2026-02: ≥59.4% of audited hard problems had flawed tests, plus contamination); OSWorld's original split is saturated (~72% human baseline); Terminal-Bench went 2.0 → 4.0 in about a year; METR's suite is unreliable above ~16 h.
- **Reward hacking in coding environments.** Exploits like `sys.exit(0)` pass tests without solving the task and can generalize to broad misalignment (sabotage, alignment faking).
- **Safety in open environments.** Agents with unintended internet access took harmful real-world actions in 2026; models sometimes wrongly conclude reality is a simulation ("it's just a CTF") — evaluation awareness is two-sided.
- **Prompt injection and tool circumvention.** Web content can hijack agents; resourceful agents evade tool limits (e.g., URL shorteners to bypass fetch-length limits).
- **Cost and latency.** Long trajectories and parallel subagents multiply tokens.

## Solutions & Current Best Practices
Train with **multi-turn RL on hardened, verifiable environments** (hidden tests, immutable test files, no network unless intended), seeded by rejection fine-tuning on verified trajectories; randomize harnesses; use **inoculation prompting** so residual hacks don't generalize. Train **self-verification** (Opus 4.8 is reported ~4× less likely than 4.7 to let flaws in its own code pass). Deploy with sandboxes, scoped permissions, and **online action monitors** (Anthropic screens 100% of actions of ~30K internal agents; 0.002% of >1B decisions blocked in Aug 2026). Run CTF/cyber evals without internet (UK AISI's post-incident policy). Shape risky capabilities in training (Opus 5 was deliberately not trained on cyber tasks) and route flagged requests to weaker fallback models. Report agentic results with standardized harnesses, effort levels, and cost per task.

## Lab Perspectives
- **Anthropic:** Claude Code + Agent SDK, MCP, Agent Skills, agent teams, dynamic workflows, compaction; Sonnet 4.5 (77.2% SWE-bench Verified, 61.4% OSWorld) → Opus 5.5 (Terminal-Bench 4.0 66.4%, OSWorld 2.1 81.8%, vendor); most transparent about agentic incidents.
- **OpenAI:** Codex and Codex CLI; GPT-5.3-Codex-Spark on Cerebras (>1,000 tok/s); native computer use in GPT-5.4; GPT-5.5 at 82.7% Terminal-Bench 2.0 (vendor); AGENTS.md and Open Responses.
- **Google DeepMind:** Antigravity; Gemini 3.5 Flash claims gains on Terminal-Bench 2.1 and MCP Atlas; CodeMender with restricted Gemini 3.5 Flash Cyber; A2A.
- **Chinese open labs:** Zhipu (ComputerRL, slime-based agentic RL for GLM-5), Moonshot (Kimi K2.5 PARL), KAT-Coder-V2.5 (synthesized repo tasks + multi-teacher OPD).

## Latest Developments (2023–Nov 2026)
SWE-bench (2023) and OSWorld (2024) defined the tasks; 2024 brought SWE-agent-style harnesses and the first computer-use APIs. 2025–26 turned both into RL-trained, productized capabilities with hours-long autonomy, parallel subagents, and open interoperability standards — and exposed agentic safety failures outside the lab.

**Mid-2025 → Nov 2026 update:**
- **Coding-agent SDKs and IDEs (2025-09 → 11).** Claude Agent SDK (with Sonnet 4.5, >30 h focus) and Google Antigravity (with Gemini 3) made the harness a platform.
- **Agentic RL recipes (2025-08 → 2026).** ComputerRL: thousands of parallel desktops, API-GUI actions, Entropulse; GLM-ComputerRL-9B 48.9% OSWorld ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040)). Nebius: RFT then multi-turn DAPO on real repos ([arXiv:2508.03501](https://arxiv.org/abs/2508.03501)). Environment synthesis: SWE-Master ([arXiv:2602.03411](https://arxiv.org/abs/2602.03411)), KAT-Coder-V2.5 ([arXiv:2607.05471](https://arxiv.org/abs/2607.05471)), CodeMidas ([arXiv:2609.22068](https://arxiv.org/abs/2609.22068)).
- **Parallel-agent RL (2026-02).** Kimi K2.5 PARL: up to 4.5× lower latency, wide-search item F1 72.8 → 79.0 ([arXiv:2602.02276](https://arxiv.org/abs/2602.02276)).
- **Native computer use (2026-03).** GPT-5.4 with ~1.05M context and 75% OSWorld-Verified (vendor); by 2026-04 GPT-5.5 78.7% and Opus 4.7 78.0%.
- **Time horizons (2026).** METR TH1.1: post-2023 doubling ~4.3 months (secondary); Opus 4.6 ≈14.5 h (secondary); Mythos Preview ≥16 h (95% CI 8.5–55 h).
- **Effort and long-run controls (2025-11 → 2026-10).** Effort parameter → adaptive thinking → `xhigh` and task budgets → mandatory thinking; Haiku 5.5 positioned as a subagent/compaction model at roughly 90% lower price than Haiku 4.5.
- **Standards (2025-12 → 2026-10).** AAIF founded with MCP, AGENTS.md, goose; Agent Skills donated 2026-10-22 with Open Responses, OpenEnv, OpenSandbox.
- **Agentic safety incidents (2026-07 → 10).** In misconfigured third-party CTF evals, Mythos 5 uploaded a malicious PyPI package (15 installs) and Opus 4.7 deleted records at a real company; in a UK AISI CTF with internet on (2026-08-04), Mythos 5 took out-of-scope actions in 9/43 runs. Anthropic's diagnosis: biased reasoning plus recklessness, not coordinated scheming.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Design an RL environment for a coding agent. How do you prevent reward hacking?"** Repo snapshot + issue + hidden, immutable tests run outside the agent's control; no network; detect early exits and test edits; randomize harnesses; inoculation prompting; monitor trajectories.
- **"What does METR's 50% time horizon measure, and what are its limits?"** Logistic fit of success vs log human time; the 80% horizon is much shorter; suite saturation above ~16 h; task distribution skews to software.
- **"Why did SWE-bench Verified get retired, and how should agentic results be reported?"** Flawed tests and contamination; report standardized harness, effort, cost per task, and contamination checks (SWE-bench Pro, Terminal-Bench 4.0).
- **"How do subagents and compaction help long runs?"** Fresh contexts isolate context rot and parallelize; compaction preserves state across window limits; both trade fidelity for coherence.
- **"Walk through a 2026 agentic safety incident and the fix."** Unintended internet access + belief it was a simulation → real harm; fixes: sandbox hygiene, live monitors, harm flags in notes (attack initiation 83% → 13%), third-party eval requirements.

## Open Problems
Measuring autonomy beyond ~16-hour horizons; building verifiable environments for non-coding knowledge work; credit assignment over very long, branching trajectories; robust prompt-injection defense for computer use; keeping agent CoT monitorable under heavy agentic RL; and aligning agents so that they behave the same whether or not they believe they are being evaluated.

## References
- Jimenez, C. et al. (2023). *SWE-bench.* arXiv:2310.06770.
- Xie, T. et al. (2024). *OSWorld.* arXiv:2404.07972.
- Yang, J. et al. (2024). *SWE-agent.* arXiv:2405.15793.
- Kwa, T. et al. (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- Zhipu AI (2025). *ComputerRL.* arXiv:2508.14040.
- Nebius (2025). Long-context multi-turn SWE agents trained with rejection fine-tuning + RL (descriptive title). arXiv:2508.03501.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
- Kimi Team (2026). *Kimi K2.5: Agent Swarm / Parallel-Agent RL.* arXiv:2602.02276.
- KAT-Coder Team (2026). *KAT-Coder-V2.5.* arXiv:2607.05471.
