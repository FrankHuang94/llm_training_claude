# Inference-Time Scaling

> **Last Updated:** 2026-11-04
> **Related Files:** [Reasoning Models](01_reasoning_models_o1_r1.md) · [Chain-of-Thought](00_chain_of_thought.md) · [Rejection Sampling & Best-of-N](../03_posttraining/08_rejection_sampling_and_best_of_N.md) · [Agentic Coding & Computer Use](06_agentic_coding_and_computer_use.md) · [Test-Time Training](../09_emerging_frontiers/03_test_time_training.md)
> **Key Papers:** Snell et al. 2024 Test-Time Compute ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314)) · Wang et al. 2022 Self-Consistency ([arXiv:2203.11171](https://arxiv.org/abs/2203.11171)) · Brown et al. 2024 Large Language Monkeys ([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)) · Wu et al. 2024 Inference Scaling Laws ([arXiv:2408.00724](https://arxiv.org/abs/2408.00724))

## Overview
For most of the LLM era, the only knob for better performance was training: more parameters, more data, more pretraining compute. Inference-time (test-time) scaling adds an orthogonal axis: **spend more compute *per query* to get better answers**, by generating and searching over many candidate reasoning paths. The striking empirical finding (o1, Snell et al. 2024) is that test-time compute can be traded against training compute — a smaller model thinking longer can beat a larger model answering instantly on hard problems. This reframes the economics of intelligence (cheap pretraining + expensive inference for hard problems) and is the theoretical underpinning of reasoning models. By 2026 test-time compute is a **product control** (effort levels, adaptive thinking), is spent increasingly on *parallel agents* rather than single chains, and is reported as tokens- or cost-per-task next to accuracy.

## Core Concepts
**The core tradeoff.** Define performance as a function of *training* FLOPs and *inference* FLOPs. Snell et al. and Wu et al. show **inference-scaling laws**: accuracy improves predictably (often log-linearly) with test-time samples/search, and there is a *compute-optimal* allocation between making the model bigger vs letting it think longer — which depends on problem difficulty. Easy problems saturate quickly; hard problems benefit most from more inference compute.

**Methods (parallel and sequential).**
- **Best-of-N / repeated sampling:** sample $N$ answers, select one. With a *verifier* (oracle), **coverage** (pass@N — probability *any* sample is correct) rises smoothly with $N$ (Brown et al.'s "Large Language Monkeys": pass@N follows a log-linear law across many orders of magnitude). Without an oracle, you need a selector.
- **Self-consistency (majority vote):** sample $N$ chains, take the most common answer — no verifier needed, but capped by the model's modal correctness.
- **Verifier-guided selection:** score samples with a reward/process model and pick the best (best-of-N weighted, beam search over steps).
- **Tree search / MCTS:** explore reasoning steps with lookahead, value estimates, and backtracking — sequential revision rather than parallel sampling.
- **Sequential revision ("think longer"):** the model iteratively critiques and improves its own answer (self-refinement) or simply produces a longer internal CoT (o1/R1). Snell et al. find sequential and parallel scaling are complementary, with the optimal mix depending on difficulty.
- **Parallel agents:** an orchestrator spawns subagents with separate contexts that work concurrently, then verifies and merges results — test-time compute that also cuts wall-clock latency.

**Coverage vs selection.** Repeated sampling raises the *chance a good answer exists* (coverage); the bottleneck becomes **selection** — picking the right one. With perfect verifiers (math/code), selection is solved and scaling is dramatic; without them, a fallible RM/judge limits gains (the "generation-verification gap").

**Effort as an API knob.** Vendors expose discrete reasoning budgets (e.g., Claude's low/medium/high/xhigh/max; OpenAI's reasoning effort) and let the model adapt within them; the relevant curve is accuracy vs *tokens per task*, not accuracy at a fixed setting.

## Key Challenges
- **Selection without verifiers.** For open-ended tasks, you can generate a correct answer but can't reliably identify it; the reward model becomes the ceiling.
- **Cost & latency.** Test-time compute multiplies inference cost; serving economics and user-facing latency constrain it.
- **Diminishing returns / inverse scaling.** Beyond a point, more samples help little; on some tasks, more "thinking" hurts (overthinking, distraction).
- **Difficulty estimation.** Compute-optimal allocation requires knowing how hard a query is — itself a hard prediction problem.
- **Harness dependence.** Agentic and interactive scores swing widely with scaffold, budget, and adapters, making cross-lab comparisons fragile.

## Solutions & Current Best Practices
**Use verifiers wherever possible** (math/code/exec) to make selection cheap and scaling strong. For general tasks, **self-consistency** + a robust judge. **Adaptive compute**: estimate difficulty and allocate thinking accordingly (don't overthink easy queries) — now trained into models as adaptive thinking under an effort cap. Reasoning models (o1/R1) **internalize** the search into a trained long CoT, which Snell et al. suggest is often more compute-efficient than external best-of-N. Combine internal reasoning with light external search or parallel attempts for the hardest problems. Train **self-verification** so the model itself is a better selector. Cache/share prefixes to amortize cost; use fast-inference hardware and fast modes where latency matters.

## Lab Perspectives
- **OpenAI:** o-series productized inference-time scaling; exposes a "reasoning effort" control; GPT-5 routes between fast and thinking models; GPT-5.3-Codex-Spark runs on Cerebras WSE-3 at >1,000 tok/s for real-time coding.
- **DeepSeek:** R1 shows internalized long-CoT scaling; DeepSeekMath-V2 scales *verification* compute; V4 offers Thinking / Non-Thinking modes.
- **Google DeepMind:** Snell et al. (DeepMind) formalized compute-optimal test-time scaling; AlphaProof/AlphaCode use heavy search at inference; Deep Think is the high-compute mode; Gemini 3.6 Flash cut output tokens ~17% vs 3.5 Flash.
- **Anthropic:** extended-thinking budget → effort parameter → adaptive thinking; reports results with and without parallel test-time compute; agent teams and hundreds of parallel subagents in Claude Code.
- **Moonshot:** RL-trained parallel-agent orchestration (Kimi K2.5) to trade compute for latency.

## Latest Developments (2023–Nov 2026)
Inference-scaling laws (Snell, Wu, Brown) gave the paradigm rigorous footing. The frontier moved from **external** search (best-of-N/MCTS) to **trained internal** reasoning (o1/R1) as the more efficient default, with external search reserved for the hardest/agentic tasks. Active work on **closing the generation-verification gap** (better selectors/verifiers for non-verifiable domains), adaptive/"router" approaches that decide how much to think, and the economics of inference-heavy products.

**Mid-2025 → Nov 2026 update:**
- **Effort levels and adaptive thinking (2025-11 → 2026-10).** Claude Opus 4.5 introduced the effort parameter (medium effort matched Sonnet 4.5's best SWE-bench Verified with 76% fewer output tokens); Opus 4.6 (2026-02) made thinking adaptive with four levels, Opus 4.7 added `xhigh` and task budgets, and Haiku 5.5 brought effort to the small tier.
- **Parallel test-time compute for agents (2025-09 → 2026-05).** Sonnet 4.5: 77.2% → 82.0% SWE-bench Verified with parallel compute (vendor); Opus 4.8's "dynamic workflows" plan large tasks, run hundreds of parallel subagents, then verify.
- **Parallel-Agent RL (2026-02).** Kimi K2.5 trains an orchestrator to decompose and spawn subagents concurrently instead of calling tools sequentially: up to 4.5× lower latency and item-F1 72.8 → 79.0 on wide search ([arXiv:2602.02276](https://arxiv.org/abs/2602.02276)).
- **Verification compute (2025-11).** DeepSeekMath-V2 scales verifier compute to label and select hard proofs ([arXiv:2511.22570](https://arxiv.org/abs/2511.22570)) — selection, not generation, is where the extra compute goes.
- **Test-time *training* as another axis (2025-12).** TTT-E2E takes next-token gradient steps on the context at test time with a meta-learned initialization, matching full attention's context scaling at 3B with constant per-token latency and 2.7× speedup at 128K ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)).
- **Latency as a product axis (2026).** GPT-5.3-Codex-Spark on Cerebras (>1,000 tok/s), Opus 4.8's 2.5× fast mode, and GPT-5.5 using fewer tokens per Codex task show vendors optimizing compute-per-task, not just accuracy.
- **Harness sensitivity (2026).** A reported GPT-6 Astra ARC-AGI-3 score of 62.7% standard vs 99.9% with a "provider adapter" (single source, unverified) illustrates how much test-time scaffolding can move results.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What is inference-time scaling and why does it matter?"** Spend more compute per query (sampling/search/longer CoT) for higher accuracy; trades against training compute.
- **"Best-of-N vs self-consistency vs MCTS — when each?"** Verifier available → best-of-N/search; none → self-consistency; structured search for exploration-heavy tasks.
- **"What's the coverage-vs-selection distinction?"** Sampling raises chance a good answer exists; selecting it is the bottleneck without verifiers.
- **"Why can a small model + more inference beat a big model?"** Compute-optimal allocation: test-time search substitutes for parameters on hard problems (Snell et al.).
- **"Sequential thinking vs parallel subagents — when does each win?"** Sequential for tightly coupled reasoning; parallel for decomposable, wide tasks (search, multi-file edits) where it also cuts wall-clock latency — but needs a verifier/merge step and costs more total tokens.

## Open Problems
Closing the **generation-verification gap** for open-ended domains (reliable selection without ground-truth verifiers) is the central unsolved problem. Optimal *adaptive* compute allocation, the ultimate limits of test-time scaling (does it plateau?), and the right economic/latency tradeoffs for inference-heavy systems remain open and consequential. How to compare models fairly when effort level, harness, and parallelism all change the score is unresolved.

## References
- Snell, C. et al. (2024). *Scaling LLM Test-Time Compute Optimally.* arXiv:2408.03314.
- Wang, X. et al. (2022). *Self-Consistency.* arXiv:2203.11171.
- Brown, B. et al. (2024). *Large Language Monkeys.* arXiv:2407.21787.
- Wu, Y. et al. (2024). *Inference Scaling Laws.* arXiv:2408.00724.
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050.
- Kimi Team (2026). *Kimi K2.5: Agent Swarm / Parallel-Agent RL.* arXiv:2602.02276.
- TTT-E2E authors (2025). *End-to-End Test-Time Training for Long Context.* arXiv:2512.23675.
