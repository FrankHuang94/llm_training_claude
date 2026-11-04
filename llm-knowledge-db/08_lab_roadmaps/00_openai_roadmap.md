# OpenAI — Lab Roadmap

> **Last Updated:** 2026-11-04
> **Related Files:** [Reasoning Models](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [RLHF](../03_posttraining/03_RLHF.md) · [Lab Comparison Matrix](06_lab_comparison_matrix.md)
> **Key Papers:** Brown et al. 2020 GPT-3 ([arXiv:2005.14165](https://arxiv.org/abs/2005.14165)) · Ouyang et al. 2022 InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) · OpenAI 2023 GPT-4 ([arXiv:2303.08774](https://arxiv.org/abs/2303.08774)) · OpenAI 2024 o1 "Learning to Reason" · OpenAI 2025 gpt-oss model card

## Overview
OpenAI is the company that brought LLMs to mass adoption (ChatGPT, Nov 2022) and has repeatedly defined the frontier: the GPT scaling line, RLHF/InstructGPT, GPT-4's multimodal leap, the **o-series reasoning paradigm** (inference-time scaling), and — since Aug 2025 — the unified **GPT-5 → GPT-6** system line. Its strategy combines aggressive scaling of *both* pretraining and RL compute, a closed/product-first model (with a notable exception: the open-weight **gpt-oss**), and a stated mission to build AGI "that benefits all of humanity." It has moved from a Microsoft-centric compute model to **gigawatt-scale, multi-vendor contracts** (NVIDIA, AMD, Broadcom, Cerebras) and a capped-profit-under-nonprofit structure that has been undergoing restructuring since 2025; an IPO is widely reported as targeted for late 2026.

## Model Lineage & Key Innovations
- **GPT-1/2/3 (2018–2020):** established decoder-only generative pretraining and **scaling laws**; GPT-3 (175B) demonstrated **in-context/few-shot learning**.
- **InstructGPT → ChatGPT (2022):** **RLHF** alignment — the usability breakthrough; a 1.3B aligned model beat 175B base on preference.
- **GPT-4 (2023):** large multimodal (vision) model, widely believed **MoE**; GPT-4-Turbo/4o added speed, 128K context, and **native any-to-any** multimodality (text/image/audio) in **GPT-4o**.
- **o1 → o3 / o4-mini (2024–2025):** **reasoning models** trained with RL to think before answering, scaling **test-time compute**. GPT-4.1 (Apr 2025) brought 1M-token context to the API.
- **gpt-oss-120b / 20b (2025-08):** first open-weight LLMs since GPT-2 (Apache 2.0). 116.8B total / 5.1B active (128 experts, top-4), alternating banded (sliding) and dense GQA layers with **learned attention-sink biases**, MoE weights post-trained to **MXFP4** (~4.25 bits/param) so the 120b fits on one 80 GB GPU; ~2.1M H100-hours; near o4-mini on reasoning.
- **GPT-5 (2025-08-07):** a *unified system* — fast model + thinking model + real-time **router** — absorbing the o-series. The router misbehaved at launch and legacy models were restored after user backlash.
- **GPT-5.1 → 5.2 (2025-11/12):** 5.1 Instant/Thinking; 5.2 (Instant/Thinking/Pro) targeted long-context reasoning and fewer hallucinations, shipped after an internal "code red" following Gemini 3.
- **GPT-5.3-Codex-Spark (2026-02):** small real-time coder on **Cerebras WSE-3** (>1,000 tok/s) — OpenAI's first production deployment on non-NVIDIA silicon.
- **GPT-5.4 (2026-03):** first general model with **native computer use** (screenshots + mouse/keyboard), ~1.05M context, OSWorld-Verified 75% (vendor).
- **GPT-5.5 (2026-04):** same per-token latency, fewer tokens per agentic task; Terminal-Bench 2.0 82.7% (5.4: 75.1%); the separate Codex line was reportedly folded in.
- **GPT-5.6 Sol / Terra / Luna (2026-06/07):** first three-tier family (flagship/balanced/cheap); limited preview reportedly under government safety review before GA on 07-09. **GPT-5.6-Cyber** (~2026-08, "High" cyber) shipped via the **Daybreak** trusted-access program.
- **GPT-6 Astra (2026-09-03):** new flagship; first OpenAI model rated **"Critical" in cybersecurity** under the Preparedness Framework (safeguards required even for internal deployment). 1.05M context, 128K output; gains attributed to large-scale pretraining plus long-horizon RL; reportedly OpenAI's largest run (>100k GPUs at the Texas Stargate site; single source).
- **GPT-6 Sol / Luna (2026-09-22)** at ~GPT-5.6 capability for ~half the price (rated "High", not Critical); **GPT-6.1 Sol** reportedly launched at DevDay (2026-09-29), near-Astra on coding/computer use at ~1/5 the price.
- **Ecosystem:** **DALL·E / GPT-image** (images), **Sora** (video / implicit world model), **Whisper** (open ASR), **Operator → Deep Research → Codex** (agents; Codex CLI defaults to GPT-6.1 Sol per a single report).

## Technical Differentiation
- **RLHF pioneers** — invented the modern post-training paradigm.
- **Inference-time scaling leadership** — the o-series productized "think longer for harder problems"; GPT-5 turned it into a *routed* product decision, and token-efficiency per task became a headline metric (5.5).
- **Re-scaled pretraining + long-horizon RL** — GPT-6 Astra is explicitly framed as a return to very large pretraining runs combined with agentic RL.
- **Agentic computer use in the base model** (GPT-5.4+) rather than a separate Operator product.
- **Tiered trusted access** — Daybreak Blue (reduced refusals on mainline models) / Red (cyber-specialized models, stronger identity verification).
- **Product + platform** — ChatGPT + API + Codex + enterprise; fastest to mass distribution.

## Published Research Focus
RLHF/InstructGPT, scaling laws (Kaplan), reward-model overoptimization (Gao), process supervision ("Let's Verify Step by Step"), weak-to-strong generalization, rule-based rewards, deliberative alignment, and the o-series. 2025–26 output centers on **CoT monitorability** (13-eval "Monitoring Monitorability" suite, arXiv:2512.18311), **anti-scheming training** with Apollo (covert actions o3 13% → 0.4%, confounded by rising eval awareness; arXiv:2509.15541), **confessions** for honesty (arXiv:2512.08093), the **hallucination-as-incentive** analysis (arXiv:2509.04664), and economic evals (**GDPval**, 1,320 tasks / 44 occupations). Model-level publication remains guarded (system cards, not technical reports), with gpt-oss's model card the main architectural disclosure.

## Compute & Infrastructure
From an exclusive Azure relationship to a **~26 GW multi-vendor portfolio** (late 2025): **NVIDIA** ≥10 GW (with up to $100B NVIDIA investment), **AMD** 6 GW across Instinct generations starting with MI450 in 2H 2026 (OpenAI holds warrants for up to ~10% of AMD), and **Broadcom** 10 GW of co-designed custom accelerators + networking; **Stargate**/Oracle sites (≈7 GW committed at the time) host training. **Cerebras** wafer-scale chips serve low-latency tiers (Codex-Spark; Sol planned at up to 750 tok/s). Amazon reportedly committed $50B and 2 GW of Trainium (single source). Funding: a record **$122B round at $852B post-money** (2026-03, secondary).

## Stated Strategic Direction
Build **AGI** safely and broadly beneficially; iterative deployment ("ship to learn") now at a ~4–8-week cadence; scaling **pretraining, RL and inference** compute together; agents that operate computers and run for hours as the primary interface (Codex, ChatGPT agent); and **tiered/trusted deployment** for dual-use capability (Daybreak; Astra's internal-deployment safeguards). Strong commercial focus (ChatGPT, API, enterprise) alongside the mission.

## Known Strengths & Weaknesses
- **Strengths:** frontier reasoning and agentic coding (Terminal-Bench, OSWorld leadership at several points in 2026), product distribution/brand (ChatGPT), the largest contracted compute pipeline, full multimodal + agent ecosystem, eval expertise (GDPval, monitorability), and fast tiered releases with steep price drops (Sol/Luna).
- **Weaknesses:** closed/opaque (system cards, not reports), launch-reliability issues (GPT-5 router), governance turbulence and restructuring, heavy capital/compute commitments, rapid model churn (5.5 retired from ChatGPT within ~6 months, single source), and close competition from Anthropic (agentic coding) and Google (price/perf, multimodal).

## Hiring Focus
RL/post-training for long-horizon agents, large-scale pretraining and inference systems (multi-vendor: NVIDIA/AMD/custom/Cerebras), evaluation (agentic, economic, monitorability), safety/preparedness (cyber, bio, scheming), computer-use/multimodal, and agentic product engineering. JDs emphasize scaling expertise, RL, distributed systems, and shipping reliable products at massive scale.

## Latest Developments (2023–Nov 2026)
2023–mid-2025 established the o-series (o1 → o3/o4-mini) and test-time compute as a scaling axis, GPT-4o's native multimodality, and GPT-4.1's 1M context. Since then OpenAI has unified chat and reasoning, re-scaled pretraining, and moved to tiered, gated deployment.

**Mid-2025 → Nov 2026 update:**
- **2025-08 — gpt-oss:** architectural disclosure of an OpenAI-style MoE (sparse 5.1B-active, sliding/dense alternation, attention sinks, MXFP4 MoE weights) — the reference design for "fits on one 80 GB GPU" open reasoning models. (OpenAI model card)
- **2025-08 — GPT-5 router:** reasoning depth became a *learned dispatch decision* (fast vs thinking model); the launch-day router failure showed routing quality is itself a user-facing capability.
- **2025-12 — Monitoring Monitorability:** CoT monitors beat action-only monitors, longer CoTs are more monitorable, direct CoT reward pressure degrades monitorability, and RL at current scale did not materially hurt it — a design constraint on reasoning-RL reward shaping. (arXiv:2512.18311)
- **2026-02 → 03 — Codex-Spark on Cerebras and GPT-5.4 computer use:** latency-tiered serving across heterogeneous hardware, and GUI control folded into the general model (~1.05M context).
- **2026-04 — GPT-5.5 / SWE-bench Verified retired:** OpenAI stopped reporting SWE-bench Verified (contamination, flawed tests) in favor of SWE-bench Pro; 5.5 emphasized fewer tokens per Codex task — cost-per-task as a first-class metric.
- **2026-06 → 09 — Sol/Terra/Luna and GPT-6:** three-tier families with ~2× price drops per generation at constant capability; GPT-6 Astra is the first "Critical"-cyber model under Preparedness Framework v2, deployed first to Daybreak enterprise customers.
- **Compute (2025-09 → 10):** ~26 GW across NVIDIA, AMD and Broadcom — compute is now contracted in gigawatts with vendor equity/warrant structures.

## Interview Angles
> 💡 **What labs actually ask (OpenAI-flavored):**
- **"Explain InstructGPT and why RLHF mattered."** Alignment > scale for usability; the SFT→RM→PPO pipeline.
- **"How does o1/o3 differ from GPT-4o — and why fold them into GPT-5?"** RL-trained reasoning + test-time compute vs fast multimodal chat; a router amortizes thinking cost over easy/hard queries but adds a failure mode.
- **"What is inference-time scaling and its tradeoffs?"** Trade test-time compute for accuracy; cost/latency; report tokens-per-task alongside accuracy.
- **"Walk through gpt-oss's architecture choices."** Top-4-of-128 MoE, sliding/dense alternation + attention sinks for long context, MXFP4 MoE weights for single-GPU serving.
- **"How would you design deployment for a 'Critical'-cyber model?"** Trusted-access tiers, identity verification, classifiers + fallback routing, internal-deployment safeguards.
- **"How would you design an eval for a reasoning model?"** (See [Evaluation](../05_alignment_and_safety/05_evaluation_and_benchmarks.md).)

## Open Problems / Watch Items
Governance/structure transition and IPO, executing on ~26 GW of contracted compute, retaining safety credibility as models cross "Critical" thresholds, keeping CoT monitorable under heavier RL, economic inference for reasoning agents, agent reliability in long-horizon autonomy, and maintaining a lead against Anthropic, Google and Chinese open-weight labs.

## References
- Brown, T. et al. (2020). *GPT-3.* arXiv:2005.14165.
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- OpenAI (2023). *GPT-4 Technical Report.* arXiv:2303.08774.
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050.
- OpenAI (2024). *Learning to Reason with LLMs (o1).*
- OpenAI (2025). *gpt-oss-120b & gpt-oss-20b Model Card.*
- Kalai, A. et al. (2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
- OpenAI & Apollo Research (2025). *Stress Testing Deliberative Alignment for Anti-Scheming Training.* arXiv:2509.15541.
- Guan, M. et al. (2025). *Monitoring Monitorability.* arXiv:2512.18311.
- Joglekar, M. et al. (2025). *Training LLMs for Honesty via Confessions.* arXiv:2512.08093.
