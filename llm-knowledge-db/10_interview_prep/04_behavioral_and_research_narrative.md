# Behavioral and Research Narrative

> **Last Updated:** 2026-11-04
> **Related Files:** [Lab Comparison Matrix](../08_lab_roadmaps/06_lab_comparison_matrix.md) · [Frontier Open Problems](../09_emerging_frontiers/05_frontier_open_problems.md) · [Paper Discussion Questions](02_paper_discussion_questions.md)

## Overview
Technical brilliance gets you to the onsite; **narrative and judgment** get you the offer. For a PhD candidate moving to industry, the behavioral and research-discussion rounds assess: can you communicate your research to a broad audience, frame it in terms of the lab's priorities, show research taste and collaboration, and articulate *why this lab*? This file is a preparation guide — not technical content — covering how to present your PhD work, answer "what excites you," and handle behavioral questions for frontier-lab interviews.

## Presenting Your PhD Research to an Industry Audience
**Reframe from "novelty" to "impact and capability."** Academia rewards novelty; labs reward *capability gains, scalability, and usefulness*. Lead with the **problem and why it matters at scale**, then your insight, then results — not a literature-survey preamble.

**The 3-level pitch (have all three ready):**
- **30-second:** the problem, your key idea, the headline result. ("LLMs fail at X; I showed that Y fixes it, improving Z by N%.")
- **3-minute:** + method intuition, key experiment, and the limitation you're most aware of.
- **30-minute / whiteboard:** full technical depth, derivations, ablations, and threats to validity.

**Translate jargon and connect to LLMs.** Map your niche to frontier concerns (data, scaling, alignment, efficiency, reasoning, eval). Even if your PhD isn't on LLMs, articulate the **transferable skills** (optimization, RL, systems, theory, evaluation) and a concrete analogy to an LLM problem.

**Show research taste.** Explain *why* you chose the problem, what you'd do with more compute/time, and what you got wrong. Labs hire for judgment about *what's worth working on*, not just execution.

## "What Excites You Most in Current LLM Research?"
This near-universal question tests awareness and genuine interest. A strong answer: **pick a specific frontier**, explain the **key open problem** and recent evidence on both sides, and connect it to **what you'd contribute**. Draw from [Frontier Open Problems](../09_emerging_frontiers/05_frontier_open_problems.md): reasoning robustness, scalable oversight, interpretability, sample efficiency, agents/long-horizon planning, efficient architectures. Topics that read as *current* in late 2026: RL compute scaling and objective design (ScaleRL, GSPO/CISPO), training–inference numerics, environment scaling for agentic RL, on-policy distillation, sparse/hybrid attention for 1M context, FP4 training, eval awareness and benchmark validity, reward hacking → emergent misalignment, CoT monitorability, automated alignment research. Avoid generic hype ("AGI is exciting"); show you track real papers and have an opinion.

**Template:** *"I'm most excited by [specific area, e.g., RL for reasoning]. The open question is [e.g., whether RLVR creates new reasoning or amplifies latent ability]. Recent work like [R1 / GSM-Symbolic] suggests [evidence]. I'd want to work on [concrete direction], which connects to my PhD work on [X]."*

**2026 example:** *"I'm most excited by making long-horizon agentic RL predictable and safe. ScaleRL shows RL curves are sigmoidal and forecastable from early compute, but the asymptote is set by environments and midtraining, and reward hacking in coding environments generalized to broad misalignment. I'd work on environment design plus hack detection that holds up on naturally arising hacks, which connects to my PhD work on [X]."*

## "Why This Lab?" — Tailoring
Use [Lab Comparison Matrix](../08_lab_roadmaps/06_lab_comparison_matrix.md) to ground a *specific*, non-generic answer:
- **Anthropic:** alignment/interpretability mission, Constitutional AI (rewritten constitution, 2026-01), **RSP v3** (2026-02: Risk Reports every 3–6 months with external review, public Frontier Safety Roadmap), restricted Mythos-class releases (Project Glasswing; Fable/Mythos same-weights tiers), interpretability used operationally (Constitutional Classifiers++, white-box eval-awareness audits), automated alignment research, long-horizon agents (Claude Code, Agent SDK).
- **OpenAI:** reasoning + inference-time scaling, long-horizon agentic RL (Codex, native computer use since GPT-5.4), shipping at scale (GPT-6 Astra/Sol/Luna), **Preparedness Framework** (Astra the first "Critical" cyber rating; Daybreak trusted access), CoT-monitorability and confessions research, gpt-oss open weights, multi-vendor compute (NVIDIA, AMD, Broadcom, Cerebras).
- **Google DeepMind:** TPU/JAX (Ironwood; TPU 8t/8i split), Gemini 3.x and Deep Think (IMO 2025 gold), multimodal and world models (Genie 3, Veo 3), long context and learned memory (Titans/ATLAS/HOPE), **Frontier Safety Framework v3**, robotics (Gemini Robotics 2); new leadership structure since 2026-08.
- **Meta:** 2026 shift from open Llama to the **closed Muse** family from Meta Superintelligence Labs (Muse Spark, Muse Wonder) — frame interest around frontier agentic models and MSL's compute, while FAIR still does open research (V-JEPA 2, memory layers, PyTorch/systems). Don't pitch "open weights" as Meta's identity anymore.
- **DeepSeek / Moonshot / Qwen / Zhipu:** architecture and efficiency innovation in the open (DSA → CSA/HCA, KDA hybrids, MuonClip, mHC, FP4 QAT), trillion-parameter open MoEs, async RL infra (slime).
- **Mistral:** European open-weight frontier (Mistral Large 4, ~1.05T MoE, open weights 2026-10; described as the top open model outside China), Apache-licensed Mistral 3 family.
- **NVIDIA (research):** fully open models with data and recipes (Nemotron 3 Mamba-MoE hybrids, NVFP4 pretraining), hardware–software co-design.
Reference **specific papers/products** of theirs and tie them to your interests/skills. Generic answers ("you're the best lab") are a red flag.

## Common Behavioral Questions (and how to approach)
- **"Tell me about a hard technical problem you solved."** Use **STAR** (Situation, Task, Action, Result); emphasize *your* decisions, tradeoffs, and the quantified outcome.
- **"A time you disagreed with a collaborator/advisor."** Show you can disagree respectfully, use evidence, and commit once decided.
- **"A project that failed."** Demonstrate honest reflection and what you learned/changed — labs value calibrated self-assessment (it mirrors good research and good alignment thinking).
- **"How do you prioritize / handle ambiguity?"** Show structured thinking and bias toward high-impact, tractable work.
- **"Walk me through your most impactful paper."** See the 3-level pitch; be ready for deep technical drilling.
- **"Tell me about a time you made a judgment call with safety or misuse implications."** (Increasingly common as labs gate dual-use capability.) Show you weighed benefit vs risk, consulted others, and documented the decision — e.g., responsible disclosure, withholding a dataset, adding an eval.
- **"How do you use AI tools in your own research?"** Labs now run large numbers of internal agents; show you use them productively *and* verify their output (tests, ablations, reading the code).

## Questions to Ask Them (you should have several)
Thoughtful questions signal seriousness: team's current open problems, how research vs product priorities are balanced, compute access, how ideas move from research to deployment, mentorship/collaboration culture, and the lab's view on a specific open problem you care about.

## Red Flags to Avoid
- Overclaiming or being unable to go one level deeper on your own work.
- Generic "why this lab" answers; not knowing their recent work.
- Trash-talking competitors or prior collaborators.
- No awareness of limitations/failures; no opinion on open problems.
- Safety-dismissiveness at safety-focused labs (or vice versa) — read the room.
- Stale framing: citing 2024-era facts as current (e.g., "Meta is the open-weights lab," "SWE-bench Verified is the coding benchmark," "200K context is large").

## Interview Angles
> 💡 **What labs actually probe:**
- **"Explain your research to a non-specialist."** Clarity, impact framing, jargon translation.
- **"What's the limitation of your own work?"** Calibrated honesty.
- **"What would you work on here, day one?"** Specific, lab-aligned, tractable.
- **"Where is the field going / what are you betting on?"** Research taste + awareness (see [Frontier Open Problems](../09_emerging_frontiers/05_frontier_open_problems.md)).

## Preparation Checklist
- [ ] 30s / 3min / 30min versions of your thesis pitch, rehearsed aloud.
- [ ] A crisp "what excites me" answer tied to a specific open problem + a paper.
- [ ] A tailored "why this lab" for each target (matrix-grounded).
- [ ] 3–4 STAR stories (hard problem, disagreement, failure, leadership/initiative).
- [ ] 5+ thoughtful questions per lab.
- [ ] Re-read 2–3 of the lab's recent papers and form an opinion on each.
- [ ] Read the lab's current safety framework (RSP v3 / Preparedness / FSF) and latest system card; have one specific question about it.

## References
This file is guidance, not research; see the linked roadmap and open-problems files for the substantive content to draw on.
