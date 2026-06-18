# Anthropic — Lab Roadmap

> **Last Updated:** 2026-06-18
> **Related Files:** [Constitutional AI & RLAIF](../03_posttraining/07_constitutional_AI_and_RLAIF.md) · [Interpretability](../05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md) · [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)
> **Key Papers:** Bai et al. 2022 Constitutional AI ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)) · Elhage et al. 2022 Toy Models of Superposition ([arXiv:2209.10652](https://arxiv.org/abs/2209.10652)) · Templeton et al. 2024 Scaling Monosemanticity · Bai et al. 2022 HH-RLHF ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862))

## Overview
Anthropic was founded in 2021 by former OpenAI researchers (Dario and Daniela Amodei and colleagues) with an explicit **safety-first** mission: build frontier AI while pioneering the techniques to make it safe and steerable. It is simultaneously a frontier lab (the **Claude** model family, competitive with GPT and Gemini) and the field's leading **alignment research** institution (Constitutional AI, mechanistic interpretability, scalable oversight). Its positioning — "we have to be at the frontier to do safety research that matters" — defines a distinctive strategy.

## Model Lineage & Key Innovations
- **Claude 1/2 (2023):** helpful-honest-harmless assistant trained with RLHF + **Constitutional AI**; early long-context leadership (100K tokens).
- **Claude 3 family — Haiku/Sonnet/Opus (2024):** tiered models (fast→capable), strong reasoning, vision, and document understanding; Opus a frontier model.
- **Claude 3.5/3.7 Sonnet (2024–2025):** leading **coding** and **agentic** performance; introduced **computer use** (controlling a GUI); strong on SWE-bench.
- **Claude with extended thinking (2025):** reasoning mode with visible, controllable "thinking" budget — Anthropic's entry into inference-time reasoning, with attention to **CoT faithfulness/monitorability**.
- **Claude 4-era (Opus/Sonnet, 2025–2026):** frontier coding/agentic models; the basis of products like **Claude Code**.

## Technical Differentiation
- **Constitutional AI / RLAIF** — principle-based alignment that reduces human-label dependence (see [CAI & RLAIF](../03_posttraining/07_constitutional_AI_and_RLAIF.md)).
- **Mechanistic interpretability leadership** — the Transformer Circuits thread, superposition, **sparse autoencoders / monosemanticity at scale**, attribution graphs ("biology of an LLM").
- **Scalable oversight** research (debate, sandwiching, weak-to-strong) tied to a concrete safety agenda.
- **Responsible Scaling Policy (RSP)** — capability-threshold "AI Safety Levels" (ASL) that gate deployment on safeguards.
- **Agentic coding** strength and **MCP** (open tool-connection standard).

## Published Research Focus
Alignment and safety: Constitutional AI, RLHF for HH, interpretability (circuits, SAEs), scalable oversight, **sleeper agents** and **alignment faking** (demonstrating deceptive-alignment risks), jailbreak defense (constitutional classifiers, many-shot jailbreaking discovery), and evaluation. Anthropic publishes prolifically on safety while keeping model details guarded.

## Compute & Infrastructure
Backed by major investments from **Amazon** (and Google); uses **AWS Trainium** and **Google TPUs** in addition to GPUs — a notably **multi-accelerator** strategy. Large-scale training infrastructure with emphasis on reliability and deterministic, reproducible training.

## Stated Strategic Direction
Reach the frontier to ensure safety research is relevant; **interpretability as a route to trustworthy AI** ("a kind of MRI for models"); scalable oversight for superhuman systems; **Responsible Scaling** as a governance template; and commercial strength in **coding/agents/enterprise** to fund safety research. Public emphasis on the seriousness of catastrophic/loss-of-control risk.

## Known Strengths & Weaknesses
- **Strengths:** alignment/interpretability research leadership, top-tier coding/agentic models, safety credibility and governance maturity (RSP), strong enterprise/coding traction, MCP ecosystem.
- **Weaknesses:** smaller consumer footprint/brand than ChatGPT, less of a broad multimodal/generative ecosystem (limited image/audio/video generation), compute dependence on cloud partners, and the inherent tension of racing while preaching caution.

## Hiring Focus
Alignment science, **mechanistic interpretability**, scalable oversight, RL/post-training, large-scale training & inference infra (incl. TPU/Trainium), evaluation/red-teaming, and agentic/coding product engineering. JDs strongly emphasize safety motivation, research rigor, and interpretability/RL skills.

## Interview Angles
> 💡 **What labs actually ask (Anthropic-flavored):**
- **"Walk through Constitutional AI."** SL critique-revise + RL with an AI preference model (see [CAI](../03_posttraining/07_constitutional_AI_and_RLAIF.md)).
- **"What is superposition, and how do SAEs help?"** (See [Interpretability](../05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md).)
- **"What is scalable oversight and why is it the core problem?"** Supervising superhuman models (debate/weak-to-strong).
- **"How would you detect deceptive alignment?"** Interpretability + evals; sleeper-agents/alignment-faking findings.

## Open Problems / Watch Items
Scaling interpretability to *complete, robust* model understanding; making scalable oversight actually work for superhuman systems; balancing frontier racing with safety; broadening product/multimodal footprint; and validating that RSP/ASL thresholds meaningfully gate risk.

## References
- Bai, Y. et al. (2022). *Constitutional AI.* arXiv:2212.08073.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Templeton, A. et al. (2024). *Scaling Monosemanticity.* Anthropic.
- Hubinger, E. et al. (2024). *Sleeper Agents.* arXiv:2401.05566.
- Greenblatt, R. et al. (2024). *Alignment Faking in LLMs.* arXiv:2412.14093.
