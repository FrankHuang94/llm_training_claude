# Constitutional AI and RLAIF

> **Last Updated:** 2026-11-04
> **Related Files:** [RLHF](03_RLHF.md) · [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md) · [Alignment Overview](../05_alignment_and_safety/00_alignment_overview.md) · [Jailbreaks & Red-Teaming](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md)
> **Key Papers:** Bai et al. 2022 Constitutional AI ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)) · Lee et al. 2023 RLAIF ([arXiv:2309.00267](https://arxiv.org/abs/2309.00267)) · Bai et al. 2022 HH-RLHF ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862)) · OpenAI 2024 Rule-Based Rewards ([arXiv:2411.01111](https://arxiv.org/abs/2411.01111)) · Constitutional Classifiers++ 2026 ([arXiv:2601.04603](https://arxiv.org/abs/2601.04603))

## Overview
Constitutional AI (CAI), introduced by Anthropic in 2022, addresses a core bottleneck of RLHF: its dependence on large volumes of **human** preference labels, which are slow, expensive, inconsistent, and require humans to read harmful content. CAI replaces most human feedback with **AI feedback** guided by an explicit, written set of principles (a "constitution"). The model critiques and revises its own outputs according to the constitution, and an AI preference model — rather than human raters — provides the RL signal. More broadly, **RLAIF (RL from AI Feedback)** is the family of methods using model-generated preferences in place of human ones. These techniques are central to Anthropic's safety strategy and to the broader question of **scalable oversight** — supervising models as they approach or exceed human ability. In 2026 Anthropic replaced its 2023 rule list with a ~23K-word constitution that *explains its reasoning*, and the same principle-driven approach now also powers deployment-time **constitutional classifiers**.

## Core Concepts
**The constitution.** A list of natural-language principles (drawn from sources like the UN Declaration of Human Rights, trust-and-safety guidelines, and Anthropic's own values) such as "choose the response that is least harmful" or "...most supportive of freedom and dignity." Principles are *transparent and editable*, making the value specification explicit rather than implicit in opaque human labels. The 2026 rewrite shifts from short rules to explained reasoning, with an explicit **priority order** for conflicts — (1) broadly safe (don't undermine legitimate human oversight), (2) broadly ethical, (3) compliant with Anthropic's guidelines, (4) genuinely helpful — plus a small set of hard constraints.

**Two-phase CAI pipeline.**
1. **Supervised (SL-CAI) — critique & revise.** Prompt the model with red-teaming/harmful queries; it generates a response, then is asked to **critique** its own response against a (randomly sampled) constitutional principle and **revise** it. Fine-tune on the revised responses. This produces a model that is harmless-by-self-revision without human harm labels.
2. **RL (RL-CAI) — AI preference model.** Generate response pairs; ask the model (the "feedback model") to choose which better follows a sampled principle, producing an **AI-labeled preference dataset**. Train a preference model on these AI labels, then run standard RL (PPO) against it. The harmlessness signal is now entirely AI-generated; helpfulness can still use human feedback.

**RLAIF generally.** Lee et al. (2023, Google) showed RLAIF can match RLHF on summarization/dialogue helpfulness, even when the AI labeler is the same size as the policy, establishing AI feedback as a viable substitute beyond just harmlessness. **Rule-Based Rewards** (OpenAI 2024) is a related idea: encode safety behaviors as explicit rules graded by an LLM, used as RL reward, for precise, auditable safety control. **Deliberative alignment** (OpenAI) trains models to reason over a written spec in their CoT before answering.

**Connection to scalable oversight.** CAI/RLAIF are early instances of using AI to *amplify* human oversight: humans write principles (high-leverage, low-volume) and AI does the labor-intensive labeling. This is a stepping stone toward overseeing systems too capable for direct human evaluation (see [Scalable Oversight](../05_alignment_and_safety/04_scalable_oversight.md)). By 2026 the AI also runs *alignment research loops* itself (literature → method → training → eval).

**Constitutional classifiers.** The constitution also defines what deployment-time input/output classifiers block. Constitutional Classifiers++ uses a cascade: a cheap linear activation probe screens all traffic, and flagged exchanges escalate to a probe+classifier ensemble that judges outputs in the context of inputs.

## Key Challenges
- **Whose values?** A constitution encodes contested value judgments; who writes it and how to handle pluralism is a governance problem (Anthropic's "Collective Constitutional AI" experimented with public input).
- **AI-labeler bias & reliability.** AI feedback inherits the labeler model's biases, blind spots, and miscalibration; errors can be systematic and self-reinforcing.
- **Sycophancy & reward hacking.** AI preference models can be gamed just like human-trained RMs; self-evaluation may be biased toward the model's own style.
- **Helpfulness–harmlessness tension.** Over-aggressive harmlessness causes over-refusal; balancing the two is delicate.
- **Verification ceiling.** AI feedback is reliable only where the labeler is competent — it doesn't solve oversight for truly superhuman outputs.
- **Evaluation awareness.** Spec-based anti-scheming training *increases* models' explicit awareness of being evaluated, confounding measured gains.

## Solutions & Current Best Practices
**Hybrid feedback**: AI feedback for scalable, sensitive, or high-volume judgments (harmlessness, format, rule compliance), human feedback for nuanced helpfulness and for *validating* the AI labeler. Use **explicit, auditable principles/rules** (constitutions, rule-based rewards, model specs) and explain the *reasons* behind them so behavior generalizes to unanticipated cases. Combine with **red-teaming** to source hard prompts (see [Jailbreaks & Red-Teaming](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md)). Monitor for over-refusal and sycophancy. Pair training-time CAI with **cheap deployment-time classifiers** (probe cascades). CAI/RLAIF are now standard components of frontier alignment stacks, often alongside RLHF and RLVR.

## Lab Perspectives
- **Anthropic:** originated and centers CAI/RLAIF; the 2026-01 constitution (CC0, primarily authored by Amanda Askell) is used at several training stages, including Claude-generated synthetic data and preference rankings; Constitutional Classifiers++ guard deployments; automated alignment researchers now close safety gaps.
- **OpenAI:** uses **Rule-Based Rewards**, a "Model Spec", and deliberative alignment against scheming; confessions as an honesty-only reward channel.
- **Google DeepMind:** demonstrated RLAIF viability (Lee et al.) and uses AI feedback in Gemini alignment.
- **Meta/DeepSeek/others:** increasingly use AI-feedback and synthetic preference data to cut labeling costs.

## Latest Developments (2023–Nov 2026)
AI feedback became the **default at scale** because human labeling cannot keep up with data needs. **Model Specs / explicit behavior specifications** (OpenAI, Anthropic) made target behavior auditable. **Collective/democratic** constitution-setting experiments addressed the "whose values" critique. RLAIF increasingly blends with **RLVR** (verifiable + AI-judge rewards) for a unified post-training stack, and "**self-rewarding**" / "**LLM-as-judge**" training loops proliferated, with active study of their biases and failure modes.

**Mid-2025 → Nov 2026 update:**
- **New Claude constitution (2026-01-22).** A ~23K-word rewrite released under CC0: four-level priority order, a few hard constraints, and a move from rules to explained reasoning; it also states uncertainty about Claude's moral status.
- **Constitutional Classifiers++ (2026-01).** Linear-probe screening plus escalation to exchange classifiers costs ~1% extra compute (vs 23.7% for the first generation), with a 0.05% refusal rate on production traffic and no universal jailbreak found in 1,700+ hours / 198K red-team attempts ([arXiv:2601.04603](https://arxiv.org/abs/2601.04603)).
- **Spec-based anti-scheming training (2025-09).** OpenAI + Apollo deliberative alignment cut covert actions ~30× (o3 13% → 0.4%, o4-mini 8.7% → 0.3%), but residual failures remained and evaluation awareness rose ([arXiv:2509.15541](https://arxiv.org/abs/2509.15541)).
- **Inoculation prompting (2025-10/11).** Recontextualizing a misbehavior as permitted in the *training prompt* stops it from generalizing to broad misalignment; Anthropic uses it in production ([arXiv:2510.05024](https://arxiv.org/abs/2510.05024), [arXiv:2511.18397](https://arxiv.org/abs/2511.18397)) — a principle-level intervention in the spirit of CAI.
- **Rubric rewards as instance-level constitutions.** Rubrics as Rewards ([arXiv:2507.17746](https://arxiv.org/abs/2507.17746)) and Rubric-ARM ([arXiv:2602.01511](https://arxiv.org/abs/2602.01511)) generate per-prompt principles that an AI judge scores.
- **AI-run alignment (2026-08).** Anthropic's automated researchers closed 26–96% of the safety gap across 10 failure classes; a weaker Sonnet 5 improving an early Opus 4.8 checkpoint closed 65% of the gap in 60 h (vs 72% for the production procedure) with ~2K examples — a weak-to-strong RLAIF-style result.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Walk through the Constitutional AI pipeline."** SL critique-revise phase → RL phase with an AI preference model labeling by sampled principles.
- **"RLHF vs RLAIF — what changes and why?"** Replace human preference labels with AI labels guided by principles; scalable, cheaper, consistent, but inherits labeler bias.
- **"How does CAI relate to scalable oversight?"** Humans specify principles (low-volume, high-leverage); AI does labeling — amplifying limited human oversight.
- **"Risks of AI feedback?"** Bias inheritance, reward hacking, sycophancy, verification ceiling for superhuman outputs.
- **"Why would a constitution explain its reasoning rather than list rules?"** Explained values generalize to unanticipated situations; rules are brittle and can be gamed; an explicit priority order resolves conflicts (safety > ethics > guidelines > helpfulness).

## Open Problems
AI feedback fundamentally cannot exceed the *competence* of the labeler, so it doesn't solve oversight of genuinely superhuman behavior — the heart of the alignment problem. Whose values a constitution should encode, how to make AI labelers robust to gaming, and how to validate AI feedback without expensive human ground truth remain open and consequential. Measuring whether principle-based training actually changes behavior, when models can tell they are being tested, is now a first-order open problem.

## References
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.
- Lee, H. et al. (2023). *RLAIF.* arXiv:2309.00267.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Mu, T. et al. (2024). *Rule-Based Rewards for Language Model Safety.* arXiv:2411.01111.
- Anthropic (2023). *Collective Constitutional AI.*
- Anthropic (2026). *Claude's Constitution.*
- Anthropic (2026). *Constitutional Classifiers++.* arXiv:2601.04603.
- OpenAI & Apollo Research (2025). *Stress Testing Deliberative Alignment for Anti-Scheming Training.* arXiv:2509.15541.
