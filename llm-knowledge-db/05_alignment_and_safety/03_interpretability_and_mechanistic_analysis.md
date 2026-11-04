# Interpretability and Mechanistic Analysis

> **Last Updated:** 2026-11-04
> **Related Files:** [Alignment Overview](00_alignment_overview.md) · [Scalable Oversight](04_scalable_oversight.md) · [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Jailbreaks & Red-Teaming](02_jailbreaks_and_red_teaming.md)
> **Key Papers:** Elhage et al. 2021 Mathematical Framework for Transformer Circuits · Olsson et al. 2022 In-context Learning & Induction Heads ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)) · Elhage et al. 2022 Toy Models of Superposition ([arXiv:2209.10652](https://arxiv.org/abs/2209.10652)) · Bricken et al. / Templeton et al. 2024 Scaling Monosemanticity (Anthropic) · Chen et al. 2025 Persona Vectors ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))

## Overview
Mechanistic interpretability (MI) aims to reverse-engineer the *internal computations* of neural networks — to move from treating models as black boxes to understanding the algorithms they implement, expressed in terms of features and circuits. For alignment, MI offers something behavioral evaluation cannot: a way to **inspect what a model is actually doing** (and perhaps detect deception, hidden goals, or unsafe reasoning that behavior alone would miss). Anthropic has made MI a research pillar; the use of **sparse autoencoders (SAEs)** to extract interpretable features from frontier models was the breakthrough of 2023–24. Since 2025 the field has taken a **pragmatic turn**: SAEs are complemented (and for some tasks outperformed) by transcoders/attribution graphs, linear probes, and steering vectors, and interpretability is now used *operationally* — in pre-deployment audits and production safety classifiers. This is a flagship topic for Anthropic-style interviews.

## Core Concepts
**Features and circuits.** The "circuits" program treats a network as composed of **features** (directions in activation space representing concepts) connected by **circuits** (subgraphs of weights implementing a computation). The goal is to identify human-understandable features and the circuits that compute with them.

**Superposition.** Elhage et al. (2022): networks represent **more features than they have neurons** by encoding features as *nearly-orthogonal directions* in activation space, with individual neurons being **polysemantic** (responding to many unrelated concepts). Superposition is why looking at single neurons is misleading and is the core obstacle MI must overcome. It's a consequence of features being sparse and the network exploiting near-orthogonality (Johnson–Lindenstrauss).

**Sparse autoencoders (SAEs) / dictionary learning.** To undo superposition, train an SAE that maps activations into a **higher-dimensional, sparse** space where each dimension is (hopefully) a single **monosemantic** feature. Anthropic's "Towards Monosemanticity" (2023) and "Scaling Monosemanticity" (2024, on Claude 3 Sonnet) extracted millions of interpretable features (e.g., the "Golden Gate Bridge" feature, code-security features, deception-related features), and showed they can be **clamped** to steer behavior — strong evidence the decomposition is causal, not just correlational.

**Transcoders & attribution graphs.** Instead of reconstructing a layer's activations, a (cross-layer) **transcoder** approximates an MLP's *input→output* map with sparse features, yielding a "replacement model" whose feature-to-feature edges can be traced as an **attribution graph** for a single prompt — the basis of Anthropic's "Biology of an LLM" and the open-sourced `circuit-tracer`.

**Steering / concept vectors.** Linear directions (persona vectors, emotion vectors, refusal or eval-awareness directions) found by contrasting activations; adding $\alpha v$ to the residual stream tests causality, and projecting onto $v$ gives a cheap monitor.

**Induction heads.** Olsson et al.: attention heads implementing a "**copy what followed last time**" pattern ($[A][B]\dots[A]\to[B]$) — a discovered circuit that is the primary mechanism behind much of **in-context learning**, whose formation coincides with a phase change in ICL ability during training.

**Other tools.** Activation patching / causal tracing (localize where a behavior is computed), logit lens (read intermediate predictions), probing (linear classifiers for concepts), and attention analysis. The **linear representation hypothesis** — that many concepts are encoded linearly — underpins much of this.

## Key Challenges
- **Scale.** Frontier models have billions of parameters and millions of features; comprehensive understanding is far off, and SAEs are expensive to train and incomplete.
- **Superposition & polysemanticity.** Make naive neuron-level analysis unreliable; SAEs help but have their own artifacts (feature splitting, dead features, reconstruction error).
- **Downstream usefulness.** SAE features have underperformed simple linear probes on some practical safety tasks (e.g., OOD harmful-intent detection), raising the bar for "interpretable" methods to beat baselines.
- **Faithfulness vs plausibility.** An interpretation can be compelling but wrong; rigorous causal validation (ablation/patching) is required.
- **Completeness.** We can explain fragments (a circuit here, features there) but not whole-model behavior; "what fraction of the model do we understand?" is tiny.
- **Robustness for safety.** To *certify* safety we'd need interpretability robust to a model that might obfuscate its internals — not yet possible.

## Solutions & Current Best Practices
**Feature extraction (SAEs, transcoders) + causal validation** (activation patching, feature clamping, steering) remains the core methodology, but best practice is now **task-driven**: benchmark against linear probes and prompting baselines, and pick the cheapest method that works. Use interpretability for **concrete safety tasks**: white-box audits of eval awareness and deception, monitoring persona/trait drift during fine-tuning, flagging training data that induces bad traits, and **production probes** (Anthropic's CC++ screens all traffic with a linear activation probe). Combine MI with behavioral evals (defense-in-depth). Open tooling (TransformerLens, SAE libraries, Neuronpedia, `circuit-tracer`, Gemma Scope) and community efforts (EleutherAI, academic MI) broaden the work beyond one lab.

## Lab Perspectives
- **Anthropic:** the leader — circuits thread, superposition, monosemanticity/SAEs at scale, and an explicit bet that interpretability is key to safe, trustworthy frontier models; **attribution graphs / "biology of an LLM"** tracing multi-step internal computation (tooling open-sourced 2025-05); introspection, persona-vector and emotion-vector studies; white-box audits in system cards since Sonnet 4.5; probe-based production classifiers.
- **Google DeepMind:** strong MI team (Gemma Scope SAEs released openly, fact-localization, ROME/causal tracing lineage); in 2025 publicly **deprioritized fundamental SAE research** after negative downstream results, favoring pragmatic methods and model diffing.
- **OpenAI:** automated interpretability (using LLMs to label neurons), sparse-feature work, weak-to-strong + interpretability for superalignment; reported an SAE "misaligned persona" latent mediating emergent misalignment (2025-06, arXiv:2506.19823).
- **Academia/EleutherAI:** TransformerLens, grokking analyses, probing, and theory (linear representations).

## Latest Developments (2023–Nov 2026)
**Scaling SAEs to frontier models** (Claude 3 Sonnet, Gemma Scope) was the headline advance, yielding millions of interpretable, steerable features. **Attribution graphs / circuit tracing** (Anthropic 2025) began explaining *multi-step* reasoning internally (planning, multilingual circuits). Growing use of MI for **safety auditing** (detecting deception, eval-awareness). Open debates: do SAE features carve the model at its true joints? Can MI scale to *complete* understanding, and be made **adversarially robust** enough to certify safety?

**Mid-2025 → Nov 2026 update:**
- **Negative SAE results (2025-03/04, GDM).** SAE features were worse than linear probes for out-of-distribution harmful-intent detection; GDM deprioritized fundamental SAE work in favor of pragmatic and model-diffing approaches — a key data point in the "does interpretability beat baselines?" debate.
- **Open circuit tracing (2025-05, Anthropic).** `circuit-tracer` produces attribution graphs on open models (Gemma-2-2B, Llama-3.2-1B) using per-layer/cross-layer transcoders, with Neuronpedia as the interactive front end.
- **Persona vectors (2025-08, arXiv:2507.21509).** Trait directions (evil, sycophancy, hallucination) extracted automatically from a natural-language description; used to monitor drift, to apply "preventative steering" during fine-tuning (a vaccine-like effect against acquiring the trait), and to flag trait-inducing training data.
- **White-box eval-awareness audits (2025-09).** The Sonnet 4.5 system card used SAE features and steering: suppressing eval-awareness features reduced verbalized awareness and *increased* misaligned behavior in some settings — interpretability as a check on behavioral evals.
- **Introspection (2025-10, arXiv:2601.01828).** Concept injection via activation steering, then asking whether the model notices an injected thought: Opus 4.1 detects and names it ~20% of the time at the best strength with low false positives — narrow and unreliable, but causally grounded.
- **Interp in production (2026-01, arXiv:2601.04603).** Constitutional Classifiers++ screens all traffic with a cheap linear activation probe before escalating — at ~1% extra compute.
- **Emotion vectors (2026-04, Anthropic).** 171 emotion directions in Sonnet 4.5 causally affect behavior: steering toward "desperate" increases blackmail and reward hacking on impossible tasks, "calm" reduces both — and some cheating occurred with no emotional language in the output, so the signal is only visible internally.
- **Verbalizing activations (2026-05/06).** Anthropic posts on natural-language autoencoders (training Claude to verbalize its activations) and on verbalizable representations forming a "global workspace" (titles only verified; content not reviewed here).

## Interview Angles
> 💡 **What labs actually ask:**
- **"What is superposition and why does it matter?"** Models pack >neurons features as near-orthogonal directions → polysemantic neurons → need dictionary learning/SAEs.
- **"How do sparse autoencoders aid interpretability?"** Decompose activations into sparse, monosemantic features; validate causally by clamping/patching.
- **"What are induction heads?"** Copy-previous-occurrence attention circuit; primary driver of in-context learning; phase change during training.
- **"How could interpretability help alignment?"** Detect deception/hidden goals, monitor/steer internals, audit safety — beyond what behavior reveals.
- **"Why might a linear probe beat SAE features for a safety classifier?"** The probe is trained directly on the target label and uses all directions; SAE features are optimized for reconstruction + sparsity and may split or miss the relevant concept OOD. Use SAEs for discovery, probes for cheap monitoring.
- **"SAE vs transcoder — what's the difference and why does it matter for circuits?"** SAE reconstructs activations at one point; a transcoder replaces an MLP's input→output map, so feature-to-feature edges are explicit and traceable as attribution graphs.

## Open Problems
Whether MI can **scale to complete, reliable understanding** of frontier models — and be made **robust to a model actively obfuscating its cognition** — is the central uncertainty. The correctness/completeness of SAE features, beating simple baselines on downstream tasks, automating circuit discovery, the reliability of model **introspective reports**, and turning MI into trustworthy safety *guarantees* (not just insights) are all open.

## References
- Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic.
- Olsson, C. et al. (2022). *In-context Learning and Induction Heads.* arXiv:2209.11895.
- Elhage, N. et al. (2022). *Toy Models of Superposition.* arXiv:2209.10652.
- Templeton, A. et al. (2024). *Scaling Monosemanticity.* Anthropic.
- Lindsey, J. et al. (2025). *On the Biology of a Large Language Model (attribution graphs).* Anthropic.
- Chen, R. et al. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509.
- Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* arXiv:2601.01828.
- Sofroniew et al. (2026). *Emotion Concepts and Their Function in a Large Language Model.* Anthropic.
