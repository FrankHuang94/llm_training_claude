# Interpretability and Mechanistic Analysis

> **Last Updated:** 2026-06-18
> **Related Files:** [Alignment Overview](00_alignment_overview.md) · [Scalable Oversight](04_scalable_oversight.md) · [Transformer Architecture](../00_foundations/00_transformer_architecture.md)
> **Key Papers:** Elhage et al. 2021 Mathematical Framework for Transformer Circuits · Olsson et al. 2022 In-context Learning & Induction Heads ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)) · Elhage et al. 2022 Toy Models of Superposition ([arXiv:2209.10652](https://arxiv.org/abs/2209.10652)) · Bricken et al. / Templeton et al. 2024 Scaling Monosemanticity (Anthropic)

## Overview
Mechanistic interpretability (MI) aims to reverse-engineer the *internal computations* of neural networks — to move from treating models as black boxes to understanding the algorithms they implement, expressed in terms of features and circuits. For alignment, MI offers something behavioral evaluation cannot: a way to **inspect what a model is actually doing** (and perhaps detect deception, hidden goals, or unsafe reasoning that behavior alone would miss). Anthropic has made MI a research pillar; the breakthrough use of **sparse autoencoders (SAEs)** to extract interpretable features from frontier models is among the most exciting recent developments. This is a flagship topic for Anthropic-style interviews.

## Core Concepts
**Features and circuits.** The "circuits" program treats a network as composed of **features** (directions in activation space representing concepts) connected by **circuits** (subgraphs of weights implementing a computation). The goal is to identify human-understandable features and the circuits that compute with them.

**Superposition.** Elhage et al. (2022): networks represent **more features than they have neurons** by encoding features as *nearly-orthogonal directions* in activation space, with individual neurons being **polysemantic** (responding to many unrelated concepts). Superposition is why looking at single neurons is misleading and is the core obstacle MI must overcome. It's a consequence of features being sparse and the network exploiting near-orthogonality (Johnson–Lindenstrauss).

**Sparse autoencoders (SAEs) / dictionary learning.** To undo superposition, train an SAE that maps activations into a **higher-dimensional, sparse** space where each dimension is (hopefully) a single **monosemantic** feature. Anthropic's "Towards Monosemanticity" (2023) and "Scaling Monosemanticity" (2024, on Claude 3 Sonnet) extracted millions of interpretable features (e.g., the "Golden Gate Bridge" feature, code-security features, deception-related features), and showed they can be **clamped** to steer behavior — strong evidence the decomposition is causal, not just correlational.

**Induction heads.** Olsson et al.: attention heads implementing a "**copy what followed last time**" pattern ($[A][B]\dots[A]\to[B]$) — a discovered circuit that is the primary mechanism behind much of **in-context learning**, whose formation coincides with a phase change in ICL ability during training.

**Other tools.** Activation patching / causal tracing (localize where a behavior is computed), logit lens (read intermediate predictions), probing (linear classifiers for concepts), and attention analysis. The **linear representation hypothesis** — that many concepts are encoded linearly — underpins much of this.

## Key Challenges
- **Scale.** Frontier models have billions of parameters and millions of features; comprehensive understanding is far off, and SAEs are expensive to train and incomplete.
- **Superposition & polysemanticity.** Make naive neuron-level analysis unreliable; SAEs help but have their own artifacts (feature splitting, dead features, reconstruction error).
- **Faithfulness vs plausibility.** An interpretation can be compelling but wrong; rigorous causal validation (ablation/patching) is required.
- **Completeness.** We can explain fragments (a circuit here, features there) but not whole-model behavior; "what fraction of the model do we understand?" is tiny.
- **Robustness for safety.** To *certify* safety we'd need interpretability robust to a model that might obfuscate its internals — not yet possible.

## Solutions & Current Best Practices
**SAE-based feature extraction + causal validation** (activation patching, feature clamping) is the current frontier methodology. Use interpretability for **concrete safety tasks**: detecting deception/sycophancy features, finding safety-relevant circuits, monitoring for dangerous concepts, and steering via feature intervention. Combine MI with behavioral evals (defense-in-depth). Open tooling (TransformerLens, SAE libraries, Neuronpedia) and community efforts (EleutherAI, academic MI) broaden the work beyond one lab.

## Lab Perspectives
- **Anthropic:** the leader — circuits thread, superposition, monosemanticity/SAEs at scale, and an explicit bet that interpretability is key to safe, trustworthy frontier models; recent **attribution graphs / "biology of an LLM"** tracing multi-step internal computation.
- **Google DeepMind:** strong MI team (Gemma Scope SAEs released openly, fact-localization, ROME/causal tracing lineage).
- **OpenAI:** automated interpretability (using LLMs to label neurons), sparse-feature work, weak-to-strong + interpretability for superalignment.
- **Academia/EleutherAI:** TransformerLens, grokking analyses, probing, and theory (linear representations).

## Latest Developments (2023–2026)
**Scaling SAEs to frontier models** (Claude 3 Sonnet, Gemma Scope) was the headline advance, yielding millions of interpretable, steerable features. **Attribution graphs / circuit tracing** (Anthropic 2025) began explaining *multi-step* reasoning internally (planning, multilingual circuits). Growing use of MI for **safety auditing** (detecting deception, eval-awareness). Open debates: do SAE features carve the model at its true joints? Can MI scale to *complete* understanding, and be made **adversarially robust** enough to certify safety?

## Interview Angles
> 💡 **What labs actually ask:**
- **"What is superposition and why does it matter?"** Models pack >neurons features as near-orthogonal directions → polysemantic neurons → need dictionary learning/SAEs.
- **"How do sparse autoencoders aid interpretability?"** Decompose activations into sparse, monosemantic features; validate causally by clamping/patching.
- **"What are induction heads?"** Copy-previous-occurrence attention circuit; primary driver of in-context learning; phase change during training.
- **"How could interpretability help alignment?"** Detect deception/hidden goals, monitor/steer internals, audit safety — beyond what behavior reveals.

## Open Problems
Whether MI can **scale to complete, reliable understanding** of frontier models — and be made **robust to a model actively obfuscating its cognition** — is the central uncertainty. The correctness/completeness of SAE features, automating circuit discovery, and turning MI into trustworthy safety *guarantees* (not just insights) are all open.

## References
- Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic.
- Olsson, C. et al. (2022). *In-context Learning and Induction Heads.* arXiv:2209.11895.
- Elhage, N. et al. (2022). *Toy Models of Superposition.* arXiv:2209.10652.
- Templeton, A. et al. (2024). *Scaling Monosemanticity.* Anthropic.
- Lindsey, J. et al. (2025). *On the Biology of a Large Language Model (attribution graphs).* Anthropic.
