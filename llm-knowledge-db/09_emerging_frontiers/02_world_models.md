# World Models

> **Last Updated:** 2026-06-18
> **Related Files:** [Audio, Video and Beyond](../07_multimodal/03_audio_video_and_beyond.md) · [Non-Transformer Architectures](04_neuromorphic_and_non_transformer.md) · [Frontier Open Problems](05_frontier_open_problems.md)
> **Key Papers:** Ha & Schmidhuber 2018 World Models ([arXiv:1803.10122](https://arxiv.org/abs/1803.10122)) · OpenAI 2024 Sora (Video Generation as World Simulators) · Bruce et al. 2024 Genie ([arXiv:2402.15391](https://arxiv.org/abs/2402.15391)) · LeCun 2022 A Path Towards Autonomous Machine Intelligence (JEPA)

## Overview
A **world model** is an internal, predictive model of how an environment evolves — enabling an agent to *imagine* the consequences of actions and plan without acting in the real world. The concept (Ha & Schmidhuber 2018) is central to model-based RL and, increasingly, to debates about LLMs and video models. Two big questions animate this frontier: (1) Are large generative video models (Sora, Veo, Genie) **implicit world models** that learn physics and dynamics? (2) Are **LLMs** themselves world models, or merely statistical pattern-matchers lacking grounded understanding? Yann LeCun's **JEPA** program is the most prominent alternative architecture explicitly designed to build world models. This is a conceptually deep, hotly contested area.

## Core Concepts
**Classic world models (Ha & Schmidhuber).** Learn a compressed latent representation of observations (a VAE) + a predictive dynamics model (an RNN) of how the latent evolves given actions; an agent ("controller") then plans/acts in this learned "dream" — even training **entirely inside the imagined world**. This established the model-based-RL paradigm (later: Dreamer, MuZero).

**Video generation as world simulation (Sora).** OpenAI framed Sora as a step toward "**general-purpose simulators of the physical world**": trained to generate video, it appears to learn aspects of 3D consistency, object permanence, and crude physics as **emergent** byproducts. Proponents argue scaling video generation yields world models; skeptics note frequent physics violations (objects morphing, impossible dynamics) revealing it models *appearance statistics*, not *causal physics*.

**Interactive/action-conditioned world models (Genie).** DeepMind's **Genie** learned, from unlabeled internet videos, a **latent action space** and a model that generates the *next frame conditioned on an inferred action* — effectively a learned, playable 2D world generator. Genie 2/3 extended this to richer 3D, controllable, longer-horizon environments — a route to training embodied agents in generated worlds.

**The LLM debate.** Do LLMs build world models? Evidence *for*: probing reveals models can encode latent state (e.g., the **Othello-GPT** board state, spatial/temporal world representations, entity tracking). Evidence *against*: brittleness, inconsistency, and reliance on surface statistics suggest no robust, causal world model. The truth is likely *partial* — LLMs build **fragmentary, task-useful** world representations without a coherent, complete model.

**JEPA (LeCun's alternative).** Joint-Embedding Predictive Architecture: instead of generating pixels/tokens, predict in an **abstract representation space** (predict the *embedding* of the future, not its raw form), trained non-contrastively (I-JEPA, V-JEPA). LeCun argues generative token/pixel prediction is the *wrong* objective for world modeling — you should predict abstract, predictable structure and discard unpredictable detail — and that this, with energy-based planning, is the path to grounded machine intelligence beyond autoregressive LLMs.

## Key Challenges
- **Physics vs appearance.** Generative video models capture look, not reliable causal dynamics; long-horizon consistency and physical plausibility fail.
- **Grounding.** LLMs learn from text *about* the world, not the world; whether language alone suffices for robust world models is contested.
- **Evaluation.** "Does it have a world model?" is hard to operationalize; probing is suggestive but not definitive.
- **Planning & action.** Turning a predictive model into reliable long-horizon planning (vs reactive generation) is unsolved.
- **Compute.** Video/world models are extremely compute- and data-intensive.

## Solutions & Current Best Practices
There is no settled approach — this is open research. Pragmatically: **action-conditioned generative models** (Genie-style) for training embodied agents in simulation; **probing + interventions** to assess what world structure a model encodes; **JEPA-style abstract prediction** as a non-generative alternative under active development; and **model-based RL** (Dreamer/MuZero lineage) where explicit world models already deliver planning gains in games/robotics. For LLMs, the practical stance is to *use* their partial world knowledge while grounding via tools/retrieval/perception.

## Lab Perspectives
- **OpenAI:** Sora as "world simulator"; bets that scaling generative video yields world models.
- **Google DeepMind:** **Genie** (interactive world generation), Dreamer/MuZero (model-based RL), Veo; deep model-based-planning heritage; explicit "world models" research org.
- **Meta (LeCun/FAIR):** **JEPA** (I-JEPA, V-JEPA) — the prominent *anti-generative* world-model program; LeCun is the loudest skeptic of LLMs-as-AGI.
- **Robotics/embodied labs:** world models for sim-to-real and planning (RT-2, robotics foundation models).

## Latest Developments (2023–2026)
**Interactive world models** advanced rapidly (Genie 2/3, generated playable environments) as a substrate for training agents. **Video generation** (Sora, Veo) reignited the "emergent physics" debate. **V-JEPA 2** and abstract-prediction approaches matured as LeCun's alternative. Growing interest in **world models for agents/robotics** (planning, sim-to-real) and in rigorously testing whether scaled generative models learn causal structure or sophisticated interpolation.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Are LLMs world models?"** Partial/fragmentary internal representations (Othello-GPT, entity tracking) but not robust/causal; nuance is expected.
- **"Is Sora a world model?"** Learns appearance + some emergent structure, but violates physics → debate between emergence and statistics.
- **"What is JEPA and why does LeCun prefer it?"** Predict in abstract embedding space (not pixels/tokens); discard unpredictable detail; argued path to grounded intelligence beyond autoregression.
- **"How would you test if a model has a world model?"** Probing latent state, causal interventions, counterfactual/physics consistency tests.

## Open Problems
Whether generative (pixel/token) prediction can yield **robust, causal** world models — or whether **abstract-prediction (JEPA)** / explicit model-based approaches are needed — is unresolved and foundational. Operationalizing "has a world model," achieving reliable **physical reasoning and long-horizon planning**, and grounding language models in genuine world structure are central open questions tied to the road to AGI.

## References
- Ha, D., Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122.
- OpenAI (2024). *Video Generation Models as World Simulators (Sora).*
- Bruce, J. et al. (2024). *Genie.* arXiv:2402.15391.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence (JEPA).*
- Li, K. et al. (2022). *Othello-GPT / Emergent World Representations.* arXiv:2210.13382.
