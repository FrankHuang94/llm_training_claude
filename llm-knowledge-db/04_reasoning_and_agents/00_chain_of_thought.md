# Chain-of-Thought Reasoning

> **Last Updated:** 2026-06-18
> **Related Files:** [Reasoning Models](01_reasoning_models_o1_r1.md) · [Inference-Time Scaling](02_inference_time_scaling.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md)
> **Key Papers:** Wei et al. 2022 Chain-of-Thought ([arXiv:2201.11903](https://arxiv.org/abs/2201.11903)) · Kojima et al. 2022 Zero-shot CoT ([arXiv:2205.11916](https://arxiv.org/abs/2205.11916)) · Wang et al. 2022 Self-Consistency ([arXiv:2203.11171](https://arxiv.org/abs/2203.11171)) · Yao et al. 2023 Tree of Thoughts ([arXiv:2305.10601](https://arxiv.org/abs/2305.10601))

## Overview
Chain-of-thought (CoT) prompting — eliciting intermediate reasoning steps before a final answer — is the discovery that reframed LLMs from pattern-matchers into apparent reasoners. Wei et al. (2022) showed that simply prompting a sufficiently large model with worked examples that *show their work* dramatically improves performance on multi-step arithmetic, commonsense, and symbolic reasoning, with the effect **emerging** only past a certain scale. CoT is the conceptual foundation of the entire 2024–2026 reasoning-model wave: o1/R1-style models are, in essence, models trained (via RL) to produce very long, high-quality chains of thought. Understanding *why* CoT works and its variants (self-consistency, tree/graph of thought) is essential modern-LLM knowledge.

## Core Concepts
**Few-shot CoT.** Provide exemplars in the prompt where the answer is preceded by step-by-step reasoning. The model imitates the format, decomposing the problem and "thinking out loud," which raises accuracy on tasks requiring sequential computation.

**Zero-shot CoT.** Kojima et al. (2022) found that simply appending **"Let's think step by step"** triggers reasoning without any exemplars — a striking demonstration that the capability is latent and just needs eliciting. A second prompt then extracts the final answer.

**Why CoT helps (mechanisms).**
- **Computation depth / serial reasoning:** a Transformer does a fixed amount of computation per token; CoT lets it *externalize* intermediate state into generated tokens, effectively performing more serial steps (each token is another "compute step" conditioned on prior work). Theory (Feng et al., Merrill & Sabharwal) shows CoT *provably* increases the expressive/computational power of fixed-depth transformers.
- **Decomposition:** breaking a problem into subproblems reduces per-step difficulty.
- **Conditioning on its own correct partial work** raises the probability of a correct final token.

**Self-consistency.** Instead of greedy decoding, **sample multiple diverse CoT paths and majority-vote** the final answers (marginalizing over reasoning paths). This robustly boosts accuracy — a simple, powerful inference-time technique (see [Inference-Time Scaling](02_inference_time_scaling.md)).

**Structured extensions.**
- **Least-to-most prompting:** explicitly decompose into ordered subproblems, solve sequentially, feeding earlier answers forward — better compositional generalization.
- **Tree of Thoughts (ToT):** explore a *tree* of partial reasoning states with lookahead, backtracking, and value-based search — good for problems needing exploration (puzzles, planning).
- **Graph of Thoughts (GoT):** generalize to a graph, allowing merging/refinement of thoughts.

## Key Challenges
- **Scale dependence.** CoT only helps (and can hurt) below a capability threshold; small models gain little.
- **Faithfulness.** The stated reasoning may not reflect the model's actual computation — it can reach the right answer for stated-but-wrong reasons, or post-hoc rationalize (a safety concern; see [Interpretability](../05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md)).
- **Error propagation.** A mistake early in the chain derails the rest; longer chains can accumulate errors.
- **Cost.** Generating long reasoning multiplies tokens/latency; ToT/GoT multiply it further.
- **Overthinking.** On easy tasks, forced CoT can *reduce* accuracy or waste compute.

## Solutions & Current Best Practices
**Self-consistency** (sample-and-vote) is the default accuracy booster when compute allows. For hard search problems, **ToT/MCTS-style search** with a value/verifier. The modern frontier *internalizes* CoT via **RL training** (RLVR/GRPO) so the model produces long, self-correcting chains natively — rather than relying on prompting — and uses **verifiers** to score/select chains. Adaptive/"think-when-needed" approaches curb overthinking on easy inputs.

## Lab Perspectives
- **Google** discovered CoT, self-consistency, and least-to-most (Brain/DeepMind), and ToT.
- **OpenAI** turned CoT into a *trained* capability (o1/o3: long internal chains via RL), and showed CoT-as-monitorable-reasoning for safety.
- **DeepSeek** (R1) demonstrated long CoT emerging from pure RL.
- **Anthropic** studies CoT **faithfulness** (does the chain reflect the true reason?) as a safety/interpretability question and uses extended "thinking" in Claude.

## Latest Developments (2023–2026)
The big shift: from **prompted** CoT to **trained, internal** long CoT (o1/R1) optimized by RL with verifiable rewards, with test-time accuracy scaling in reasoning length (see [Inference-Time Scaling](02_inference_time_scaling.md)). Active research on **faithful/monitorable CoT** (keeping the chain a true window into the model's reasoning, important for oversight), **latent/continuous CoT** (reasoning in hidden space, e.g., Coconut), and curbing **overthinking**. Debate over whether long CoT is genuine reasoning or sophisticated pattern-matching.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does chain-of-thought improve performance?"** Externalizes intermediate computation (more serial steps), decomposition, conditioning on correct partial work; provable expressivity gains.
- **"What is self-consistency and why does it work?"** Sample multiple chains, majority-vote — marginalizes over reasoning paths, cancels independent errors.
- **"Is the reasoning faithful?"** Not necessarily; models can rationalize — a key safety caveat.
- **"How do o1/R1 relate to CoT?"** They train (via RL) to produce long, self-correcting CoT natively, scaling test-time compute.

## Open Problems
Whether CoT reflects genuine reasoning vs amplified pattern-matching is contested. **Faithfulness** — ensuring the chain is a truthful account of the computation — is unsolved and safety-critical. The right way to allocate reasoning compute adaptively, reason in latent space, and guarantee robustness to error propagation remain open.

## References
- Wei, J. et al. (2022). *Chain-of-Thought Prompting.* arXiv:2201.11903.
- Kojima, T. et al. (2022). *Large LMs are Zero-Shot Reasoners.* arXiv:2205.11916.
- Wang, X. et al. (2022). *Self-Consistency.* arXiv:2203.11171.
- Yao, S. et al. (2023). *Tree of Thoughts.* arXiv:2305.10601.
- Feng, G. et al. (2023). *Towards Revealing the Mystery behind CoT.* arXiv:2305.15408.
