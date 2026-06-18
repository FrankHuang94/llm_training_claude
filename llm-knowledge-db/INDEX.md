# LLM Training Knowledge Database

> **Last Updated:** 2026-06-18
> **Purpose:** A structured, PhD-level research reference and technical-interview preparation resource covering the full lifecycle of Large Language Model training — foundations, data, pretraining, post-training, reasoning, alignment, efficiency, multimodality, lab strategy, and emerging frontiers.
> **Audience:** A 5th-year CS PhD candidate targeting research/engineering roles at OpenAI, Anthropic, Google DeepMind, Meta AI, DeepSeek, Mistral, xAI, and peers.

Every topic file follows a consistent template: **Overview → Core Concepts → Key Challenges → Solutions & Current Best Practices → Lab Perspectives → Latest Developments (2023–2026) → Interview Angles → Open Problems → References.**

---

## 00_foundations — Transformer & Scaling Fundamentals
- [`00_transformer_architecture.md`](00_foundations/00_transformer_architecture.md) — Vaswani 2017, decoder-only vs encoder-decoder, self-attention math, modern variants.
- [`01_attention_mechanisms.md`](00_foundations/01_attention_mechanisms.md) — FlashAttention 1/2/3, MQA, GQA, sliding-window, ring attention, KV cache.
- [`02_tokenization.md`](00_foundations/02_tokenization.md) — BPE, WordPiece, SentencePiece, Unigram, byte-level, fertility, the tokenization-bottleneck debate.
- [`03_positional_encoding.md`](00_foundations/03_positional_encoding.md) — sinusoidal, learned, RoPE, ALiBi, YaRN, LongRoPE.
- [`04_scaling_laws.md`](00_foundations/04_scaling_laws.md) — Kaplan vs Chinchilla, compute-optimal training, emergence debate.
- [`05_model_architecture_variants.md`](00_foundations/05_model_architecture_variants.md) — GPT/PaLM/LLaMA/Mistral/Mixtral/DeepSeek design choices, norm/activation/MoE.
- [`06_information_theory_basics.md`](00_foundations/06_information_theory_basics.md) — entropy, cross-entropy, KL, perplexity, bits-per-byte, MDL.

## 01_data — Data Curation & Synthesis
- [`00_data_collection_and_curation.md`](01_data/00_data_collection_and_curation.md) — Common Crawl, C4, The Pile, RedPajama, DCLM, FineWeb; legal/ethics.
- [`01_data_quality_and_deduplication.md`](01_data/01_data_quality_and_deduplication.md) — MinHash LSH, exact vs fuzzy dedup, perplexity/classifier filtering.
- [`02_tokenizer_training.md`](01_data/02_tokenizer_training.md) — training BPE/Unigram, vocab size, multilingual, domain coverage.
- [`03_data_mixing_and_curriculum.md`](01_data/03_data_mixing_and_curriculum.md) — domain weights, DoReMi, curricula, annealing.
- [`04_synthetic_data_generation.md`](01_data/04_synthetic_data_generation.md) — Self-Instruct, Alpaca, WizardLM, Orca, phi-series, distillation, model collapse.
- [`05_data_contamination_and_evaluation_leakage.md`](01_data/05_data_contamination_and_evaluation_leakage.md) — n-gram/embedding detection, canary strings, benchmark leakage.

## 02_pretraining — Distributed Training at Scale
- [`00_pretraining_objectives.md`](02_pretraining/00_pretraining_objectives.md) — CLM, MLM, prefix-LM, FIM, UL2, span corruption.
- [`01_distributed_training.md`](02_pretraining/01_distributed_training.md) — DP/TP/PP/SP, ZeRO, FSDP, Megatron, 3D parallelism.
- [`02_training_stability.md`](02_pretraining/02_training_stability.md) — loss spikes, clipping, warmup, init, muP, z-loss, qk-norm.
- [`03_optimizer_choices.md`](02_pretraining/03_optimizer_choices.md) — AdamW, Adafactor, LION, Sophia, Muon; LR schedules (cosine, WSD).
- [`04_mixed_precision_and_quantization.md`](02_pretraining/04_mixed_precision_and_quantization.md) — fp16/bf16/fp8, loss scaling, Transformer Engine.
- [`05_compute_and_memory_efficiency.md`](02_pretraining/05_compute_and_memory_efficiency.md) — checkpointing, offloading, MFU, the 6N rule.
- [`06_checkpoint_and_resumption.md`](02_pretraining/06_checkpoint_and_resumption.md) — sharded checkpoints, async saving, fault tolerance.

## 03_posttraining — Alignment & Preference Optimization
- [`00_posttraining_overview.md`](03_posttraining/00_posttraining_overview.md) — the SFT→RM→RL pipeline; the modern post-training stack.
- [`01_supervised_finetuning_SFT.md`](03_posttraining/01_supervised_finetuning_SFT.md) — chat templates, loss masking, packing, multi-turn.
- [`02_instruction_tuning.md`](03_posttraining/02_instruction_tuning.md) — FLAN, T0, OpenAssistant, ShareGPT; generalization to unseen tasks.
- [`03_RLHF.md`](03_posttraining/03_RLHF.md) — InstructGPT pipeline, PPO, KL penalty, reward hacking.
- [`04_reward_modeling.md`](03_posttraining/04_reward_modeling.md) — Bradley-Terry, overoptimization, PRM vs ORM, RM ensembles.
- [`05_DPO_and_preference_optimization.md`](03_posttraining/05_DPO_and_preference_optimization.md) — DPO derivation, IPO, KTO, SimPO, ORPO.
- [`06_GRPO_and_RLVR.md`](03_posttraining/06_GRPO_and_RLVR.md) — DeepSeek GRPO, verifiable rewards, R1-Zero.
- [`07_constitutional_AI_and_RLAIF.md`](03_posttraining/07_constitutional_AI_and_RLAIF.md) — Anthropic CAI, AI feedback, scalable oversight.
- [`08_rejection_sampling_and_best_of_N.md`](03_posttraining/08_rejection_sampling_and_best_of_N.md) — best-of-N, RAFT, STaR, iterative distillation.

## 04_reasoning_and_agents — Reasoning & Tool Use
- [`00_chain_of_thought.md`](04_reasoning_and_agents/00_chain_of_thought.md) — CoT, zero-shot CoT, least-to-most, ToT, GoT.
- [`01_reasoning_models_o1_r1.md`](04_reasoning_and_agents/01_reasoning_models_o1_r1.md) — o1/o3, DeepSeek-R1, thinking tokens, PRM/ORM.
- [`02_inference_time_scaling.md`](04_reasoning_and_agents/02_inference_time_scaling.md) — best-of-N, self-consistency, verifier-guided, compute-optimal test-time.
- [`03_tool_use_and_function_calling.md`](04_reasoning_and_agents/03_tool_use_and_function_calling.md) — ReAct, function calling, code interpreters, MCP.
- [`04_agentic_frameworks.md`](04_reasoning_and_agents/04_agentic_frameworks.md) — LangChain, AutoGen, CrewAI, agent loops, memory.
- [`05_multi_agent_systems.md`](04_reasoning_and_agents/05_multi_agent_systems.md) — multi-agent debate, mixture-of-agents, orchestration.

## 05_alignment_and_safety — Safety, Interpretability, Evaluation
- [`00_alignment_overview.md`](05_alignment_and_safety/00_alignment_overview.md) — outer/inner alignment, specification gaming, the alignment problem.
- [`01_hallucination_causes_and_mitigations.md`](05_alignment_and_safety/01_hallucination_causes_and_mitigations.md) — intrinsic/extrinsic, causes, RAG, factuality tuning.
- [`02_jailbreaks_and_red_teaming.md`](05_alignment_and_safety/02_jailbreaks_and_red_teaming.md) — prompt injection, adversarial suffixes, automated red-teaming.
- [`03_interpretability_and_mechanistic_analysis.md`](05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md) — superposition, SAEs, circuits, induction heads.
- [`04_scalable_oversight.md`](05_alignment_and_safety/04_scalable_oversight.md) — debate, RRM, weak-to-strong, process supervision.
- [`05_evaluation_and_benchmarks.md`](05_alignment_and_safety/05_evaluation_and_benchmarks.md) — MMLU, GPQA, MATH, SWE-bench, Arena, contamination.

## 06_efficiency — Parameter & Inference Efficiency
- [`00_parameter_efficient_finetuning.md`](06_efficiency/00_parameter_efficient_finetuning.md) — adapters, prefix/prompt tuning, (IA)³, BitFit.
- [`01_LoRA_and_variants.md`](06_efficiency/01_LoRA_and_variants.md) — LoRA, QLoRA, DoRA, LoRA+, LoftQ, rank selection.
- [`02_quantization_techniques.md`](06_efficiency/02_quantization_techniques.md) — GPTQ, AWQ, SmoothQuant, fp8/int8/int4, PTQ vs QAT.
- [`03_knowledge_distillation.md`](06_efficiency/03_knowledge_distillation.md) — Hinton KD, sequence-level KD, DistilBERT, on-policy distillation.
- [`04_speculative_decoding.md`](06_efficiency/04_speculative_decoding.md) — draft+verify, Medusa, EAGLE, lookahead.
- [`05_mixture_of_experts.md`](06_efficiency/05_mixture_of_experts.md) — Switch, GLaM, Mixtral, DeepSeek-MoE, load balancing.
- [`06_context_window_extension.md`](06_efficiency/06_context_window_extension.md) — position interpolation, YaRN, ring attention, RAG.

## 07_multimodal — Vision, Audio & Beyond
- [`00_vision_language_models.md`](07_multimodal/00_vision_language_models.md) — CLIP, LLaVA, GPT-4V, Gemini, connectors.
- [`01_multimodal_pretraining.md`](07_multimodal/01_multimodal_pretraining.md) — LAION, ALIGN, native multimodal pretraining.
- [`02_cross_modal_alignment.md`](07_multimodal/02_cross_modal_alignment.md) — contrastive alignment, projection vs cross-attention, modality gap.
- [`03_audio_video_and_beyond.md`](07_multimodal/03_audio_video_and_beyond.md) — Whisper, AudioPaLM, video-language, any-to-any models.

## 08_lab_roadmaps — Frontier Lab Strategy
- [`00_openai_roadmap.md`](08_lab_roadmaps/00_openai_roadmap.md)
- [`01_anthropic_roadmap.md`](08_lab_roadmaps/01_anthropic_roadmap.md)
- [`02_google_deepmind_roadmap.md`](08_lab_roadmaps/02_google_deepmind_roadmap.md)
- [`03_meta_ai_roadmap.md`](08_lab_roadmaps/03_meta_ai_roadmap.md)
- [`04_deepseek_roadmap.md`](08_lab_roadmaps/04_deepseek_roadmap.md)
- [`05_mistral_and_others.md`](08_lab_roadmaps/05_mistral_and_others.md)
- [`06_lab_comparison_matrix.md`](08_lab_roadmaps/06_lab_comparison_matrix.md)

## 09_emerging_frontiers — Research Frontiers
- [`00_long_context_and_memory.md`](09_emerging_frontiers/00_long_context_and_memory.md) — 1M–10M tokens, MemGPT, Titans, lost-in-the-middle.
- [`01_continual_learning.md`](09_emerging_frontiers/01_continual_learning.md) — catastrophic forgetting, EWC, rehearsal, plastic-vs-stable.
- [`02_world_models.md`](09_emerging_frontiers/02_world_models.md) — Sora, Genie, JEPA, the world-model debate.
- [`03_test_time_training.md`](09_emerging_frontiers/03_test_time_training.md) — TTT layers, online adaptation.
- [`04_neuromorphic_and_non_transformer.md`](09_emerging_frontiers/04_neuromorphic_and_non_transformer.md) — Mamba, RWKV, Hyena, RetNet, SSMs.
- [`05_frontier_open_problems.md`](09_emerging_frontiers/05_frontier_open_problems.md) — reasoning, compositionality, sample efficiency, AGI planning.

## 10_interview_prep — Interview Preparation
- [`00_system_design_questions.md`](10_interview_prep/00_system_design_questions.md) — training infra, RLHF pipelines, eval design.
- [`01_common_technical_questions.md`](10_interview_prep/01_common_technical_questions.md) — PPO KL, DPO vs RLHF, GQA, loss spikes, Chinchilla.
- [`02_paper_discussion_questions.md`](10_interview_prep/02_paper_discussion_questions.md) — must-know papers and deep-dive prep.
- [`03_coding_exercises_for_LLM_engineers.md`](10_interview_prep/03_coding_exercises_for_LLM_engineers.md) — attention, BPE, PPO loop, LoRA injection.
- [`04_behavioral_and_research_narrative.md`](10_interview_prep/04_behavioral_and_research_narrative.md) — presenting PhD research, framing for labs.

---

## Recommended Reading Order for Interview Prep

1. **Foundations** (`00_foundations/`) — attention math and scaling laws are probed in nearly every interview.
2. **Post-training** (`03_posttraining/`) — highest interview relevance: RLHF, DPO, GRPO.
3. **Reasoning & agents** (`04_reasoning_and_agents/`) — the 2024–2026 frontier.
4. **Efficiency** (`06_efficiency/`) — LoRA, quantization, MoE recur constantly.
5. **Pretraining & data** (`02_pretraining/`, `01_data/`) — system-design depth.
6. **Alignment & safety** (`05_alignment_and_safety/`) — required for Anthropic/OpenAI safety roles.
7. **Lab roadmaps** (`08_lab_roadmaps/`) — tailor answers to your target lab.
8. **Capstone** (`10_interview_prep/`) — practice questions and coding drills.

## Conventions
- Math uses LaTeX-style inline `$...$` and block `$$...$$` notation.
- Papers cited with author, year, and arXiv ID where available.
- Content reflects public information through mid-2026.
