# 大语言模型训练知识库（LLM Training Knowledge Database）

> **最后更新：** 2026-11-04
> **用途：** 一套结构化、博士水平的研究参考与技术面试准备资料，覆盖大语言模型训练的完整生命周期——基础、数据、预训练、后训练、推理、对齐、效率、多模态、实验室战略与新兴前沿。
> **目标读者：** 以 OpenAI、Anthropic、Google DeepMind、Meta AI、DeepSeek、Mistral、xAI 等机构研究/工程岗位为目标的计算机科学五年级博士生。
> **译本说明：** 本目录为英文原版 [`../llm-knowledge-db/`](../llm-knowledge-db/) 的简体中文参考译本，仅供参考；如有歧义，以英文原版为准。术语译法见 [`GLOSSARY.md`](GLOSSARY.md)。

每个主题文件都遵循统一模板：**概述 → 核心概念 → 关键挑战 → 解决方案与当前最佳实践 → 各实验室观点 → 最新进展（2023–2026 年 11 月）→ 面试要点 → 开放问题 → 参考文献。**

> 🆕 **如果你读过旧版本，请从这里开始：** [`WHATS_NEW_2026.md`](WHATS_NEW_2026.md)——一页纸速览 2025 年年中至 2026 年 11 月间的变化（前沿模型、混合/稀疏注意力、万亿参数级开放权重 MoE、FP4 训练、强化学习缩放、智能体、评测、安全框架），并附有指向各主题文件的链接。
>
> 📄 **可打印 PDF：** [`LLM训练知识库_简体中文版.pdf`](LLM训练知识库_简体中文版.pdf)（A4，含目录、页码与书签）。

---

## 00_foundations — Transformer 与缩放基础
- [`00_transformer_architecture.md`](00_foundations/00_transformer_architecture.md) — Vaswani 2017，仅解码器与编码器-解码器架构，自注意力数学，现代变体。
- [`01_attention_mechanisms.md`](00_foundations/01_attention_mechanisms.md) — FlashAttention 1/2/3、MQA、GQA、MLA、滑动窗口、环形注意力、KV 缓存、门控注意力、稀疏注意力（DSA）与混合线性注意力。
- [`02_tokenization.md`](00_foundations/02_tokenization.md) — BPE、WordPiece、SentencePiece、Unigram、字节级分词、分词膨胀率（fertility）、分词瓶颈之争。
- [`03_positional_encoding.md`](00_foundations/03_positional_encoding.md) — 正弦、可学习、RoPE、ALiBi、YaRN、LongRoPE。
- [`04_scaling_laws.md`](00_foundations/04_scaling_laws.md) — Kaplan 与 Chinchilla 之争、计算最优训练、涌现之争。
- [`05_model_architecture_variants.md`](00_foundations/05_model_architecture_variants.md) — GPT/PaLM/LLaMA/Mistral/DeepSeek/Qwen/Kimi/gpt-oss 的设计选择，归一化/激活函数/MoE，2026 年万亿级混合架构。
- [`06_information_theory_basics.md`](00_foundations/06_information_theory_basics.md) — 熵、交叉熵、KL 散度、困惑度、每字节比特数、最小描述长度（MDL）。

## 01_data — 数据整理与合成
- [`00_data_collection_and_curation.md`](01_data/00_data_collection_and_curation.md) — Common Crawl、C4、The Pile、RedPajama、DCLM、FineWeb；法律与伦理。
- [`01_data_quality_and_deduplication.md`](01_data/01_data_quality_and_deduplication.md) — MinHash LSH、精确与模糊去重、基于困惑度/分类器的过滤。
- [`02_tokenizer_training.md`](01_data/02_tokenizer_training.md) — 训练 BPE/Unigram 分词器、词表大小、多语言、领域覆盖。
- [`03_data_mixing_and_curriculum.md`](01_data/03_data_mixing_and_curriculum.md) — 领域权重、DoReMi、课程学习、退火。
- [`04_synthetic_data_generation.md`](01_data/04_synthetic_data_generation.md) — Self-Instruct、Alpaca、WizardLM、Orca、phi 系列、蒸馏、模型崩溃。
- [`05_data_contamination_and_evaluation_leakage.md`](01_data/05_data_contamination_and_evaluation_leakage.md) — n-gram/嵌入检测、金丝雀字符串、基准泄漏。

## 02_pretraining — 大规模分布式训练
- [`00_pretraining_objectives.md`](02_pretraining/00_pretraining_objectives.md) — CLM、MLM、前缀语言模型、FIM、UL2、片段破坏（span corruption）。
- [`01_distributed_training.md`](02_pretraining/01_distributed_training.md) — DP/TP/PP/SP、ZeRO、FSDP、Megatron、3D 并行。
- [`02_training_stability.md`](02_pretraining/02_training_stability.md) — 损失尖峰、梯度裁剪、预热、初始化、muP、z-loss、qk-norm。
- [`03_optimizer_choices.md`](02_pretraining/03_optimizer_choices.md) — AdamW、Adafactor、LION、Sophia、Muon/MuonClip/Dion；学习率调度（余弦、WSD）。
- [`04_mixed_precision_and_quantization.md`](02_pretraining/04_mixed_precision_and_quantization.md) — fp16/bf16/fp8、MXFP4/NVFP4 预训练、损失缩放、Transformer Engine。
- [`05_compute_and_memory_efficiency.md`](02_pretraining/05_compute_and_memory_efficiency.md) — 激活检查点、卸载、MFU、6N 法则。
- [`06_checkpoint_and_resumption.md`](02_pretraining/06_checkpoint_and_resumption.md) — 分片检查点、异步保存、容错。

## 03_posttraining — 对齐与偏好优化
- [`00_posttraining_overview.md`](03_posttraining/00_posttraining_overview.md) — SFT→RM→RL 流水线；现代后训练技术栈。
- [`01_supervised_finetuning_SFT.md`](03_posttraining/01_supervised_finetuning_SFT.md) — 对话模板、损失掩码、序列打包、多轮对话。
- [`02_instruction_tuning.md`](03_posttraining/02_instruction_tuning.md) — FLAN、T0、OpenAssistant、ShareGPT；向未见任务的泛化。
- [`03_RLHF.md`](03_posttraining/03_RLHF.md) — InstructGPT 流水线、PPO、KL 惩罚、奖励投机。
- [`04_reward_modeling.md`](03_posttraining/04_reward_modeling.md) — Bradley-Terry、过度优化、PRM 与 ORM、奖励模型集成。
- [`05_DPO_and_preference_optimization.md`](03_posttraining/05_DPO_and_preference_optimization.md) — DPO 推导、IPO、KTO、SimPO、ORPO。
- [`06_GRPO_and_RLVR.md`](03_posttraining/06_GRPO_and_RLVR.md) — DeepSeek GRPO、DAPO/GSPO/CISPO、可验证奖励、R1-Zero、强化学习缩放定律、异步强化学习。
- [`07_constitutional_AI_and_RLAIF.md`](03_posttraining/07_constitutional_AI_and_RLAIF.md) — Anthropic 宪法式 AI（CAI）、AI 反馈、可扩展监督。
- [`08_rejection_sampling_and_best_of_N.md`](03_posttraining/08_rejection_sampling_and_best_of_N.md) — best-of-N、RAFT、STaR、迭代蒸馏。

## 04_reasoning_and_agents — 推理与工具使用
- [`00_chain_of_thought.md`](04_reasoning_and_agents/00_chain_of_thought.md) — CoT、零样本 CoT、由简至繁（least-to-most）、思维树（ToT）、思维图（GoT）。
- [`01_reasoning_models_o1_r1.md`](04_reasoning_and_agents/01_reasoning_models_o1_r1.md) — o1/o3、DeepSeek-R1、思考 token、PRM/ORM。
- [`02_inference_time_scaling.md`](04_reasoning_and_agents/02_inference_time_scaling.md) — best-of-N、自洽性、验证器引导、计算最优的测试时计算。
- [`03_tool_use_and_function_calling.md`](04_reasoning_and_agents/03_tool_use_and_function_calling.md) — ReAct、函数调用、代码解释器、MCP。
- [`04_agentic_frameworks.md`](04_reasoning_and_agents/04_agentic_frameworks.md) — LangChain、AutoGen、CrewAI、智能体循环、记忆。
- [`05_multi_agent_systems.md`](04_reasoning_and_agents/05_multi_agent_systems.md) — 多智能体辩论、智能体混合（mixture-of-agents）、编排。
- [`06_agentic_coding_and_computer_use.md`](04_reasoning_and_agents/06_agentic_coding_and_computer_use.md) — 🆕 长时程编码智能体、计算机操作、智能体强化学习环境、子智能体、METR 时间跨度。

## 05_alignment_and_safety — 安全、可解释性与评测
- [`00_alignment_overview.md`](05_alignment_and_safety/00_alignment_overview.md) — 外部/内部对齐、规范博弈（specification gaming）、对齐问题。
- [`01_hallucination_causes_and_mitigations.md`](05_alignment_and_safety/01_hallucination_causes_and_mitigations.md) — 内在/外在幻觉、成因、RAG、事实性微调。
- [`02_jailbreaks_and_red_teaming.md`](05_alignment_and_safety/02_jailbreaks_and_red_teaming.md) — 提示注入、对抗性后缀、自动化红队测试。
- [`03_interpretability_and_mechanistic_analysis.md`](05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md) — 叠加、SAE、回路、归纳头。
- [`04_scalable_oversight.md`](05_alignment_and_safety/04_scalable_oversight.md) — 辩论、递归奖励建模、弱到强泛化、过程监督。
- [`05_evaluation_and_benchmarks.md`](05_alignment_and_safety/05_evaluation_and_benchmarks.md) — MMLU、GPQA、HLE、SWE-bench Pro、ARC-AGI-3、GDPval、METR、Arena、污染、评测感知。
- [`06_frontier_safety_frameworks_and_governance.md`](05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md) — 🆕 RSP / Preparedness / FSF、分级与受限发布、欧盟《人工智能法》、SB 53、版权判决。

## 06_efficiency — 参数效率与推断效率
- [`00_parameter_efficient_finetuning.md`](06_efficiency/00_parameter_efficient_finetuning.md) — 适配器、前缀/提示微调、(IA)³、BitFit。
- [`01_LoRA_and_variants.md`](06_efficiency/01_LoRA_and_variants.md) — LoRA、QLoRA、DoRA、LoRA+、LoftQ、秩的选择。
- [`02_quantization_techniques.md`](06_efficiency/02_quantization_techniques.md) — GPTQ、AWQ、SmoothQuant、fp8/int8/int4、训练后量化（PTQ）与量化感知训练（QAT）。
- [`03_knowledge_distillation.md`](06_efficiency/03_knowledge_distillation.md) — Hinton 知识蒸馏、序列级蒸馏、DistilBERT、同策略蒸馏。
- [`04_speculative_decoding.md`](06_efficiency/04_speculative_decoding.md) — 草稿+验证、Medusa、EAGLE、前瞻解码（lookahead）。
- [`05_mixture_of_experts.md`](06_efficiency/05_mixture_of_experts.md) — Switch、GLaM、Mixtral、DeepSeek-MoE、负载均衡。
- [`06_context_window_extension.md`](06_efficiency/06_context_window_extension.md) — 位置插值、YaRN、环形注意力、RAG。

## 07_multimodal — 视觉、音频及更多模态
- [`00_vision_language_models.md`](07_multimodal/00_vision_language_models.md) — CLIP、LLaVA、GPT-4V、Gemini、连接器。
- [`01_multimodal_pretraining.md`](07_multimodal/01_multimodal_pretraining.md) — LAION、ALIGN、原生多模态预训练。
- [`02_cross_modal_alignment.md`](07_multimodal/02_cross_modal_alignment.md) — 对比对齐、投影与交叉注意力、模态鸿沟。
- [`03_audio_video_and_beyond.md`](07_multimodal/03_audio_video_and_beyond.md) — Whisper、AudioPaLM、视频-语言模型、任意到任意（any-to-any）模型。

## 08_lab_roadmaps — 前沿实验室战略
- [`00_openai_roadmap.md`](08_lab_roadmaps/00_openai_roadmap.md)
- [`01_anthropic_roadmap.md`](08_lab_roadmaps/01_anthropic_roadmap.md)
- [`02_google_deepmind_roadmap.md`](08_lab_roadmaps/02_google_deepmind_roadmap.md)
- [`03_meta_ai_roadmap.md`](08_lab_roadmaps/03_meta_ai_roadmap.md)
- [`04_deepseek_roadmap.md`](08_lab_roadmaps/04_deepseek_roadmap.md)
- [`05_mistral_and_others.md`](08_lab_roadmaps/05_mistral_and_others.md) — Mistral、Qwen、Kimi、GLM、MiniMax、xAI、NVIDIA、AI2 等
- [`06_lab_comparison_matrix.md`](08_lab_roadmaps/06_lab_comparison_matrix.md)

## 09_emerging_frontiers — 研究前沿
- [`00_long_context_and_memory.md`](09_emerging_frontiers/00_long_context_and_memory.md) — 100 万至 1000 万 token、MemGPT、Titans、"迷失在中间"（lost-in-the-middle）。
- [`01_continual_learning.md`](09_emerging_frontiers/01_continual_learning.md) — 灾难性遗忘、EWC、回放、可塑性与稳定性之争。
- [`02_world_models.md`](09_emerging_frontiers/02_world_models.md) — Sora、Genie、JEPA、世界模型之争。
- [`03_test_time_training.md`](09_emerging_frontiers/03_test_time_training.md) — TTT 层、在线自适应。
- [`04_neuromorphic_and_non_transformer.md`](09_emerging_frontiers/04_neuromorphic_and_non_transformer.md) — Mamba、RWKV、Hyena、RetNet、状态空间模型。
- [`05_frontier_open_problems.md`](09_emerging_frontiers/05_frontier_open_problems.md) — 推理、组合性、样本效率、AGI 规划。
- [`06_diffusion_and_hybrid_language_models.md`](09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md) — 🆕 扩散语言模型（LLaDA、Gemini Diffusion、Mercury）、混合线性/稀疏注意力的量产模型、多 token 预测（MTP）。

## 10_interview_prep — 面试准备
- [`00_system_design_questions.md`](10_interview_prep/00_system_design_questions.md) — 训练基础设施、RLHF 流水线、评测设计。
- [`01_common_technical_questions.md`](10_interview_prep/01_common_technical_questions.md) — PPO 中的 KL、DPO 与 RLHF、GQA、损失尖峰、Chinchilla。
- [`02_paper_discussion_questions.md`](10_interview_prep/02_paper_discussion_questions.md) — 必读论文与深入讨论准备。
- [`03_coding_exercises_for_LLM_engineers.md`](10_interview_prep/03_coding_exercises_for_LLM_engineers.md) — 注意力、BPE、PPO 循环、LoRA 注入。
- [`04_behavioral_and_research_narrative.md`](10_interview_prep/04_behavioral_and_research_narrative.md) — 介绍博士研究、针对不同实验室的表述。

---

## 面试准备推荐阅读顺序

0. **最新动态**（`WHATS_NEW_2026.md`）——如果你读过旧版本，请从这里开始。
1. **基础**（`00_foundations/`）——注意力数学与缩放定律几乎在每场面试中都会被考察。
2. **后训练**（`03_posttraining/`）——与面试相关性最高：RLHF、DPO、GRPO。
3. **推理与智能体**（`04_reasoning_and_agents/`）——2024–2026 年的前沿。
4. **效率**（`06_efficiency/`）——LoRA、量化、MoE 反复出现。
5. **预训练与数据**（`02_pretraining/`、`01_data/`）——系统设计的深度。
6. **对齐与安全**（`05_alignment_and_safety/`）——Anthropic/OpenAI 安全岗位的必备内容。
7. **实验室路线图**（`08_lab_roadmaps/`）——针对目标实验室调整回答。
8. **综合练习**（`10_interview_prep/`）——练习题与编程训练。

## 约定
- 数学公式采用 LaTeX 风格的行内 `$...$` 与独立 `$$...$$` 记法。
- 论文引用注明作者、年份，并尽可能给出 arXiv 编号。
- 内容反映截至 2026 年 11 月初的公开信息（2026-11-04 进行了大规模更新）。厂商自报的基准数字均已注明；来源较弱的条目已作审慎表述。
