# 最新动态速览：2025 年年中 → 2026 年 11 月（What's New）

> **最后更新：** 2026-11-04
> **用途：** 用一页纸概述 2025 年年中至 2026 年 11 月间大语言模型训练与前沿实验室格局的变化。如果你读过本知识库的旧版本，请先读本文；每个主题都链接到深入讨论它的主题文件。
> **来源说明：** 所有事实均已于 2026-11-04 对照实验室公告、系统卡（system card）、arXiv 论文及可靠媒体进行核实。厂商自报的基准数字均已注明；来源较弱的条目已作审慎表述（"据报道"）或予以省略。面试中引用具体日期和数字前，请对照原始来源再次核实。

---

## 1. 前沿模型格局（截至 2026 年 11 月）

| 实验室 | 当前旗舰模型 | 要点 |
|---|---|---|
| **OpenAI** | GPT-6 Astra（2026-09）、GPT-6 Sol / Luna（2026-09-22）、GPT-6.1 Sol（DevDay，2026-09-29） | GPT-5（2025-08）→ 5.1 → 5.2 → 5.3-Codex → 5.4（原生计算机操作，约 100 万上下文）→ 5.5 → 5.6 Sol/Terra/Luna → GPT-6。Astra 是 OpenAI 首个在 Preparedness Framework 下被评为网络安全 **"Critical（严重）"** 级的模型，首先通过 **Daybreak** 可信访问计划提供。gpt-oss（2025-08）是 OpenAI 自 GPT-2 以来首个开放权重的大语言模型。 |
| **Anthropic** | Claude Opus 5.5 / Sonnet 5.5 / Haiku 5.5（2026 年 9–10 月）；Claude Fable 5.1 / Mythos 5.1（2026-09） | 2026 年初之前为 Claude 4.x（Opus 4.6：100 万上下文、自适应思考）。**Claude Mythos Preview**（2026-04）因网络攻击能力而未全面发布（Project Glasswing）。自 2026 年 6 月起，**同一套权重以两种防护等级发布**：Fable（全面开放，带分类器路由）与 Mythos（需验证的受限访问）。RSP v3（2026-02）新增定期风险报告与公开的前沿安全路线图。 |
| **Google DeepMind** | Gemini 3.1 Pro（2026-02）、Gemini 3.5 Flash（2026 I/O）、Gemini 3.6 Flash（2026-07） | Gemini 3 Pro + Deep Think（2025-11）。Gemini 3.5 Pro 已宣布但推迟；据报道 Gemini 4 于 2026 年 9 月下旬进入后训练。Genie 3 世界模型（2025-08）。受限发布的 **Gemini 3.5 Flash Cyber**。FSF v3.x 新增"跟踪能力等级"（Tracked Capability Levels）。 |
| **Meta（MSL）** | Muse Spark（2026-04）、Muse Wonder（2026-10） | Llama 4 之后，Meta 的前沿产品线转向**闭源权重**。 |
| **DeepSeek** | V4-Pro（总参数 1.6T / 激活 49B）与 V4-Flash（284B / 13B），2026-04-24；V4.1-Flash（2026-09） | V3.1（UE8M0 FP8）→ V3.2 引入 **DeepSeek 稀疏注意力（DSA）**→ V4 采用压缩稀疏注意力、Muon、专家层 FP4 量化感知训练、100 万上下文。MIT 许可证。 |
| **Qwen（阿里巴巴）** | Qwen3.8（开放权重，总参数 2.4T / 激活约 95B，2026-08） | Qwen3-Next 引入 Gated DeltaNet 混合架构，Qwen3.5/3.6 延续之。 |
| **月之暗面（Kimi）** | Kimi K3（总参数 2.8T / 激活约 104B，2026-07）；K3.1 Pro（2026-10） | K2（1T，2025-07）使用 **MuonClip** 训练，全程无损失尖峰；K3 以 3:1 比例混合 Kimi Delta Attention 与 MLA。 |
| **Mistral** | Mistral Large 4（约 1.05T / 约 52B 激活，2026-10 开放权重） | Large 3（675B/41B，Apache 2.0，2025-12）；Medium 3.5（128B 稠密，开放权重）。 |

各实验室详情见[实验室路线图](08_lab_roadmaps/06_lab_comparison_matrix.md)。

## 2. 对训练最重要的十大主题

1. **混合注意力与稀疏注意力进入量产。** 有两种配方。一种以约 3:1 的比例交错线性/delta 规则层与全注意力层（Qwen3-Next、Kimi Linear/K3）。另一种在压缩后的 KV 缓存上做可学习的 top-k 稀疏注意力（DeepSeek DSA → V4，GLM-5 亦采用）。MiniMax M2 以多跳推理失败为由回归全注意力，这是标准的反例。→ [注意力机制](00_foundations/01_attention_mechanisms.md)、[扩散与混合语言模型](09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)
2. **开放权重 MoE 迈入万亿级。** 约 15 个月内，开放模型的总参数从 1T（Kimi K2）增至 2.8T（Kimi K3）。激活参数保持在约 13–104B，因此稀疏度持续上升（例如 896 个专家中激活 16 个）。→ [混合专家](06_efficiency/05_mixture_of_experts.md)
3. **精度降至 4 比特。** FP8 已成基线。NVFP4 预训练在 10T+ token 上与 FP8 持平（NVIDIA），Nemotron 3 以 NVFP4 预训练，DeepSeek V4 对专家层采用 FP4 量化感知训练。硬件开始以 FP4 FLOPs 标称算力。→ [混合精度](02_pretraining/04_mixed_precision_and_quantization.md)、[量化技术](06_efficiency/02_quantization_techniques.md)
4. **Muon 系优化器进入主流，但需客观看待。** Muon/MuonClip（QK-clip）已用于万亿级训练。调参公平的基准测试显示，其相对 AdamW 的加速上限约 1.4×，且随规模增大而缩小。→ [优化器](02_pretraining/03_optimizer_choices.md)
5. **强化学习缩放成为一门科学。** 几项进展：
   - 强化学习的计算曲线呈 S 形，可由早期训练预测（ScaleRL）。
   - 目标函数趋于收敛：token 级归一化、解耦裁剪（DAPO）、序列级比率（GSPO）以及重要性权重裁剪（CISPO）。
   - 训练与推断之间的数值不一致如今被视为算法问题。
   - 异步强化学习基础设施已成标配。

   → [GRPO 与 RLVR](03_posttraining/06_GRPO_and_RLVR.md)
6. **同策略蒸馏介于 SFT 与强化学习之间。** 学生模型自行采样、教师模型逐 token 打分，以远低于强化学习的成本获得稠密奖励。LoRA 在强化学习中即便秩极低也有效。→ [知识蒸馏](06_efficiency/03_knowledge_distillation.md)、[LoRA](06_efficiency/01_LoRA_and_variants.md)
7. **智能体是能力的主轴。** 这一转变的迹象：
   - 长时程编码与计算机操作智能体（长达数小时的自主会话、并行子智能体、上下文压缩）。
   - "努力程度"（effort）与自适应思考作为产品控制项开放。
   - 自 2023 年以来，METR 50% 时间跨度约每 4 个月翻一番。
   - 互操作标准：MCP、A2A、Agent Skills。

   → [智能体编码与计算机操作](04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)
8. **基准饱和速度前所未有。** OpenAI 弃用 SWE-bench Verified（测试有缺陷且存在污染），改用 SWE-bench Pro。HLE 在使用工具时已超过 60%。交互式基准 ARC-AGI-3 发布时前沿模型得分低于 1%。经济类 Elo 评测（GDPval-AA）与**评测感知**（模型察觉自己正在被测试）成为核心议题。→ [评测与基准](05_alignment_and_safety/05_evaluation_and_benchmarks.md)
9. **安全进入训练与发布流水线。** 几项变化：
   - 生产环境强化学习中的奖励投机可泛化为广泛的失对齐，**接种提示**（inoculation prompting）是一种低成本缓解手段。
   - 思维链可监控性被视为一种脆弱的安全属性。
   - **能力塑形**已是有意为之：Anthropic 未在网络安全任务上训练 Opus 5。
   - 同一套权重配合分类器路由的**分级发布**（Fable/Mythos、Daybreak、Flash Cyber）。
   - 安全框架从阈值承诺转向定期公开报告。

   → [前沿安全框架与治理](05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md)
10. **算力以吉瓦为单位、跨厂商采购，政府开始介入。** 例如：
    - OpenAI：在 NVIDIA、AMD 与 Broadcom 之间合计约 26 GW。
    - Anthropic：TPU（最多 100 万颗芯片）、Trainium 与 NVIDIA 并用。
    - 定制芯片开始分化为训练芯片与推断芯片（TPU 8t/8i、Rubin CPX）。
    - 美国出口管制曾短暂导致一款前沿模型下线（2026 年 6 月）。
    - 欧盟《人工智能法》的通用目的 AI（GPAI）义务已经适用，加州 SB 53 已经生效。

    → [分布式训练](02_pretraining/01_distributed_training.md)、[实验室对比](08_lab_roadmaps/06_lab_comparison_matrix.md)

## 3. 本次更新新增的文件
- [`04_reasoning_and_agents/06_agentic_coding_and_computer_use.md`](04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)
- [`05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md`](05_alignment_and_safety/06_frontier_safety_frameworks_and_governance.md)
- [`09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md`](09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)

## 4. 一小时快速复习清单
1. 推导门控 delta 规则的更新式，并解释混合架构为何优于纯线性注意力（[注意力机制](00_foundations/01_attention_mechanisms.md)）。
2. 比较 GRPO → DAPO → GSPO → CISPO，并解释训练与推断的数值不一致如何破坏"同策略"（[GRPO 与 RLVR](03_posttraining/06_GRPO_and_RLVR.md)）。
3. 解释 NVFP4 与 MXFP4 的区别，以及让 FP4 预训练保持稳定的技巧（[混合精度](02_pretraining/04_mixed_precision_and_quantization.md)）。
4. 解释为何奖励投机会导致涌现式失对齐，以及接种提示为何有效（[RLHF](03_posttraining/03_RLHF.md)、[对齐概述](05_alignment_and_safety/00_alignment_overview.md)）。
5. 解释 METR 时间跨度、评测感知，以及 SWE-bench Verified 为何被弃用（[评测与基准](05_alignment_and_safety/05_evaluation_and_benchmarks.md)）。
6. 熟悉目标实验室 2026 年的定位（[实验室对比矩阵](08_lab_roadmaps/06_lab_comparison_matrix.md)）。
