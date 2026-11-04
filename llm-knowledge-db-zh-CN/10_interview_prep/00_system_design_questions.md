# 系统设计问题（System Design Questions）

> **最后更新：** 2026-11-04
> **相关文件：** [分布式训练](../02_pretraining/01_distributed_training.md) · [RLHF](../03_posttraining/03_RLHF.md) · [评测与基准](../05_alignment_and_safety/05_evaluation_and_benchmarks.md) · [GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [扩散与混合语言模型](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)
> **关键论文：** Narayanan et al. 2021 (3D parallelism) · Ouyang et al. 2022 InstructGPT · Grattafiori et al. 2024 Llama 3 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) · Khatri et al. 2025 ScaleRL ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786))

## 概述
大语言模型（LLM）系统设计面试考察的是你能否**端到端地推演如何构建和运营大模型**——在计算、内存、数据和可靠性约束下做出并论证权衡。与单一事实类问题不同，这些问题是开放式的：面试官希望看到结构化思维、对瓶颈的认识、粗略估算（back-of-envelope）以及对真实系统做法的了解。本文件给出一个**框架**以及经典题目的参考解答提纲——包括面试官如今偏爱的 2025–2026 年新增题目：**智能体式（agentic）强化学习基础设施**、**以 1M 上下文服务万亿参数级混合专家（MoE）模型**，以及**面向两用能力的发布把关**。

## 通用框架（适用于任何题目）
1. **澄清需求与约束：** 模型规模、token 预算、硬件（数量/类型/互连）、延迟/吞吐量（throughput）目标、截止时间、预算。
2. **粗略估算：** FLOPs（$C\approx 6ND$）、内存（$\approx 16\Psi$ 字节的 Adam 状态 + 激活值）、KV 缓存、MFU 目标、墙钟时间估计。
3. **核心设计：** 并行策略、数据管线、优化器/调度、精度、检查点（checkpoint）。
4. **瓶颈与缓解：** 内存墙、通信、流水线气泡、掉队节点、训练不稳定。
5. **可靠性与监控：** 容错、损失/梯度监控、评测节奏。
6. **权衡与替代方案：** 说明在不同约束下你会如何调整。

## 参考解答提纲

### "设计 GPT-4 规模的训练基础设施" / "搭建一次 1000 GPU 的训练"
- **先算账：** 例如，70B 模型训练 15T token → $C\approx 6\cdot70\text{e}9\cdot15\text{e}12 \approx 6.3\text{e}24$ FLOPs。在 1000 张 H100（每张约 1e15 bf16 FLOP/s）上以约 40% MFU 运行 → 有效约 $4\text{e}17$ FLOP/s → 约 **180 天**。表明你能估算并随后*调整*（更多 GPU、更小模型、更少 token）。
- **并行：** **节点内 TP=8**（NVLink），**跨节点 PP**（交错式 1F1B 以缩小气泡），外层 **ZeRO/FSDP 数据并行**，长上下文用**序列并行**（见[分布式训练](../02_pretraining/01_distributed_training.md)）。用互连带宽来论证。
- **内存：** bf16 计算 + fp32 主权重，ZeRO 切分的优化器状态，**选择性梯度检查点**，FlashAttention。
- **数据：** 分片、去重、去污染、**可续训**的流式加载器；有文档记录的数据配比 + 退火阶段。
- **稳定性：** 预热（warmup）+ WSD/余弦调度、梯度裁剪、z-loss/QK-norm、用 μP 迁移超参数；监控梯度范数；遇到尖峰时**回滚 + 跳过批次**。
- **2026 年可提及的选项：** **Muon**（配合权重衰减、RMS 匹配的更新以及 **QK-Clip**——Kimi K2 训练了 15.5T token，零损失尖峰；相对调好的 AdamW 预期收益 ≲1.4×）、**FP8 → NVFP4** 预训练（对 Wgrad 输入做 Hadamard 变换、随机舍入、少数层保持 BF16；Nemotron 3 Super 训练了 25T token），在 Blackwell 级硬件上要用你实际训练所用的精度来报 FLOPs。
- **可靠性：** **异步分片检查点**（主机内存快照 + 后台落盘）、自动检测与重启、热备节点（见[检查点](../02_pretraining/06_checkpoint_and_resumption.md)）。
- **指标：** 优化 **MFU**；报告有效训练时间。

### "从零设计一条 RLHF 管线"
- **阶段：** 监督微调（SFT，精选/合成示范，带损失掩码）→ **奖励模型**（Bradley-Terry、成对偏好、长度去偏、集成）→ **PPO**（策略 + 参考 + 奖励模型 + 价值函数；KL 惩罚；裁剪）——或为简单/省钱选 **DPO/GRPO**（见[RLHF](../03_posttraining/03_RLHF.md)）。
- **基础设施：** PPO 需要**内存中同时放 4 个模型** + 为训练提供数据的**生成 rollout**（推断引擎，如 vLLM）——要设计 **actor/learner** 拆分、批处理和 KV 复用。框架：TRL/OpenRLHF/verl（最常用的开源框架）、slime（Megatron + SGLang）、AReaL。
- **失效模式：** 奖励投机（reward hacking）/过度优化 → 调 **β**、监控奖励-KL 曲线、迭代偏好数据；长度偏差 → 对奖励模型去偏。
- **数据飞轮：** 多轮迭代（用更新后的策略重新收集偏好）；用拒绝采样产生 SFT 数据。
- **评测：** 偏好胜率、安全/拒答、能力退化（对齐税）、Arena 式 + 留出集（见[评测](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)）。

### "设计一个高吞吐的推断/服务系统"
- **瓶颈：** 解码是**内存带宽受限**的；KV 缓存占据主要内存。
- **技术：** **连续批处理**、**PagedAttention**（vLLM）、**前缀缓存**、**量化**（W4A16/fp8 + KV 量化）、**投机解码**（EAGLE/Medusa）、**GQA/MLA**、大模型的张量/流水线并行、**预填充/解码分离**。
- **权衡：** 延迟（小批次、投机解码）vs 吞吐（大批次）；对推理模型要为**测试时计算**做预算（如今已是产品旋钮：effort 等级、自适应思考）。

### "为长时程智能体式编程设计 RL 后训练系统"（2026 年热门题）
- **环境是瓶颈：** 成千上万个容器化、*可验证*的代码仓库任务（由智能体合成的任务、测试框架随机化），在**不开放互联网**的沙箱中运行——2026 年的评测事件表明智能体会利用配置错误的网络访问。
- **异步架构：** 将 rollout 工作进程（SGLang/vLLM）与训练器（Megatron/FSDP）解耦；限制策略陈旧度（AReaL 的陈旧度感知 PPO；PipelineRL 的飞行中权重更新；slime）。长轨迹会让同步 RL 的大部分 GPU 处于空闲。
- **目标函数：** GRPO 家族，采用 token 级聚合、**clip-higher**（DAPO）或**重要性采样权重裁剪**（CISPO），或为 MoE 稳定性采用**序列级比率**（GSPO）；**零方差过滤**（丢弃全对/全错的组）；超长惩罚塑形。
- **训练–推断失配：** 采样器与训练器的对数概率不一致（BF16 舍入、非批不变 kernel、MoE 路由）→ 悄然变成异策略（off-policy）。补救：RL 使用 **FP16**、**批不变 kernel**、截断重要性采样、**路由重放**、**FP32 LM-head logits**；监控采样器与训练器之间的 KL。
- **可预测性：** 在早期计算上拟合 ScaleRL 的 S 形曲线 $R(C)=R_0+(A-R_0)/(1+(C_{\text{mid}}/C)^B)$ 来预测渐近线 $A$；大多数技巧改变的是效率 $B$，而不是 $A$。
- **安全与奖励投机：** 检测篡改测试的行为（`sys.exit(0)`、修改测试）、使用留出评分器、**接种式提示（inoculation prompting）**、对轨迹做监控；追踪思维链可监控性；决定哪些能力（例如进攻性网络能力）不放入环境配比。
- **低成本阶段：** 用**同策略蒸馏（on-policy distillation）**合并领域专家或恢复行为；LoRA 即使在低秩下也适用于 RL。

### "以 1M token 上下文服务一个约 1–3T 参数的 MoE"
- **先算内存：** MXFP4/NVFP4 权重（约 4.25 比特/参数 → 每 1T 参数约 0.5 TB）需要一个**机柜级 NVLink 域**（GB300 NVL72 级别）配合专家并行；1M 下的 KV 必须靠架构削减——MLA、**DSA/CSA top-k 稀疏注意力**（DeepSeek-V4 报告在 1M 下 KV 约为 V3.2 的 10%）或 **3:1 线性混合**（KDA/GDN：只有全注意力层持有 KV）。
- **分离部署：** 把预填充（计算受限；可用 Rubin CPX 等预填充专用部件）与解码（带宽受限）分开；对注意力开销低的模型可考虑**注意力–FFN 分离**；注意 DSA 的 **indexer K 缓存**会使 KV 传输复杂化。
- **解码速度：** 用 MTP 头作自草稿 / EAGLE-3；INT4/FP4 量化感知训练（QAT）检查点（Kimi K2 Thinking INT4、gpt-oss MXFP4）可避免事后量化损失。
- **权衡：** 稀疏 vs 混合 vs 全注意力（MiniMax 因多跳损失回退到全注意力）、专家数量 vs all-to-all 成本、为智能体循环做前缀缓存。

### "你会如何评测一个新的推理模型？"
- 组合：FrontierMath、GPQA、HLE（分别报告有/无工具）、LiveCodeBench、**SWE-bench Pro**（OpenAI 于 2026-02 停止报告 **SWE-bench Verified**：在审计的困难题中 ≥59.4% 测试有缺陷，并存在污染）、**Terminal-Bench**、OSWorld、**ARC-AGI-3**、METR 式**时间跨度**；**污染检查** + **扰动测试**；固定**测试框架/脚手架**（厂商脚手架会让 SWE-bench Pro 得分相差 10–30 分）；控制**推断预算**并报告**每任务 token/成本**；人工审查；安全/拒答评测。警惕饱和（基准"半衰期"约 12–18 个月）和**评测意识**（见[评测](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)）。

### "为一个具备两用（如网络安全）能力的模型设计发布把关流程"
- 依据框架阈值做**能力评测**（Anthropic 负责任扩展政策（RSP）v3 的 ASL + 定期风险报告；OpenAI Preparedness 的 High/Critical；GDM FSF 的 CCL）；在**与互联网隔离**的沙箱中做发布前第三方测试（METR、各国 AISI）。
- **训练时塑形：** 决定 RL 配比中包含什么（Anthropic 没有在网络安全任务上训练 Opus 5）；过滤数据。
- **部署栈：** 廉价的激活探针 → 升级分类器 → 交换级分类器（Constitutional Classifiers++：约 1% 计算开销，0.05% 拒答率）；将被标记的请求**按能力路由**到较弱的后备模型；**分级访问**（相同权重、不同防护：Fable vs Mythos；OpenAI Daybreak；验证计划）。
- **监控与响应：** 对智能体的在线动作监控、事件上报（SB 53：15 天内）、回滚/暂停预案（Fable 5/Mythos 5 在 2026-06 因防护被绕过而暂停后重新部署）、反蒸馏（保留思考（preserved thinking））。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"估算在 N 张 GPU 上训练模型 X 的时间/成本。"** 一定要大声推演 $6ND$ + MFU 的计算。
- **"你的训练 OOM / 出现损失尖峰 / 某个节点宕机——你怎么办？"** 对应到内存（检查点/并行）、稳定性（裁剪/z-loss/回滚）、可靠性（异步检查点 + 重启）。
- **"选一个并行策略并用硬件来论证。"** 节点内 TP、节点间 PP、外层 FSDP、长上下文用 SP。
- **"你的 RLHF 系统用 DPO 还是 PPO——选哪个，为什么？"** 成本/稳定性 vs 同策略质量。
- **"你的 RL 训练奖励在涨但评测在跌 / 在 N 步后崩溃——如何排查？"** 检查采样器与训练器对数概率失配、熵坍缩（clip-higher、Clip-Cov/KL-Cov）、奖励投机（检视轨迹）、异步 rollout 的陈旧度、MoE 路由漂移（GSPO/路由重放）。
- **"1M 上下文的 KV 缓存有多大，你会如何把它削减 10×？"** $2\,n_{\text{layers}}\,n_{kv}\,d_h\,L$ 个元素 × 每元素字节数；MLA、在压缩 KV 上做稀疏 top-k、线性混合、KV 量化。

## 应避免的常见陷阱
- 没有**澄清约束**或**算账**就直接给出方案。
- 忽视**内存墙**（只关注 FLOPs）。
- 忘记大规模下的**可靠性/检查点**（故障是常态）。
- 不说明**权衡**，或不说明设计在不同约束下如何变化。

## 参考文献
- Narayanan, D. et al. (2021). *Efficient Large-Scale LM Training (3D parallelism).* arXiv:2104.04473.
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- Grattafiori, A. et al. (2024). *Llama 3 Herd.* arXiv:2407.21783.
- Kwon, W. et al. (2023). *vLLM / PagedAttention.* arXiv:2309.06180.
- Khatri, D. et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL).* arXiv:2510.13786.
- Fu, W. et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.* arXiv:2505.24298.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
