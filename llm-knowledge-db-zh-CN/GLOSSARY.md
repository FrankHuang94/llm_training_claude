# 术语对照表（Glossary）

> 本译本统一使用以下译法。每个文件中术语**首次出现**时采用"中文（English）"形式，此后可只用中文或通用英文缩写。模型名、论文标题、作者名、arXiv 编号、代码、公式、基准名称（MMLU、SWE-bench 等）一律保留英文原文。

## 文件模板章节

| English | 中文 |
|---|---|
| Overview | 概述 |
| Core Concepts | 核心概念 |
| Key Challenges | 关键挑战 |
| Solutions & Current Best Practices | 解决方案与当前最佳实践 |
| Lab Perspectives | 各实验室观点 |
| Latest Developments | 最新进展 |
| Interview Angles | 面试要点 |
| What labs actually ask | 实验室实际会问的问题 |
| Open Problems | 开放问题 |
| References | 参考文献 |
| Last Updated | 最后更新 |
| Related Files | 相关文件 |
| Key Papers | 关键论文 |

## 核心术语

| English | 中文 | 备注 |
|---|---|---|
| Large Language Model (LLM) | 大语言模型（LLM） | |
| Transformer | Transformer | 不译 |
| attention / self-attention | 注意力 / 自注意力 | |
| multi-head / grouped-query / multi-query attention | 多头 / 分组查询 / 多查询注意力（MHA/GQA/MQA） | |
| KV cache | KV 缓存 | |
| token / tokenizer / tokenization | token / 分词器 / 分词 | token 不译 |
| embedding | 嵌入 | |
| positional encoding | 位置编码 | |
| scaling law | 缩放定律 | |
| compute-optimal | 计算最优 | |
| emergence / emergent ability | 涌现 / 涌现能力 | |
| pretraining / post-training / mid-training | 预训练 / 后训练 / 中期训练 | |
| fine-tuning | 微调 | |
| supervised fine-tuning (SFT) | 监督微调（SFT） | |
| instruction tuning | 指令微调 | |
| reinforcement learning (RL) | 强化学习（RL） | |
| RLHF | 基于人类反馈的强化学习（RLHF） | |
| RLAIF | 基于 AI 反馈的强化学习（RLAIF） | |
| RLVR (RL with verifiable rewards) | 可验证奖励强化学习（RLVR） | |
| reward model / reward hacking | 奖励模型 / 奖励投机（reward hacking） | |
| reward over-optimization | 奖励过度优化 | |
| preference optimization | 偏好优化 | |
| policy / value function / critic | 策略 / 价值函数 / 评论家（critic） | |
| rollout | rollout（采样轨迹） | |
| on-policy / off-policy | 同策略（on-policy）/ 异策略（off-policy） | |
| distillation | 蒸馏 | |
| rejection sampling | 拒绝采样 | |
| reasoning | 推理 | |
| inference | 推断（inference） | 为与 reasoning 区分；"inference cost" → 推断成本 |
| inference-time / test-time compute | 推断时 / 测试时计算 | |
| chain of thought (CoT) | 思维链（CoT） | |
| thinking tokens | 思考 token | |
| agent / agentic | 智能体 / 智能体式（agentic） | |
| tool use / function calling | 工具使用 / 函数调用 | |
| computer use | 计算机操作（computer use） | |
| alignment | 对齐 | |
| scalable oversight | 可扩展监督 | |
| interpretability / mechanistic interpretability | 可解释性 / 机制可解释性 | |
| superposition / feature / circuit | 叠加 / 特征 / 回路 | |
| sparse autoencoder (SAE) | 稀疏自编码器（SAE） | |
| jailbreak / red teaming | 越狱 / 红队测试 | |
| prompt injection | 提示注入 | |
| hallucination | 幻觉 | |
| sycophancy | 谄媚 | |
| deceptive alignment / scheming | 欺骗性对齐 / 图谋（scheming） | |
| benchmark / evaluation | 基准 / 评测 | |
| contamination / leakage | 污染 / 泄漏 | |
| deduplication | 去重 | |
| synthetic data | 合成数据 | |
| curriculum | 课程学习 | |
| data mixture | 数据配比 | |
| mixture of experts (MoE) | 混合专家（MoE） | |
| expert / router / load balancing | 专家 / 路由器 / 负载均衡 | |
| dense model | 稠密模型 | |
| sparse attention / linear attention | 稀疏注意力 / 线性注意力 | |
| state space model (SSM) | 状态空间模型（SSM） | |
| quantization | 量化 | |
| mixed precision | 混合精度 | |
| parameter-efficient fine-tuning (PEFT) | 参数高效微调（PEFT） | |
| LoRA / adapter | LoRA / 适配器 | |
| speculative decoding | 投机解码 | |
| draft model | 草稿模型 | |
| context window / long context | 上下文窗口 / 长上下文 | |
| retrieval-augmented generation (RAG) | 检索增强生成（RAG） | |
| data / tensor / pipeline parallelism | 数据 / 张量 / 流水线并行 | |
| sharding / checkpoint | 分片 / 检查点 | |
| activation checkpointing | 激活检查点 | |
| throughput / latency | 吞吐量 / 延迟 | |
| optimizer / learning rate / warmup | 优化器 / 学习率 / 预热 | |
| loss spike | 损失尖峰 | |
| gradient clipping | 梯度裁剪 | |
| continual learning / catastrophic forgetting | 持续学习 / 灾难性遗忘 | |
| test-time training (TTT) | 测试时训练（TTT） | |
| world model | 世界模型 | |
| multimodal / vision-language model (VLM) | 多模态 / 视觉-语言模型（VLM） | |
| open weights / closed | 开放权重 / 闭源 | |
| frontier lab / frontier model | 前沿实验室 / 前沿模型 | |
| system card | 系统卡（system card） | |
| Responsible Scaling Policy (RSP) | 负责任扩展政策（RSP） | |
| export controls | 出口管制 | |
