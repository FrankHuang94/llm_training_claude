# Meta AI——实验室路线图（Meta AI — Lab Roadmap）

> **最后更新：** 2026-11-04
> **相关文件：** [Transformer 架构](../00_foundations/00_transformer_architecture.md) · [模型架构变体](../00_foundations/05_model_architecture_variants.md) · [实验室对比矩阵](06_lab_comparison_matrix.md)
> **关键论文：** Touvron et al. 2023 LLaMA（[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)）· Touvron et al. 2023 Llama 2（[arXiv:2307.09288](https://arxiv.org/abs/2307.09288)）· Grattafiori et al. 2024 Llama 3（[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)）· Zhang et al. 2022 OPT（[arXiv:2205.01068](https://arxiv.org/abs/2205.01068)）

## 概述
Meta AI（FAIR + 应用型 GenAI 部门，2025 年中重组为 **Meta 超级智能实验室（Meta Superintelligence Labs，MSL）**）直到 Llama 4 为止，一直是**开放权重（open weights）**前沿模型的主导力量。**LLaMA** 系列让强大的基座模型变得人人可得，实际上创造了开放 LLM 生态——其架构选择（RMSNorm + RoPE + SwiGLU）成为社区标准，并催生了成千上万的衍生模型。Meta 最初的押注是：**开放**能带来生态主导、吸引人才、通过公众审视实现安全，并使模型层商品化。**2026年4月，这一押注在前沿层面发生了改变**：MSL 的首个模型 **Muse Spark** 以**闭源权重**发布，随后是同样闭源的 **Muse Wonder** 前沿模型（2026年10月）。Meta 表示未来的 Muse 模型*可能*开放，但开放权重前沿如今由中国实验室、gpt-oss、Gemma 和 Mistral 引领。Meta 仍然主导着业界主要框架 **PyTorch**。

## 模型谱系与关键创新
- **OPT（2022）：** 对 GPT-3 的早期开放复现，附带坦诚的训练**日志**（记录了不稳定问题）——透明度的里程碑。
- **LLaMA 1（2023）：** 高效、**过度训练**的较小模型（推断最优范式）；确立了开放配方（RMSNorm/RoPE/SwiGLU）。泄露/发布的权重点燃了开放生态（Alpaca、Vicuna 等）。
- **Llama 2（2023）：** **附带商业许可**的开放模型；公开记录了 **RLHF + 拒绝采样（rejection sampling）**流水线、Ghost Attention 和安全措施（Llama Guard）。
- **Llama 3 / 3.1 / 3.2 / 3.3（2024）：** 扩展到 **405B**，约 15T token，128K 上下文，多语言/代码/数学能力强；3.2 新增**视觉**（交叉注意力适配器（adapter））和小型端侧模型；技术报告详尽且公开。
- **Llama 4（2025-04）：Scout / Maverick / Behemoth**——Meta 转向**混合专家（MoE）**（17B 激活；Scout 16 个专家、10M token 上下文，Maverick 128 个专家），并采用原生早期融合多模态；Behemoth 发布时只做了预览而未发布。与 DeepSeek/Qwen 相比反响不一；而且在测试了约 27 个私有变体之后，其 LMArena 参赛版本是一个与正式发布模型不同的"实验"变体。
- **Muse Spark（2026-04-08）：** MSL 的首个模型，也是 **Muse** 家族的第一个；**闭源权重**，向合作伙伴提供私有预览 API。Meta 声称其推理能力达到或超过 Llama-4-Maverick 水平而所用算力远少，在多模态理解和健康领域最强。
- **Muse Wonder / Muse Wonder Full（2026-10-12）：** 面向智能体（agent）工作的闭源前沿模型，带有用于错误恢复的"**Active Memory**"和"**Active Focus**"机制；据报道上下文为 1M，定价 $4/$20（二手来源）。Meta 声称其胜过 OpenAI 和 Anthropic 的模型；独立竞技场把它们排在接近但低于 Claude Opus 5.5 的位置。
- **其他：** **Code Llama**、**SAM**（分割）、**ImageBind**、**Chameleon**（早期融合）、**V-JEPA** 世界模型（world model）、**Movie Gen**、**Massively Multilingual Speech**，以及大量 FAIR 研究。有报道称存在较小的开放 Muse 变体，但**未经核实**。

## 技术差异化
- **（曾经的）前沿开放权重**——Llama 确立了社区基线架构；Muse 产品线标志着闭源前沿模型与（可能）开放的较小模型之间的有意分化。
- **为推断效率而过度训练**——小而强、服务成本低的模型；Muse Spark 的卖点依然是计算效率。
- **PyTorch + FSDP/TorchTitan** 生态的领导地位。
- 截至 Llama 3 的**公开、详尽的技术报告**，以及开放的安全工具（**Purple Llama / Llama Guard**）；Muse 系列迄今未附带同等水平的报告。

## 已发表研究重点
开放基础模型与详尽报告、高效架构、多语言与语音、多模态（Chameleon、ImageBind、SAM）、自监督世界模型（JEPA 系列）、大规模数据整理/去重（deduplication）、训练稳定性（OPT）、无分词器建模（Byte Latent Transformer），以及广泛的 FAIR 基础研究组合。Meta 研究人员联署了 2025 年关于**思维链（CoT）可监控性**的跨实验室立场论文（arXiv:2507.11473）。重心已从 FAIR 的开放发表文化转向 MSL 以产品为导向、较少公开的前沿工作。

## 算力与基础设施
超大规模 GPU 集群（Llama 3 在 **1.6 万块 H100** 上训练，并详细披露了容错方案；目标是到 2024 年底拥有 35 万块以上的 H100 等效算力），如今以**吉瓦**规模签约：先是扩大与 NVIDIA 的合作，数日后又签下**最多 6 GW 的 AMD GPU**（2026-02-24）。Meta 的 Open Rack Wide 规范是 AMD **Helios** MI450 机架设计的基础。吉瓦级园区（据报道为 *Prometheus* 和 *Hyperion*）已在规划中，但其进展在此未经核实。Meta 推动着业界通用的 **PyTorch**、FSDP 和开放训练栈；并与 Instagram/WhatsApp/Facebook 以及设备（Ray-Ban Meta）垂直整合。

## 公开战略方向
2024 年的口号是"**开源 AI 是前进之路**"（Zuckerberg）。自 2025 年中起，公开目标变为通过 MSL 实现**"超级智能"**，辅以激进招聘和吉瓦级算力，开放则变为**选择性**的：前沿 Muse 模型闭源，较小模型日后可能开放。Meta 还拒绝签署欧盟 GPAI 行为准则（2025），这与其对前沿严格监管的怀疑态度一致。AI 已整合进 Meta 的各类应用、广告和硬件（智能眼镜、助手）。

## 已知优势与劣势
- **优势：** 巨大的分发渠道与算力、PyTorch 主导权、制定架构标准的历史积累、多模态与健康领域的优势（Muse Spark）、2025 年招聘后的人才密度，以及快速进步的闭源前沿产品线（Muse Wonder 在竞技场中接近顶端）。
- **劣势：** Llama 4 反响不一及基准（benchmark）展示争议、反复重组、开放生态领导地位被 Qwen/DeepSeek/Kimi 夺走、转向闭源后研究透明度下降，以及在智能体式（agentic）编码上相对 OpenAI/Anthropic/Google 尚未证明自己。

## 招聘重点
大规模预训练（pretraining）与分布式系统（PyTorch/FSDP）、面向推理和智能体的强化学习（RL）/后训练（post-training）、多模态与语音、记忆/长上下文架构（Active Memory）、高效架构（MoE），以及端侧/效率。职位描述强调 PyTorch、规模化和系统工程；MSL 岗位强调前沿模型经验。

## 最新进展（2023–2026年11月）
2023 年至 2025 年中，Meta 从 Llama 2 走到公开详尽记录的 Llama 3 405B 模型，再到 Llama 4 的 MoE 转型，随后组建了 MSL。此后 Meta 以闭源模型重启了其前沿产品线。

**2025年中 → 2026年11月更新：**
- **2025——Llama 4 余波：** MoE + 10M 上下文并未转化为领先地位，而竞技场变体事件使"评测的究竟是哪个检查点（checkpoint）？"成为标准的评测卫生问题（见[评测](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)）。
- **2025（年中）——MSL 重组：** 前沿工作整合到 Meta 超级智能实验室之下，并进行大规模招聘；FAIR 的开放发表模式被弱化。
- **2026-02——吉瓦级多供应商算力：** 在扩大与 NVIDIA 合作的基础上，再签下最多 6 GW 的 AMD GPU——Meta 与 OpenAI/Anthropic 一样，开始以吉瓦为单位跨供应商签约算力。
- **2026-04——Muse Spark（闭源）：** 与开放前沿权重的战略决裂；Meta 强调计算效率（以远少的算力达到 Maverick 级推理）以及多模态/健康领域优势。
- **2026-10——Muse Wonder：** 面向智能体的"Active Memory/Active Focus"错误恢复机制，据报道上下文为 1M；竞技场排名接近但低于 Opus 5.5，说明 Meta 已重回前沿附近，但尚未领先。

## 面试要点
> 💡 **实验室实际会问的问题（Meta 风格）：**
- **"为什么要把小模型训练到超过 Chinchilla 最优？"** 推断成本在模型生命周期总成本中占主导（见[缩放定律](../00_foundations/04_scaling_laws.md)）。
- **"描述 Llama 2/3 的后训练流水线。"** SFT + 拒绝采样 + RLHF（PPO）/DPO（见[后训练](../03_posttraining/00_posttraining_overview.md)）。
- **"Llama 如何处理 128K 上下文以及 1.6 万块 GPU 上的故障？"** RoPE 缩放 + 检查点/容错（见[检查点](../02_pretraining/06_checkpoint_and_resumption.md)）。
- **"开放权重发布有何权衡——一家实验室为什么可能停止发布其前沿模型？"** 生态/安全审视 vs 滥用、被竞争对手蒸馏（distillation），以及丧失差异化。
- **"你会如何为错误恢复设计智能体记忆？"** 外部/可学习记忆、检索 vs 压缩、长时程上的信用分配（参见 Muse Wonder 的 Active Memory）。

## 开放问题 / 关注事项
Muse Wonder/Full 能否登上智能体基准的最顶端；Meta 是否会开放任何 Muse 权重（以及以什么规模）；MSL 在反复重组后的持久性；吉瓦级算力的回报；在欧盟的监管摩擦；以及社区标准架构如今是否会追随中国开放模型而非 Llama。

## 参考文献
- Touvron, H. et al. (2023). *LLaMA.* arXiv:2302.13971.
- Touvron, H. et al. (2023). *Llama 2.* arXiv:2307.09288.
- Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Zhang, S. et al. (2022). *OPT.* arXiv:2205.01068.
- Pagnoni, A. et al. (2024). *Byte Latent Transformer.* arXiv:2412.09871.
- Korbak, T. et al. (2025). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* arXiv:2507.11473.
