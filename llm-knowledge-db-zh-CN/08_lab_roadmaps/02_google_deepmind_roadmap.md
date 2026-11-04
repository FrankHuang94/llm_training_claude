# Google DeepMind——实验室路线图（Google DeepMind — Lab Roadmap）

> **最后更新：** 2026-11-04
> **相关文件：** [缩放定律](../00_foundations/04_scaling_laws.md) · [多模态预训练](../07_multimodal/01_multimodal_pretraining.md) · [世界模型](../09_emerging_frontiers/02_world_models.md) · [实验室对比矩阵](06_lab_comparison_matrix.md)
> **关键论文：** Hoffmann et al. 2022 Chinchilla（[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)）· Chowdhery et al. 2022 PaLM（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)）· Gemini Team 2023（[arXiv:2312.11805](https://arxiv.org/abs/2312.11805)）· Gemini 1.5 2024（长上下文）（[arXiv:2403.05530](https://arxiv.org/abs/2403.05530)）

## 概述
Google DeepMind（GDM）由 Google Brain 与 DeepMind 于 2023 年合并而成，融合了 **Transformer** 的发明者（Vaswani 等人当时在 Google）与 DeepMind 在深度强化学习和科学 AI 方面的积淀（AlphaGo、AlphaFold）。它可以说是业务最广的 AI 研究机构，涵盖前沿大语言模型（LLM）（**Gemini**）、开放模型（**Gemma**）、形式化推理（**AlphaProof/AlphaGeometry**）、世界模型（world model）（**Genie**）、机器人（**Gemini Robotics**）、科学（AlphaFold）以及基础研究（Chinchilla 缩放定律（scaling law））。其差异化在于**原生多模态（multimodal）**、**TPU 垂直整合**（如今也以吉瓦规模对外销售）、**Flash 档位极具侵略性的性价比**，以及深厚的科学推理底蕴。2026年8月，Demis Hassabis 出任 GDM 董事长兼 Alphabet 首席科学家，**Koray Kavukcuoglu** 以高级副总裁身份执掌 DeepMind，直接向 Sundar Pichai 汇报。

## 模型谱系与关键创新
- **奠基性工作：** **Transformer**（2017）；**Chinchilla**（2022）——修正了 Kaplan 结论的计算最优（compute-optimal）缩放；**PaLM**（540B，并行层，多查询注意力（MQA））。
- **Gemini 1.0（2023）：** **原生多模态**（从预训练起即涵盖文本/图像/音频/视频）旗舰；Ultra/Pro/Nano 三档。
- **Gemini 1.5（2024）：** **混合专家（MoE）** + 突破性的**长上下文（1M→2M token）**，大海捞针召回能力强；Flash（蒸馏、快速）。
- **Gemini 2.0/2.5（2024–2025）：** 智能体式（agentic）（Project Mariner）、**原生工具使用（tool use）**、思考模型（2.5 Pro/Flash），数学/编码/科学能力强。
- **Gemini 3（2025-11-18）：** Gemini 3 Pro（HLE 无工具 37.5%，SWE-bench Verified 76.2%），外加 **Gemini 3 Deep Think**（HLE 41%，ARC-AGI-2 45.1%）与 **Google Antigravity**——一款智能体优先的 IDE，也支持 Claude 与 gpt-oss 智能体。它的发布引发了 OpenAI 的"红色警报"。
- **Gemini 3.1 Pro（2026-02-19）：** ARC-AGI-2 经验证为 77.1%（3 Pro：31.1%），GPQA Diamond 94.3%。
- **Gemini 3.5 / 3.6（2026-05 → 07）：** I/O 大会上，**3.5 Flash** 成为 Gemini 应用和 AI Mode 的默认模型，据称在 Terminal-Bench 2.1、GDPval-AA 和 MCP Atlas 上超过 3.1 Pro；**3.5 Pro 已发布公告但推迟上线**。7 月新增 **3.6 Flash**（输出 token 比 3.5 Flash 少约 17%，二手来源）、**3.5 Flash-Lite**，以及受限的 **3.5 Flash Cyber**（在 CodeMender 智能体中修复漏洞，仅限政府/可信合作伙伴）。
- **Gemini 4：** Pichai 于 2026年7月表示"我们正在训练 Gemini 4"；Kavukcuoglu 在 9 月下旬表示其**处于后训练（post-training）阶段**，目标在 2026 年底之前尽早发布（未披露规格）。
- **Gemma（开放模型，2024+）：** Gemma 1–3、Gemma Scope（开放的稀疏自编码器（SAE））；**Gemma 4**（2026-04-02）——首个采用 **Apache 2.0** 的版本：E2B/E4B 端侧模型（逐层嵌入（per-layer embeddings）；文本/图像/音频）、一个 **26B-A4B MoE** 和一个 **31B 稠密模型（dense model）**（256K 上下文），基于 Gemini 3 研究成果构建。
- **专门化推理与科学：** **AlphaProof/AlphaGeometry**（IMO 2024 银牌水平的形式化数学）、**AlphaCode**、AlphaFold；据报道，一个"AI 合作数学家"智能体在 FrontierMath Tier 4 上达到 48%（2026-05，二手来源）。
- **世界模型、机器人与生成：** **Genie 3**（2025-08：实时 720p/24fps 可交互世界，可保持数分钟一致，被定位为智能体训练场）；**Gemini Robotics 2 / ER 2**（2026-07：全身人形机器人 VLA，外加具备多机器人任务拆分能力的具身推理规划器）；**Veo**、**Imagen**、**Lyria**。

## 技术差异化
- 从预训练起的**原生多模态**（相对于外挂编码器）——文本/图像/音频/视频统一，现已扩展到"Omni"与机器人。
- **长上下文**——最初的 1–2M token 领先者；到 2026 年 1M 上下文已成为行业标配（Opus 4.6、GPT-5.4、DeepSeek V4），因此优势转向*可用的*召回能力与成本。
- **TPU 垂直整合**——Ironwood（TPU v7：每芯片 192 GB HBM、约 4.6 PF FP8；2026-03-31 全面上市），第 8 代拆分为 **TPU 8t（训练）** 和 **TPU 8i（推断）**；JAX/Pathways 软件栈。TPU 如今也是对外销售的商品（Anthropic：最多 100 万块芯片）。
- **Flash 档性价比**——以蒸馏（distillation）为主的 Flash 模型屡屡追平上一代 Pro（3.5 Flash 对 3.1 Pro）。
- **科学与形式化推理**——AlphaProof/AlphaGeometry/AlphaFold；深度强化学习 + 搜索的传统；Deep Think 并行推理模式。

## 已发表研究重点
缩放定律（Chinchilla、DoReMi 数据配比（data mixture））、长上下文、多模态架构、RLHF/RLAIF、推理与搜索（AlphaProof）、世界模型（Genie）、机器人、可解释性、安全（前沿安全框架（Frontier Safety Framework））以及庞大的科学研究组合。值得注意的是，GDM 可解释性团队发表了 **SAE 在下游任务上的负面结果**（在分布外有害意图检测上，线性探针优于 SAE 特征），并降低了基础 SAE 研究的优先级，转向务实的方法与模型差分（model-diffing）方法。Gemini 的技术细节公开程度不及 Gemma。

## 算力与基础设施
庞大的 **TPU** 集群，配合 **JAX/Pathways/GSPMD** 实现星球级训练——完整的软硬件垂直整合，相对租用 GPU 的竞争对手具有结构性成本优势。自 2025 年起，Google 还**以吉瓦规模对外销售 TPU 算力**（Anthropic：最多 100 万块 TPU，2026 年超过 1 GW），使 TPU 成为战略性云产品。代际节奏：Trillium（v6）→ Ironwood（v7，2026年3月全面上市）→ TPU 8t/8i（2026年4月发布；可用时间为 2026 年末或 2027 年，各来源说法不一）。与 Search（AI Mode）、Workspace、Android 和 Cloud 深度整合。

## 公开战略方向
通过广泛能力 + 科学影响实现 AGI；以**多模态 + 智能体式**的 Gemini 作为统一平台（Gemini Spark 常驻智能体、Antigravity、CodeMender）；以**世界模型**（Genie）和**机器人**作为通往具身智能体的路径；以 Apache 2.0 的**开放 Gemma** 培育生态；AI for Science；并通过**前沿安全框架 v3**（有害操纵 CCL、抗关停协议）以及对具备网络攻击能力模型的受限发布来保障安全。2026年8月领导层整合到 Kavukcuoglu 之下，以收紧从研究到产品的闭环。

## 已知优势与劣势
- **优势：** 原生多模态、TPU 成本/规模优势（以及 TPU 收入）、无可比拟的广度（LLM + 科学 + 强化学习 + 世界模型 + 机器人 + 生成）、巨大的分发渠道（Search/Android/Cloud）、出色的推理基准（benchmark）成绩（3.1 Pro 的 ARC-AGI-2），以及 Flash 档性价比。
- **劣势：** 交付节奏不稳（3.5 Pro 已公告但推迟）、组织复杂且 2026 年领导层重组、在智能体式编码上不时被认为落后于 Anthropic/OpenAI，以及 Gemini 的技术披露不如 Gemma 完整。

## 招聘重点
基于 TPU/JAX 的大规模训练、多模态与长上下文架构、强化学习与搜索（推理、智能体）、世界模型与机器人、数据/缩放研究、可解释性与安全（前沿安全），以及 AI for Science。职位描述强调 JAX/TPU、强化学习、研究深度和星球级系统。

## 最新进展（2023–2026年11月）
2023 年至 2025 年中交付了 Gemini 1.0 → 2.5（原生多模态、1–2M 上下文、思考模型）、Gemma 3 和 AlphaProof。此后 GDM 转向快速的 Pro/Flash 发布节奏，彻底开放了 Gemma，并大力推进世界模型与机器人。

**2025年中 → 2026年11月更新：**
- **2025-08——Genie 3：** 实时、可提示的交互式世界模型（720p/24fps，可保持数分钟一致），被明确定位为面向智能体强化学习的可扩展环境生成器（见[世界模型](../09_emerging_frontiers/02_world_models.md)）。
- **2025-09——FSF v3.0：** 新增有害操纵 CCL，以及针对抗关停/抗修改的失准协议；安全论证（safety case）审查扩展到大规模内部部署。据报道的 **v3.1**（2026-04；版本号存在争议）在 CCL 之下新增早期预警的**跟踪能力等级（Tracked Capability Levels）**，并提升 CBRN/网络/操纵类 CCL 的安全等级。
- **2025-11 → 2026-02——Gemini 3 → 3.1 Pro：** ARC-AGI-2 在一次小版本更新中从 31.1% 升至 77.1%；同一版本线内如此大幅的跃升提示其来自后训练/强化学习而非新的预训练（GDM 未披露具体配方），也说明推理类基准饱和之快。
- **2026-03 → 04——Ironwood 全面上市与 TPU 8t/8i：** TPU 产品线首次拆分为训练专用和推断专用芯片（据称相对 Ironwood 训练性价比提升 2.7 倍、推断性价比最多提升 80%），与业界的预填充/解码分离相呼应。
- **2026-04——Gemma 4（Apache 2.0）：** 26B-A4B MoE 和逐层嵌入端侧模型，把 Gemini 3 时代的配方带到了宽松许可的开放权重（open weights）上。
- **2026-05 → 07——Flash 优先发布与受限网络安全模型：** 3.5 Flash 成为默认消费级模型；3.5 Flash Cyber 仅限经审核用户使用——GDM 与 Anthropic/OpenAI 一道对两用能力实行分级访问。
- **2026-07 → 09——Gemini 4 进入训练/后训练；Gemini Robotics 2：** 具身 VLA + 推理规划器，支持多机器人协作。

## 面试要点
> 💡 **实验室实际会问的问题（GDM 风格）：**
- **"解释 Chinchilla 缩放，以及它为何修正了 Kaplan。"**（见[缩放定律](../00_foundations/04_scaling_laws.md)。）
- **"你会如何构建一个 1M token 上下文的模型——并让它足够便宜？"** 位置编码外推 + 稀疏/混合注意力 + KV 缓存（KV cache）管理（见[上下文扩展](../06_efficiency/06_context_window_extension.md)）。
- **"原生多模态与后期融合多模态——各有何权衡？"**（见[多模态预训练](../07_multimodal/01_multimodal_pretraining.md)。）
- **"AlphaProof 如何实现形式化推理？"** 在形式化（Lean）证明上进行强化学习 + 搜索，使用可验证奖励。
- **"为什么要把加速器拆分为训练芯片和推断芯片？"** 训练、预填充与解码对计算/带宽/内存的比例需求不同；每 token 成本。
- **"你会如何用 Genie 3 这样的世界模型训练智能体？"** 环境生成、仿真到现实的差距、一致性时长、奖励设定。

## 开放问题 / 关注事项
发布 Gemini 4 和推迟的 3.5 Pro；把研究广度转化为智能体式编码领域的领先；2026 年领导层变动的影响；把长上下文扩展到*可用*的召回水平；把 Genie 和 Gemini Robotics 转化为真实世界的智能体收益；以及把 TPU 和科学优势（包括对外销售 TPU）转化为持久的前沿领先。

## 参考文献
- Hoffmann, J. et al. (2022). *Chinchilla.* arXiv:2203.15556.
- Chowdhery, A. et al. (2022). *PaLM.* arXiv:2204.02311.
- Gemini Team (2023). *Gemini.* arXiv:2312.11805.
- Gemini Team (2024). *Gemini 1.5.* arXiv:2403.05530.
- AlphaGeometry: Trinh, T. et al. (2024). *Nature.*
- Google DeepMind (2025). *Frontier Safety Framework v3.0.*
- Google DeepMind (2025). *Genie 3: A New Frontier for World Models.*
