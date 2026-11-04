# OpenAI——实验室路线图（OpenAI — Lab Roadmap）

> **最后更新：** 2026-11-04
> **相关文件：** [推理模型](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md) · [RLHF](../03_posttraining/03_RLHF.md) · [实验室对比矩阵](06_lab_comparison_matrix.md)
> **关键论文：** Brown et al. 2020 GPT-3（[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)）· Ouyang et al. 2022 InstructGPT（[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)）· OpenAI 2023 GPT-4（[arXiv:2303.08774](https://arxiv.org/abs/2303.08774)）· OpenAI 2024 o1 "Learning to Reason" · OpenAI 2025 gpt-oss model card

## 概述
OpenAI 是把大语言模型（LLM）带入大众应用的公司（ChatGPT，2022年11月），并多次定义了前沿：GPT 缩放路线、RLHF/InstructGPT、GPT-4 的多模态飞跃、**o 系列推理范式**（推断时（inference-time）缩放），以及自2025年8月起统一的 **GPT-5 → GPT-6** 系统产品线。其战略结合了对预训练与强化学习（RL）算力的*同步*激进扩展、闭源/产品优先的模式（一个显著例外是开放权重（open weights）的 **gpt-oss**），以及构建"惠及全人类"的 AGI 这一公开使命。它已从以 Microsoft 为中心的算力模式转向**吉瓦级、多供应商合同**（NVIDIA、AMD、Broadcom、Cerebras），其"非营利组织下的利润上限公司"结构自2025年起一直处于重组之中；据广泛报道，其 IPO 目标定在2026年末。

## 模型谱系与关键创新
- **GPT-1/2/3（2018–2020）：** 确立了仅解码器的生成式预训练（pretraining）与**缩放定律（scaling law）**；GPT-3（175B）展示了**上下文学习/少样本学习**。
- **InstructGPT → ChatGPT（2022）：** 基于人类反馈的强化学习（RLHF）对齐——可用性上的突破；1.3B 的对齐模型在偏好评测上胜过 175B 基座模型。
- **GPT-4（2023）：** 大型多模态（视觉）模型，普遍认为是**混合专家（MoE）**；GPT-4-Turbo/4o 带来了更快速度、128K 上下文，以及 **GPT-4o** 中**原生任意模态到任意模态**的多模态能力（文本/图像/音频）。
- **o1 → o3 / o4-mini（2024–2025）：** 用强化学习训练、先思考后回答的**推理模型**，扩展**测试时计算（test-time compute）**。GPT-4.1（2025年4月）为 API 带来 1M token 上下文。
- **gpt-oss-120b / 20b（2025-08）：** 自 GPT-2 以来首批开放权重 LLM（Apache 2.0）。总参数 116.8B / 激活 5.1B（128 个专家，top-4），交替使用带状（滑动窗口）与稠密 GQA 层，并带有**可学习的注意力汇（attention sink）偏置**；MoE 权重经后训练（post-training）转为 **MXFP4**（约 4.25 bit/参数），使 120b 可放进单张 80 GB GPU；约 210 万 H100 小时；推理能力接近 o4-mini。
- **GPT-5（2025-08-07）：** 一个*统一系统*——快速模型 + 思考模型 + 实时**路由器（router）**——吸收了 o 系列。路由器在发布当天出现故障，用户强烈反弹后旧模型被恢复。
- **GPT-5.1 → 5.2（2025-11/12）：** 5.1 有 Instant/Thinking 两版；5.2（Instant/Thinking/Pro）主打长上下文推理与更少幻觉，在 Gemini 3 发布后内部"红色警报（code red）"之下推出。
- **GPT-5.3-Codex-Spark（2026-02）：** 运行在 **Cerebras WSE-3** 上的小型实时编码模型（>1,000 tok/s）——OpenAI 首次在非 NVIDIA 芯片上进行生产部署。
- **GPT-5.4（2026-03）：** 首个具备**原生计算机操作（computer use）**（截图 + 鼠标/键盘）的通用模型，约 1.05M 上下文，OSWorld-Verified 75%（厂商自报）。
- **GPT-5.5（2026-04）：** 单 token 延迟不变，每个智能体任务消耗的 token 更少；Terminal-Bench 2.0 82.7%（5.4：75.1%）；据报道独立的 Codex 产品线已并入其中。
- **GPT-5.6 Sol / Terra / Luna（2026-06/07）：** 首个三档模型家族（旗舰/均衡/低价）；据报道，有限预览期间接受了政府安全审查，7月9日全面上线。**GPT-5.6-Cyber**（约2026-08，网络安全评级为"High"）通过 **Daybreak** 可信访问计划提供。
- **GPT-6 Astra（2026-09-03）：** 新旗舰；首个在准备度框架（Preparedness Framework）下网络安全能力被评为**"Critical"**的 OpenAI 模型（即使内部部署也需安全防护措施）。1.05M 上下文、128K 输出；性能提升归因于大规模预训练加长时程（long-horizon）强化学习；据报道是 OpenAI 规模最大的一次训练（在得克萨斯州 Stargate 站点使用超过 10 万块 GPU；单一来源）。
- **GPT-6 Sol / Luna（2026-09-22）：** 能力约为 GPT-5.6 水平，价格约减半（评级为"High"而非 Critical）；据报道 **GPT-6.1 Sol** 在 DevDay（2026-09-29）发布，在编码/计算机操作上接近 Astra，价格约为其 1/5。
- **生态：** **DALL·E / GPT-image**（图像）、**Sora**（视频/隐式世界模型（world model））、**Whisper**（开源语音识别）、**Operator → Deep Research → Codex**（智能体（agent）；据单一报道，Codex CLI 默认模型为 GPT-6.1 Sol）。

## 技术差异化
- **RLHF 开创者**——发明了现代后训练范式。
- **推断时缩放的领导者**——o 系列将"难题多想一会儿"产品化；GPT-5 将其变为由*路由*决定的产品决策，单任务 token 效率成为头条指标（5.5）。
- **重新扩展预训练 + 长时程强化学习**——GPT-6 Astra 被明确定位为回归超大规模预训练，并结合智能体式（agentic）强化学习。
- **基座模型内置智能体式计算机操作**（GPT-5.4 起），而非独立的 Operator 产品。
- **分级可信访问**——Daybreak Blue（主线模型上减少拒答）/ Red（网络安全专用模型，更严格的身份验证）。
- **产品 + 平台**——ChatGPT + API + Codex + 企业服务；最快实现大规模分发。

## 已发表研究重点
RLHF/InstructGPT、缩放定律（Kaplan）、奖励模型过度优化（Gao）、过程监督（"Let's Verify Step by Step"）、弱到强泛化、基于规则的奖励、审慎对齐（deliberative alignment）以及 o 系列。2025–26 年的产出集中在：**思维链（CoT）可监控性**（13 项评测的 "Monitoring Monitorability" 套件，arXiv:2512.18311）；与 Apollo 合作的**反图谋（anti-scheming）训练**（o3 的隐蔽行为从 13% 降到 0.4%，但被上升的评测感知所混淆；arXiv:2509.15541）；用于诚实性的**"自白"（confessions）**（arXiv:2512.08093）；**幻觉即激励问题**的分析（arXiv:2509.04664）；以及经济类评测（**GDPval**，1,320 个任务 / 44 种职业）。模型层面的公开仍然谨慎（发布系统卡（system card）而非技术报告），gpt-oss 的 model card 是主要的架构披露。

## 算力与基础设施
从与 Azure 的独家关系转向**约 26 GW 的多供应商组合**（2025年末）：**NVIDIA** ≥10 GW（NVIDIA 最多投资 1000 亿美元）、**AMD** 6 GW（覆盖多代 Instinct，从 2026年下半年的 MI450 开始；OpenAI 持有最多约 10% AMD 股份的认股权证），以及 **Broadcom** 10 GW 的联合设计定制加速器与网络；**Stargate**/Oracle 站点（当时已承诺约 7 GW）承载训练。**Cerebras** 晶圆级芯片服务低延迟档位（Codex-Spark；Sol 计划最高 750 tok/s）。据报道 Amazon 承诺投资 500 亿美元并提供 2 GW Trainium（单一来源）。融资：创纪录的 **1220 亿美元融资，投后估值 8520 亿美元**（2026-03，二手来源）。

## 公开战略方向
安全且广泛有益地构建 **AGI**；迭代式部署（"ship to learn"），如今节奏约为每 4–8 周一次；同步扩展**预训练、强化学习与推断**算力；能操作计算机、可连续运行数小时的智能体成为主要交互界面（Codex、ChatGPT agent）；对两用能力采用**分级/可信部署**（Daybreak；Astra 的内部部署防护）。在使命之外具有很强的商业导向（ChatGPT、API、企业）。

## 已知优势与劣势
- **优势：** 前沿推理与智能体式编码（2026 年多个时间点在 Terminal-Bench、OSWorld 上领先）、产品分发/品牌（ChatGPT）、最大的已签约算力储备、完整的多模态 + 智能体生态、评测专长（GDPval、可监控性），以及伴随大幅降价的快速分级发布（Sol/Luna）。
- **劣势：** 闭源/不透明（只有系统卡而无报告）、发布可靠性问题（GPT-5 路由器）、治理动荡与重组、沉重的资本/算力承诺、模型迭代过快（5.5 约 6 个月内即从 ChatGPT 下线，单一来源），以及来自 Anthropic（智能体式编码）和 Google（性价比、多模态）的激烈竞争。

## 招聘重点
面向长时程智能体的强化学习/后训练、大规模预训练与推断系统（多供应商：NVIDIA/AMD/定制芯片/Cerebras）、评测（智能体、经济、可监控性）、安全/准备度（网络、生物、图谋）、计算机操作/多模态，以及智能体产品工程。职位描述强调规模化经验、强化学习、分布式系统，以及在超大规模下交付可靠产品。

## 最新进展（2023–2026年11月）
2023 年至 2025 年中确立了 o 系列（o1 → o3/o4-mini）与测试时计算这一缩放轴、GPT-4o 的原生多模态，以及 GPT-4.1 的 1M 上下文。此后 OpenAI 统一了对话与推理，重新扩展了预训练，并转向分级、受控的部署。

**2025年中 → 2026年11月更新：**
- **2025-08——gpt-oss：** 披露了 OpenAI 风格 MoE 的架构（稀疏的 5.1B 激活、滑动/稠密交替、注意力汇、MXFP4 MoE 权重）——成为"单张 80 GB GPU 即可运行"的开放推理模型的参考设计。（OpenAI model card）
- **2025-08——GPT-5 路由器：** 推理深度变成*可学习的分发决策*（快速模型 vs 思考模型）；发布当天的路由故障表明，路由质量本身就是面向用户的能力。
- **2025-12——Monitoring Monitorability：** CoT 监控器优于仅看行为的监控器，更长的 CoT 更易监控，直接对 CoT 施加奖励压力会损害可监控性，而当前规模的强化学习并未实质性损害它——这对推理强化学习的奖励设计构成约束。（arXiv:2512.18311）
- **2026-02 → 03——Cerebras 上的 Codex-Spark 与 GPT-5.4 计算机操作：** 跨异构硬件的按延迟分级服务，以及把 GUI 控制并入通用模型（约 1.05M 上下文）。
- **2026-04——GPT-5.5 / 弃用 SWE-bench Verified：** OpenAI 不再报告 SWE-bench Verified（污染、测试有缺陷），改用 SWE-bench Pro；5.5 强调每个 Codex 任务更少的 token——单任务成本成为一等指标。
- **2026-06 → 09——Sol/Terra/Luna 与 GPT-6：** 三档模型家族，在能力不变时每代价格约降一半；GPT-6 Astra 是准备度框架 v2 下首个"Critical"级网络安全模型，首先部署给 Daybreak 企业客户。
- **算力（2025-09 → 10）：** 在 NVIDIA、AMD 和 Broadcom 之间约 26 GW——算力如今以吉瓦为单位签约，并附带供应商股权/认股权证结构。

## 面试要点
> 💡 **实验室实际会问的问题（OpenAI 风格）：**
- **"解释 InstructGPT 以及 RLHF 为何重要。"** 就可用性而言对齐胜过规模；SFT→RM→PPO 流水线。
- **"o1/o3 与 GPT-4o 有何不同——为什么要把它们并入 GPT-5？"** 用强化学习训练的推理 + 测试时计算，对比快速多模态对话；路由器在简单/困难查询之间摊薄思考成本，但也引入了新的失效模式。
- **"什么是推断时缩放，有何权衡？"** 用测试时计算换准确率；成本/延迟；在报告准确率的同时报告单任务 token 数。
- **"讲讲 gpt-oss 的架构选择。"** 128 选 4 的 MoE，用滑动/稠密交替 + 注意力汇支持长上下文，用 MXFP4 MoE 权重实现单 GPU 服务。
- **"你会如何为一个网络安全'Critical'级模型设计部署方案？"** 可信访问分级、身份验证、分类器 + 回退路由、内部部署防护。
- **"你会如何为推理模型设计评测？"**（见[评测](../05_alignment_and_safety/05_evaluation_and_benchmarks.md)。）

## 开放问题 / 关注事项
治理/结构转型与 IPO；兑现约 26 GW 已签约算力；在模型越过"Critical"阈值时保持安全信誉；在更重的强化学习下保持 CoT 可监控；推理智能体的经济化推断；长时程自主中的智能体可靠性；以及面对 Anthropic、Google 和中国开放权重实验室保持领先。

## 参考文献
- Brown, T. et al. (2020). *GPT-3.* arXiv:2005.14165.
- Ouyang, L. et al. (2022). *InstructGPT.* arXiv:2203.02155.
- OpenAI (2023). *GPT-4 Technical Report.* arXiv:2303.08774.
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050.
- OpenAI (2024). *Learning to Reason with LLMs (o1).*
- OpenAI (2025). *gpt-oss-120b & gpt-oss-20b Model Card.*
- Kalai, A. et al. (2025). *Why Language Models Hallucinate.* arXiv:2509.04664.
- OpenAI & Apollo Research (2025). *Stress Testing Deliberative Alignment for Anti-Scheming Training.* arXiv:2509.15541.
- Guan, M. et al. (2025). *Monitoring Monitorability.* arXiv:2512.18311.
- Joglekar, M. et al. (2025). *Training LLMs for Honesty via Confessions.* arXiv:2512.08093.
