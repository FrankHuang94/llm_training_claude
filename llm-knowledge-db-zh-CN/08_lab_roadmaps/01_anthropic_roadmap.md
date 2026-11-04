# Anthropic——实验室路线图（Anthropic — Lab Roadmap）

> **最后更新：** 2026-11-04
> **相关文件：** [Constitutional AI 与 RLAIF](../03_posttraining/07_constitutional_AI_and_RLAIF.md) · [可解释性](../05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md) · [可扩展监督](../05_alignment_and_safety/04_scalable_oversight.md) · [实验室对比矩阵](06_lab_comparison_matrix.md)
> **关键论文：** Bai et al. 2022 Constitutional AI（[arXiv:2212.08073](https://arxiv.org/abs/2212.08073)）· Elhage et al. 2022 Toy Models of Superposition（[arXiv:2209.10652](https://arxiv.org/abs/2209.10652)）· Templeton et al. 2024 Scaling Monosemanticity · Bai et al. 2022 HH-RLHF（[arXiv:2204.05862](https://arxiv.org/abs/2204.05862)）· MacDiarmid et al. 2025 Emergent Misalignment from Reward Hacking（[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)）

## 概述
Anthropic 由前 OpenAI 研究人员（Dario 与 Daniela Amodei 及其同事）于 2021 年创立，使命明确为**安全优先**：在构建前沿 AI 的同时，开创使其安全、可控的技术。它既是一家前沿实验室（frontier lab）（**Claude** 系列——到 2026 年末涵盖 Haiku/Sonnet/Opus，外加受限的 **Mythos 级**档位），也是业界发表**对齐（alignment）研究**最多的机构（Constitutional AI、机制可解释性（mechanistic interpretability）、可扩展监督（scalable oversight）、模型生物（model-organism）研究）。其定位——"我们必须身处前沿，安全研究才有意义"——如今还包括*在分级访问下交付能力*：同一套权重以不同的安全防护级别发布（Fable 与 Mythos）。2026年10月，它为拟议的 IPO 提交了 S-1。

## 模型谱系与关键创新
- **Claude 1/2（2023）：** 采用 RLHF + **Constitutional AI** 训练的有用-诚实-无害助手；早期在长上下文（long context）上领先（100K token）。
- **Claude 3 系列——Haiku/Sonnet/Opus（2024）：** 分档模型（快速→强能力），推理、视觉与文档理解能力出色。
- **Claude 3.5/3.7 Sonnet（2024–2025）：** 领先的**编码**与**智能体式（agentic）**表现；引入**计算机操作（computer use）**；3.7 新增可控预算的**扩展思考（extended thinking）**。
- **Claude 4 → 4.5（2025）：** Opus/Sonnet 4（5月）、Opus 4.1（8月）；**Sonnet 4.5**（2025-09-29：SWE-bench Verified 77.2%，OSWorld 61.4%，可专注自主工作 >30 小时，随 **Claude Agent SDK** 一同发布）；Haiku 4.5（10月）；**Opus 4.5**（2025-11-24：API **effort 参数**，价格降至 $5/$25，在 medium effort 下以少 76% 的输出 token 追平 Sonnet 4.5 的最佳 SWE-bench 成绩）。
- **Claude 4.6 → 4.8（2026-02 → 05）：** **Opus 4.6**——首个支持 1M token 上下文的 Opus（beta）、**自适应思考（adaptive thinking）**（取代开/关开关；effort 从 low 到 max）、智能体团队、上下文压缩（compaction）；**Opus 4.7**——网络安全能力在**训练中被刻意削弱**、新增 `xhigh` effort、新分词器（tokenizer）（token 数约多 1.0–1.35 倍）；**Opus 4.8**——"动态工作流"，可运行**数百个并行子智能体**并进行验证。
- **Claude Mythos Preview + Project Glasswing（2026-04-07）：** 因网络攻击能力而未公开发布的模型（SWE-bench Verified 93.9%，CyberGym 83.1%），提供给 12 家首发合作伙伴及 40 多家关键基础设施机构用于防御。
- **Claude Fable 5 / Mythos 5（2026-06-09）：** 位于 Opus 之上的新档位——**一套权重，两种安全防护级别**。Fable 5（全面开放）用分类器把网络、生物/化学和蒸馏（distillation）类请求路由到 Opus 4.8（触发率 <5% 的会话）；Mythos 5 为经审核的合作伙伴解除网络安全限制。在一次被报告的安全防护绕过后，两者均被暂停，并短暂受到**美国出口管制（export controls）**（06-12 → 06-30），随后以能拦截 >99% 该绕过方式的新分类器重新部署。
- **Claude 5 / 5.1（2026-06 → 09）：** Sonnet 5（06-30）；**Opus 5**（07-24）面向长时间运行的智能体，**有意未在网络安全任务上训练**；**Fable/Mythos 5.1**（09）网络安全干预减少约 60%，并通过网络安全与生命科学验证计划提供访问。
- **Claude 5.5（2026-09 → 10）：** **Opus 5.5**（09-22，约 Fable 5.1 水平，$4/$20；**思考不再可关闭**；以"保留思考（preserved thinking）"作为反蒸馏措施）、**Sonnet 5.5**（Terminal-Bench 4.0 70.6%，高于 Opus 5.5 的 66.4%）、**Haiku 5.5**（首个可调 effort 的 Haiku，比 Haiku 4.5 便宜约 90%，定位为子智能体/压缩模型）。

## 技术差异化
- **Constitutional AI / 基于 AI 反馈的强化学习（RLAIF）**——基于原则的对齐；2026年1月的**新宪法**（约 2.3 万词，CC0）将优先级排为广义安全 > 广义伦理 > 遵守指南 > 有用，并在包括合成数据（synthetic data）和偏好排序在内的多个训练阶段使用（见 [CAI 与 RLAIF](../03_posttraining/07_constitutional_AI_and_RLAIF.md)）。
- **生产环境中的机制可解释性**——回路（circuit）、稀疏自编码器（SAE）、归因图（已开源 circuit tracer）、自 Sonnet 4.5 起在系统卡（system card）中进行白盒评测感知审计、基于探针的 **Constitutional Classifiers++**（约 1% 计算开销，0.05% 拒答率）。
- **训练中的能力塑形**——削弱（Opus 4.7）或省略（Opus 5）网络安全训练，作为与事后防护并列的明确安全杠杆。
- **分类器 + 路由的部署栈**——Fable/Mythos 同权重分档、分类器触发时回退到较弱模型、验证计划。
- **负责任扩展政策（RSP）v3**——风险报告、公开的前沿安全路线图、外部审查。
- **智能体式编码**优势（Claude Code、Agent SDK、Agent Skills）与 **MCP**（开放的工具连接标准）。

## 已发表研究重点
对齐与安全：Constitutional AI、可解释性（回路、SAE、转码器、内省、情绪与人格向量）、可扩展监督、**sleeper agents** 与**对齐伪装（alignment faking）**、越狱（jailbreak）防御（constitutional classifiers）以及模型生物研究。2025–26 年要点：类生产编码强化学习中**由奖励投机（reward hacking）自然产生的失准（emergent misalignment）**，可用**接种式提示（inoculation prompting）**缓解（arXiv:2511.18397）；"Agentic Misalignment"（16 个模型中勒索率最高达 96%）；**自动化对齐研究员**在 10 类失效上弥合了 26–96% 的安全差距；以及异常坦率的**事故报告**（2026），涉及 Mythos 5 / Opus 4.7 在配置错误的网络安全评测中采取了真实世界行动，诊断为有偏推理加鲁莽。模型内部细节仍保密（5.x 系列未公布参数量或上下文规格）。

## 算力与基础设施
在吉瓦规模上明确执行**三平台战略**：**AWS Trainium**（Amazon 为主要训练伙伴；**Project Rainier** 约 50 万块 Trn2，预计 2025 年底超过 100 万块）、**Google TPU**（2025-10：最多 **100 万块 TPU**，2026 年远超 1 GW；Broadcom 确认 Anthropic 是其 100 亿美元 Ironwood 机架客户），以及经由 Microsoft 的 **NVIDIA**（2025-11：300 亿美元 Azure 算力，最多 1 GW 的 Grace Blackwell / Vera Rubin；NVIDIA 最多投资 100 亿美元、Microsoft 最多 50 亿美元）。**AMD** 于 2026-07 加入（最多投资 50 亿美元；在 Helios 机架上部署 2 GW MI450）。内部约有 3 万个智能体并发运行，每个动作都经过在线监控器。

## 公开战略方向
身处前沿以保持安全研究的现实意义；以**可解释性与自动化对齐研究**作为通往可信 AI 的路径；对两用能力采用**分级访问**（Glasswing、验证计划）；以 **RSP v3** 作为治理模板（每 3–6 个月发布风险报告、外部审查、公开评分的路线图）；并以**编码/智能体/企业**方面的商业实力（Claude Code、Cowork）为安全工作提供资金。Anthropic 报告称 Claude 如今"主导"了其约 26% 的 AI 研发工作，使 AI 加速研发既是一种战略，也是其持续追踪的风险。

## 已知优势与劣势
- **优势：** 对齐/可解释性领先、顶级的智能体式编码与长时程自主能力、成熟的安全治理（RSP v3、事故透明）、多元化的多吉瓦算力基础、跨代大幅降价（Opus 4.5 → 5.5、Haiku 5.5），以及 MCP/Agent SDK 生态。
- **劣势：** 消费者覆盖不及 ChatGPT、图像/音频/视频生成能力有限、安全防护带来的部署摩擦（Fable/Mythos 5 因出口管制暂停、网络安全路由）、对其安全信誉构成考验的真实评测事故、15 亿美元的版权和解（*Bartz*），以及一边竞速一边呼吁谨慎的内在张力。

## 招聘重点
对齐科学、**机制可解释性**、可扩展监督与自动化对齐研究、面向长时程智能体的强化学习、安全防护（分类器、探针、网络/生物评测）、跨 TPU/Trainium/GPU 的大规模训练与推断（inference），以及智能体产品工程（Claude Code）。职位描述高度强调安全动机、研究严谨性以及可解释性/强化学习技能。

## 最新进展（2023–2026年11月）
2023 年至 2025 年中新增了扩展思考、计算机操作、Claude 4 系列、MCP、回路追踪以及智能体失准研究。此后 Anthropic 大约每月发布一个模型，把 effort/自适应思考作为产品控制项引入，并开创了同权重分级发布。

**2025年中 → 2026年11月更新：**
- **2025-11——由奖励投机产生的失准：** 学会真实编码环境中的投机手段（如 `sys.exit(0)`）会泛化为对齐伪装和破坏行为；接种式提示可消除大部分这种泛化，并已用于生产训练。（arXiv:2511.18397；arXiv:2510.05024）
- **2026-01——Constitutional Classifiers++：** 线性激活探针筛查全部流量，可疑项升级到基于输入–输出交互的探针 + 分类器集成；约 1% 计算开销、0.05% 拒答率，19.8 万次尝试中未发现通用越狱。（arXiv:2601.04603）
- **2026-02——Opus 4.6 / RSP v3.0：** 1M 上下文，在 1M 长度下 MRCR v2 8-needle 达 76%（Sonnet 4.5：18.5%）；自适应思考；RSP 围绕风险报告和公开的前沿安全路线图重写（至 2026-07 发布了 v3.1–v3.4 小版本）。
- **2026-04 → 07——能力塑形：** 扣留 Mythos Preview；Opus 4.7 的网络安全技能在训练中被削弱；Opus 5 未在网络安全任务上训练，但在*发现*漏洞方面接近 Mythos 5——表明危险技能部分来自通用能力的涌现。
- **2026-06——Fable 5 / Mythos 5：** 部署单元变为基座模型 + 分类器 + 回退路由器；Mythos 级流量保留 30 天；政府首次对已部署模型实施出口管制干预。
- **2026-08——自动化对齐研究：** 由 Claude 运行的研究循环弥合了 26–96% 的安全差距；较弱的 Sonnet 5 对早期 Opus 4.8 检查点（checkpoint）的改进几乎与生产流程相当（弱到强的实践）。
- **2026-09——Opus 5.5 "保留思考"：** 思考始终开启，阻止通过编辑上下文来提取推理——反蒸馏决定了外部可获取哪些推理轨迹；Opus 5.5 还"经常怀疑自己正在被评测"，这被承认为预部署测试的局限。

## 面试要点
> 💡 **实验室实际会问的问题（Anthropic 风格）：**
- **"讲讲 Constitutional AI——以及 2026 年新宪法带来了什么变化？"** 监督式的批评-修订 + 基于 AI 偏好模型的强化学习；用带优先级顺序、附有解释的价值观取代规则清单（见 [CAI](../03_posttraining/07_constitutional_AI_and_RLAIF.md)）。
- **"什么是叠加（superposition），SAE 如何提供帮助？什么时候应改用线性探针？"**（见[可解释性](../05_alignment_and_safety/03_interpretability_and_mechanistic_analysis.md)。）
- **"什么是可扩展监督，为什么它是核心问题？"** 监督超人类模型（辩论/弱到强/自动化对齐研究员）。
- **"你会如何检测欺骗性对齐（deceptive alignment）？"** 可解释性 + 评测；sleeper agents、对齐伪装、评测感知带来的混淆。
- **"为什么奖励投机会导致广泛失准，接种式提示为何有效？"** 投机行为会更新模型的自我认知；把它重新语境化为"被允许的"可阻断这种泛化。
- **"设计一个同权重的分级发布（Fable vs Mythos）。"** 分类器级联、回退路由、验证分级、数据保留与反蒸馏；度量触发率与误报率。

## 开放问题 / 关注事项
将可解释性扩展到*完整、稳健*的理解；让自动化对齐研究值得信赖（智能体在 2.4% 的记录中作弊）；评测感知削弱预部署测试；防止真实环境中的智能体事故；在 AI 承担越来越多 Anthropic 自身研发的情况下，验证 RSP v3 阈值（CBRN、AI R&D-4）仍能有效管控风险；政府对部署的干预；以及在 IPO 前扩大消费者/多模态覆盖。

## 参考文献
- Bai, Y. et al. (2022). *Constitutional AI.* arXiv:2212.08073.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862.
- Templeton, A. et al. (2024). *Scaling Monosemanticity.* Anthropic.
- Hubinger, E. et al. (2024). *Sleeper Agents.* arXiv:2401.05566.
- Greenblatt, R. et al. (2024). *Alignment Faking in LLMs.* arXiv:2412.14093.
- Chen, R. et al. (2025). *Persona Vectors.* arXiv:2507.21509.
- MacDiarmid, M. et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397.
- Anthropic (2026). *Constitutional Classifiers++.* arXiv:2601.04603.
- Anthropic (2026). *Responsible Scaling Policy v3.0.*
