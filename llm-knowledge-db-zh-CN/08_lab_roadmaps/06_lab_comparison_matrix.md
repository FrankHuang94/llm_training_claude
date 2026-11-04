# 实验室对比矩阵（Lab Comparison Matrix）

> **最后更新：** 2026-11-04
> **相关文件：** [OpenAI](00_openai_roadmap.md) · [Anthropic](01_anthropic_roadmap.md) · [Google DeepMind](02_google_deepmind_roadmap.md) · [Meta](03_meta_ai_roadmap.md) · [DeepSeek](04_deepseek_roadmap.md) · [Mistral 及其他](05_mistral_and_others.md)

本页提炼了各大前沿实验室（frontier lab）在**技术理解**和**面试针对性准备**两方面都至关重要的维度上的差异。可用它快速对比各家路线，并组织"为什么选择这家实验室？"的回答。截至2026年11月，格局相比2025年初已发生重大变化：Meta 已将其前沿权重闭源，开放权重（open weights）前沿由中国实验室引领，三家美国闭源实验室都把最危险的能力置于分级访问之后，算力则以吉瓦为单位、跨多家芯片供应商签约。

---

## 总对比表

| 维度 | **OpenAI** | **Anthropic** | **Google DeepMind** | **Meta AI** | **DeepSeek** | **中国开放权重实验室（Qwen / Kimi / GLM）** | **Mistral** |
|---|---|---|---|---|---|---|---|
| **旗舰模型（2026年11月）** | GPT-6 Astra；GPT-6 / 6.1 Sol、Luna；gpt-oss（开放） | Opus / Sonnet / Haiku 5.5；Fable 5.1 / Mythos 5.1（Mythos 级） | Gemini 3.1 Pro、3.5 / 3.6 Flash；Gemini 4 处于后训练阶段；Gemma 4（开放） | Muse Spark、Muse Wonder / Full（闭源）；Llama 4（遗留开放模型） | V4.1-Flash（552B，视觉）；V4-Pro（1.6T，逐步下线） | Qwen3.8-2.4T-A95B；Kimi K3（2.8T）/ K3.1 Pro；GLM-5 / 5.1（744B） | Large 4（约 1.05T，开放）；Medium 3.5（128B 稠密） |
| **架构选择** | 细节闭源；gpt-oss：top-4 混合专家（MoE）、滑动/稠密交替 + 注意力汇、MXFP4 | 细节闭源；1M 上下文（Opus 4.6 起） | **原生多模态（multimodal）**、MoE、长上下文（long context）；Gemma 4 MoE + PLE 端侧模型 | Llama：RMSNorm/RoPE/SwiGLU → MoE（L4）；Muse 未披露 | **MLA → DSA → CSA/HCA** + 细粒度 MoE + MTP；mHC | 线性混合架构（**Gated DeltaNet**、**KDA**，约 3:1）或 **DSA**；512–896 个专家 | 细粒度 MoE（Large 3/4）；128B 稠密；SWA 传统 |
| **训练数据规模** | 未披露；Astra 被定位为最大规模的预训练 | 未披露；由宪法驱动的合成数据 | 海量多模态（文本/图像/音频/视频） | 约 15T（Llama 3）；Muse 未披露 | 约 14.8T（V3）；据报道约 32T（V4） | 36T（Qwen3-Max）；15.5T（K2）；28.5T（GLM-5） | 未披露 |
| **后训练（post-training）方法** | RLHF 开创者 → 大规模**面向长时程智能体的强化学习（RL）**；路由式思考（GPT-5） | **CAI / RLAIF** + 面向智能体的强化学习；接种式提示；自适应/始终开启的思考 | RLHF + RLAIF + 推理强化学习；Deep Think 并行推理 | SFT + 拒绝采样 + RLHF/DPO（Llama）；Muse 未披露 | **GRPO + RLVR**；混合思考/非思考；以验证器为奖励（Math-V2） | GRPO 系强化学习；异步智能体强化学习（GLM slime）；后训练中的 INT4 QAT（K2 Thinking） | 指令 + 推理 + 代码合并为单一检查点 |
| **开放 vs 闭源** | 前沿**闭源**；**gpt-oss 开放**（Apache 2.0） | **闭源** | **混合**：Gemini 闭源，**Gemma 4 采用 Apache 2.0** | **自2026年4月起前沿闭源**（Muse）；遗留 Llama 开放 | **开放**（MIT）+ 详尽报告 | **开放**，"MIT，但……"式自定义许可；部分旗舰仅托管（Qwen3-Max） | **开放**（Apache 2.0 / 修改版 MIT，含 MaaS 例外条款）+ API |
| **受限 / 分级发布** | **Daybreak** Blue / Red；GPT-5.6-Cyber；Astra 网络安全"Critical" | **Fable vs Mythos**（同一权重）；Glasswing；网络安全与生命科学验证计划；分类器路由 | **3.5 Flash Cyber** 面向政府/可信合作伙伴 | 面向合作伙伴的私有预览 API | 无（开放权重） | 开放检查点中移除部分能力（K3.1 网络安全数据）；先托管 | 未见报道 |
| **安全方法论** | 准备度框架 v2（High / Critical）、CoT 可监控性评测、反图谋、自白 | **RSP v3**（风险报告、公开路线图）、CAI、可解释性、CC++ 分类器、训练中的能力塑形 | **FSF v3**（操纵类 CCL、抗关停；据报道 v3.1 新增跟踪等级）、受限网络安全模型 | Llama Guard / Purple Llama；拒签欧盟 GPAI 准则 | 标准安全微调；公开安全形象较弱 | 标准安全微调；开放版本中移除数据 | 标准安全微调；企业级管控 |
| **关键技术创新** | RLHF、缩放定律（scaling law）、**推断时（inference-time）缩放**、模型**路由器**、原生计算机操作（5.4） | **Constitutional AI**、SAE / 归因图、MCP、effort 控制、同权重分级 | **Transformer**、**Chinchilla**、原生多模态、Genie 3 世界模型、TPU 8t / 8i | **LLaMA** 开放配方、过度训练、**PyTorch / FSDP**、BLT | **MLA**、**GRPO**、FP8、DualPipe、**DSA**、CSA/HCA、UE8M0 | **MuonClip**、KDA、AttnRes（Kimi）；Qwen3-Next 混合架构；slime（GLM） | SWA、开放 MoE（Mixtral）、首个西方约 1T 开放 MoE |
| **推断时计算（inference-time compute）理念** | 快速 vs 思考路由；以单任务 token 数为指标 | **effort 等级**（low → max）；思考始终开启（Opus 5.5） | 思考 + Deep Think；Flash 档效率 | 未披露（Muse） | 单一检查点中的混合思考/非思考 | 思考模式；Qwen3.8 开放权重思考始终开启 | 推理并入通用检查点 |
| **算力 / 硬件** | NVIDIA、AMD（MI450）与 Broadcom 定制芯片共约 26 GW；Stargate；Cerebras 用于低延迟 | **TPU**（最多 100 万块）+ **Trainium**（Rainier）+ NVIDIA / Azure + AMD（2 GW），各自均达 ≥1 GW 级 | 自研 **TPU**（Ironwood 全面上市；8t / 8i）+ JAX；对外销售 TPU | NVIDIA + 最多 **6 GW AMD**；吉瓦级园区 | H800 遗留；面向国产芯片的数值格式；V4 硬件存疑 | 大多未披露；据报道 GLM-5 全部基于昇腾；Qwen3.8 在 GB300 NVL72 上服务 | 欧盟 NVIDIA 集群（预算较小） |
| **分发 / 护城河** | ChatGPT 品牌 + Codex + API | 企业 / 智能体式编码（Claude Code）+ 安全信誉 | Google 产品面（Search AI Mode / Android / Cloud）+ TPU | Meta 应用 + 设备 + PyTorch | 开放 + 效率/成本叙事 | 开放生态的无处不在；价格 | 欧盟主权 + 企业/本地化部署 |
| **主要短板** | 闭源；治理与资本密集 | 部署摩擦；评测事故；消费者覆盖小 | 交付不稳（3.5 Pro 推迟）；重组 | 失去开放领导地位；反复重组；闭源前沿尚未证明自己 | 硬件天花板；2026 年节奏放缓 | 出口管制；许可碎片化；在西方的信任问题 | 算力预算；落后于中国开放模型 |

---

## 战略原型（一句话概括）

- **OpenAI——"前沿 + 产品，全面扩展。"** 重新扩展预训练*与*长时程强化学习、吉瓦级多供应商算力、快速分级发布、对 Critical 级能力实行可信访问。
- **Anthropic——"以前沿为手段实现安全，并分级交付。"** 在对齐/可解释性和智能体式编码上领先；同一权重以不同安全防护级别发布。
- **Google DeepMind——"广度 + 垂直整合。"** 原生多模态、TPU（如今也是商品）、Flash 档效率、世界模型与机器人。
- **Meta——"从以开放实现商品化，转向闭源超级智能。"** Muse 标志着向闭源前沿模型的转向。
- **DeepSeek——"效率胜过算力。"** 开放、MIT 许可、在硬件约束下进行注意力效率创新。
- **中国开放权重实验室——"万亿规模的开放前沿。"** 快速迭代、极致稀疏、次二次复杂度注意力、4 bit 训练、自定义许可。
- **Mistral——"开放、高效、欧洲。"** 主权 + 开放，如今达到约 1T 规模。

## 分歧轴（各实验室真正的分歧所在）

1. **开放权重 vs 闭源权重。** 开放前沿：DeepSeek、Qwen/Kimi/GLM/MiniMax、Mistral；选择性开放：OpenAI（gpt-oss）、Google（Gemma）；闭源：Anthropic，以及自 2026 年起的 **Meta**。押注于生态/安全审视，还是控制、变现、防滥用与防蒸馏（distillation）。
2. **谁能拿到最强的模型。** 全面开放 vs **受限分级**（Mythos、Daybreak Red、Flash Cyber）vs 移除部分能力后的开放权重——部署如今是基座模型 + 分类器 + 路由 + 身份验证。
3. **算力至上 vs 效率。** OpenAI/Anthropic/Google/Meta 签下**多吉瓦、多供应商**的算力；DeepSeek 和中国实验室则推动稀疏化、压缩注意力和 4 bit 格式，且越来越多地为国产芯片而设计。
4. **安全姿态。** Anthropic（RSP v3、能力塑形、事故披露）↔ OpenAI/Google（准备度 / FSF 阈值 + 受限发布）↔ Meta（拒签欧盟准则；转向闭源后安全相关发表减少）↔ 开放权重实验室（公开安全研究较少、移除数据）。
5. **长上下文的注意力方案。** 稠密 + 未披露的技巧（美国闭源）vs **可学习稀疏**（DSA/CSA、MSA）vs **线性混合**（Gated DeltaNet、KDA）vs **全注意力**（MiniMax M2 的论点）。
6. **推理过程的暴露程度。** 可见或摘要化的思维链（CoT）vs **保留/绑定账户的思考**与蒸馏分类器（Anthropic）、被路由器隐藏的推理深度（OpenAI）vs 完全公开的轨迹（开放模型）。
7. **后训练方法。** 各家都在做面向长时程智能体的强化学习，但侧重点不同：CAI/宪法（Anthropic）、路由式推理（OpenAI）、GRPO + 验证器（DeepSeek）、异步智能体强化学习（GLM）。

## 最新进展（2023–2026年11月）
截至 2025 年中，这一矩阵围绕开放（Meta、DeepSeek、Mistral）vs 闭源（OpenAI、Anthropic）vs 混合（Google），以及谁拥有推理模型来组织。此后这些边界都已移动。

**2025年中 → 2026年11月更新：**
- **2025-08 → 2026-04——开放/闭源格局重组：** OpenAI 发布 gpt-oss（Apache 2.0），Google 把 Gemma 4 改为 Apache 2.0，Meta 则以 Muse Spark 闭源其前沿——开放前沿留给了 DeepSeek、Qwen、Kimi、GLM 和 Mistral。
- **2026-04 → 09——分级发布成为标准：** Mythos Preview/Glasswing、Fable vs Mythos 5、Gemini 3.5 Flash Cyber、Daybreak，以及 GPT-6 Astra 的网络安全"Critical"评级；能力也在*训练中*被塑形（Opus 4.7/5）。
- **2025-09 → 2026-07——吉瓦级多供应商算力：** OpenAI 约 26 GW（NVIDIA/AMD/Broadcom），Anthropic 横跨 TPU/Trainium/NVIDIA/AMD，Meta 6 GW AMD；TPU 对外销售；芯片拆分为训练与推断专用（TPU 8t/8i、Rubin CPX）。
- **2025-07 → 2026-10——1–2.8T 的开放 MoE：** 次二次复杂度注意力、1M 上下文和 4 bit 训练成为中国旗舰的默认配置；Mistral Large 4 是首个西方约 1T 开放模型。
- **节奏与价格：** 每家实验室约每 4–8 周发布一次；分档家族（Sol/Terra/Luna、Opus/Sonnet/Haiku、Pro/Flash/Lite）在能力不变时价格下降 40–90%。
- **治理：** 美国出口管制曾短暂下架 Fable/Mythos 5（2026-06）；欧盟 AI 法案的 GPAI 义务已生效（Meta 拒签行为准则）；安全框架从阈值转向定期报告（RSP v3 风险报告、FSF 跟踪等级）。

## 如何用于"为什么选择这家实验室？"的面试回答

- **Anthropic：** 对齐/可解释性、CAI 与宪法、自动化对齐研究、安全防护工程（分类器、分级发布）、长时程智能体式编码。
- **OpenAI：** 大规模强化学习/后训练、推理与路由、计算机操作智能体、可监控性、超大规模交付。
- **Google DeepMind：** TPU/JAX、多模态、世界模型与机器人、强化学习/搜索、科学推理。
- **Meta：** 大规模系统/PyTorch、多模态、智能体记忆、在强大基础设施之上打造闭源前沿产品线。
- **DeepSeek / 中国开放权重实验室 / Mistral：** 效率（稀疏与线性注意力、MoE、FP8/FP4、Muon）、开放可复现性、万亿参数 MoE 的服务部署。

## 注意事项

本矩阵是截至2026年11月的**战略抽象**。闭源实验室（OpenAI、Anthropic、Gemini 与 Muse 的内部细节）的模型细节是根据公开报道和系统卡（system card）推断的，可能不够精确；2026 年的许多基准与定价数据为厂商自报，部分事项（如 GLM-5 完全基于昇腾训练、DeepSeek V4 的硬件、Kimi K3.1 开放权重）有报道但未经确认。各实验室会随时间趋同——如今几乎都在做 MoE、长时程强化学习、1M 上下文和分档模型家族——因此请把各列视为*侧重点与渊源*，而非僵硬的边界。
