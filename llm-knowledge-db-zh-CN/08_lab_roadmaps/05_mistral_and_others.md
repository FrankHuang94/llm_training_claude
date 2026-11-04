# Mistral 及其他实验室（Mistral and Other Labs）

> **最后更新：** 2026-11-04
> **相关文件：** [混合专家](../06_efficiency/05_mixture_of_experts.md) · [注意力机制](../00_foundations/01_attention_mechanisms.md) · [优化器选择](../02_pretraining/03_optimizer_choices.md) · [DeepSeek](04_deepseek_roadmap.md) · [实验室对比矩阵](06_lab_comparison_matrix.md)
> **关键论文：** Jiang et al. 2023 Mistral 7B（[arXiv:2310.06825](https://arxiv.org/abs/2310.06825)）· Jiang et al. 2024 Mixtral（[arXiv:2401.04088](https://arxiv.org/abs/2401.04088)）· Kimi Team 2025 Kimi K2（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）· Kimi Team 2025 Kimi Linear（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）· Groeneveld et al. 2024 OLMo（[arXiv:2402.00838](https://arxiv.org/abs/2402.00838)）

## 概述
在"五巨头"之外，第二梯队的实验室如今主导着大量议程——并且到 2026 年已**引领开放权重（open weights）前沿**。最重要的几家是 **Mistral AI**（欧洲的前沿挑战者，于2026年10月发布了首个西方约 1T 参数的开放模型）以及**中国开放权重实验室**——**阿里巴巴 Qwen**、**月之暗面（Moonshot，Kimi）**、**智谱/Z.ai（GLM）**、**MiniMax** 和**小米 MiMo**（DeepSeek 另有[单独文件](04_deepseek_roadmap.md)）——它们的开放混合专家（MoE）模型在2025年7月至2026年8月间达到**总参数 1–2.8T**，并支持 1M token 上下文。围绕它们的还有 **xAI**、**NVIDIA（Nemotron）**、**AI2（OLMo）**、**Thinking Machines**、**Microsoft（phi/MAI）**和 **Cohere**。它们共同推动着开放模型竞争、次二次复杂度注意力、低精度训练、Muon 系优化器（optimizer）、主权 AI 以及完全开放的科学。

## Mistral AI
**概述。** 2023 年由前 DeepMind/Meta 研究人员（Mensch、Lample、Lacroix）在巴黎创立；欧洲领先的前沿实验室（frontier lab），定位于**开放权重**、**效率**和**欧洲 AI 主权**。

**模型谱系与创新。**
- **Mistral 7B（2023）：** 表现远超其规模；在一个简洁的 Apache 2.0 开放模型中推广了**滑动窗口注意力（Sliding-Window Attention，SWA）**和**分组查询注意力（GQA）**。
- **Mixtral 8×7B / 8×22B（2023–2024）：** 让**开放稀疏 MoE** 成为主流（8 选 2 的专家；47B 中约 13B 激活）。
- **2024 → 2025 年中：** Mistral Large / Small / Nemo（12B，与 NVIDIA 合作）/ **Codestral** / **Pixtral**；Mistral Small 3 与 Medium 3；**Magistral** 推理模型与 **Devstral** 编码智能体（2025）。
- **Mistral 3（2025-12-02）：** **Mistral Large 3**，一个"细粒度"稀疏 MoE，**总参数 675B / 激活 41B**（≈673B/39B 语言模型 + 2.5B 视觉编码器），256K 上下文，基座和指令版均为 Apache 2.0（已于 2026-10-31 弃用）；**Ministral 3** 有 3B/8B/14B 三种规模，各有基座、指令和推理版本。
- **Mistral Medium 3.5（2026-04-29）：** **128B 稠密**开放权重（修改版 MIT），把指令、推理和编码合并进一个检查点（checkpoint），取代 Magistral 和 Devstral 2；支持图像输入（上下文据报道为 256K 或 128K）。
- **Mistral Large 4，"Le Chonk"（预览 2026-10-06；权重 2026-10-29）：** **总参数约 1.05T / 激活 52B** 的 MoE（49B 是每 token 的专家计算量；52B 还计入了嵌入层和输出层），外加一个 1.6B 视觉编码器，采用 **Mistral Ecosystem Open-Weight License**（修改版 MIT；模型即服务（MaaS）提供商需商业许可，无营收门槛）。被称为中国以外最强的开放模型，但整体上仍落后于中国领先模型。

**差异化。** 效率优先的架构（SWA、GQA、细粒度 MoE）、将多种模式合并进单一检查点、欧洲数据主权定位，以及本地化/企业部署。**算力：** 位于欧盟的 NVIDIA 集群（2025 年中与 NVIDIA 共同宣布了"Mistral Compute"平台），规模远小于美国的吉瓦级合同。**战略：** 成为企业和政府可选的开放、主权替代方案。**优势：** 约 1T 规模的开放、效率、欧盟监管/信任定位。**劣势：** 算力预算、研究广度较窄，以及与美国闭源模型和中国开放模型两方面的差距。

## 中国开放权重实验室
- **阿里巴巴 Qwen。** 覆盖最广的开放家族：**Qwen3**（2025年4月；稠密 0.6B–32B，外加 30B-A3B 和 235B-A22B 两个 MoE，混合思考）→ **Qwen3-Next-80B-A3B**（2025-09；**Gated DeltaNet 与门控注意力 3:1** 的混合架构，512 专家 MoE，10 个路由 + 1 个共享，262K 上下文；在 32K 以上长度吞吐量（throughput）超过 Qwen3-32B 的 10 倍，训练成本不到其 1/10）→ **Qwen3.5-397B-A17B**（2026-02；GDN 混合架构，原生视觉-语言，Apache 2.0）→ **Qwen3.6**（2026-04；开放 35B-A3B 和 27B 稠密）→ **Qwen3.8-2.4T-A95B**（2026-08 开放权重；512 个专家，原生 262K 上下文，可扩展到约 1M；开放检查点仅文本且思考始终开启，采用自定义许可）。闭源的 **Qwen3-Max**（>1T，36T token）和 Qwen3.8-Max-Preview（声称"仅次于 Fable 5"）只提供托管服务。
- **月之暗面（Kimi）。** **Kimi K2**（2025-07；约 1.04T / 激活 32B，使用 **MuonClip** 训练 15.5T token——Muon + QK-clip，当最大 logit 超过阈值时按头重新缩放 Q/K 投影——零损失尖峰（loss spike））；**Kimi Linear**（2025-10；**Kimi Delta Attention**，即通道级门控的 Gated DeltaNet，与 MLA 按 3:1 交错；在 1M 长度下 KV 缓存（KV cache）最多减少 75%，解码速度提升 6 倍）；**K2 Thinking**（2025-11；在后训练（post-training）中对 MoE 权重做 **INT4 仅权重量化感知训练（QAT）**，所有基准均以 INT4 报告）；**Kimi K3**（2026-07-16 托管上线，07-27 发布权重；**总参数 2.8T / 激活 104B**，896 个专家，16 个路由 + 2 个共享，69 层 KDA + 24 层门控 MLA，块级注意力残差（Attention Residuals），原生视觉，1M 上下文，MXFP4 权重；自定义许可要求年营收超过 2000 万美元的公司进行协商）；**K3.1 Pro**（2026-10-23，API；承诺于 11-04 开放权重，尚未确认，据报道开放版本移除了网络安全训练数据）。另有用于长上下文服务的 **MoBA** 块稀疏注意力。
- **智谱 / Z.ai（GLM）。** **GLM-4.5**（2025-07；355B / 激活 32B）→ **GLM-4.6**（2025-09；200K 上下文，MIT）→ **GLM-5**（2026-02-11；**744B / 激活 40B**，**采用 DeepSeek 稀疏注意力**，28.5T token，基于 **slime** 框架的异步智能体强化学习；据报道完全在**华为昇腾 + MindSpore** 上训练，Z.ai 未确认）→ **GLM-5.1**（2026-04；后训练更新）。
- **MiniMax。** M1 的混合闪电（线性）注意力在 **MiniMax-M2**（2025-10；约 230B / 激活 10B）中被换回**全注意力**，因为 SWA 和线性变体损害了多跳推理——这是混合架构趋势的关键反例；M2.5（2026-02）；**M3**（约 2026-06；**MiniMax 稀疏注意力（MSA）**：索引器为 128-token 的块打分，每个 GQA 组保留 top-16 个块；1M 上下文；声称在 1M 长度下预填充提速 9 倍、解码提速 15 倍（相对 M2）；承诺开放权重）。
- **小米 MiMo。** **MiMo-V2-Flash**（2025-12；309B / 激活 15B，混合滑动窗口注意力）→ **MiMo-V2.5 / V2.5-Pro**（2026-04；MIT，原生视觉与音频，1M 上下文；据报道 SWA 与全局注意力为 5:1，窗口 128 token）。

## 其他重要实验室
- **xAI（Grok）。** Musk 创立（2023）；Grok 模型与 X 集成；以惊人速度建成 **Colossus** 超级集群（约 10 万→20 万块 H100）——押注算力优先；Grok 3（2025年2月）在推理上具备竞争力。后续发布（Grok 4 / 4.1、正在训练的 Grok 5）、Colossus 2 以及与 SpaceX 的公司合并均被广泛报道，但**在我们的资料中未经核实**。
- **NVIDIA——Nemotron 3。** 完全开放（权重、数据、配方）的**混合 Mamba-2/Transformer MoE** 家族：**Nano** 30B-A3B（2025-12；23 层 Mamba-2/MoE + 6 层注意力，1M 上下文）、**Super** 120B / 激活 12B（2026-03；**家族中首个以 NVFP4 预训练的模型**，约 25T token），以及 **Ultra**（约 2026 年中，在 Blackwell 上以 NVFP4 训练；最终规模未确认）。NVIDIA 的 NVFP4 论文（12B 模型以 4 bit 训练 10T token，接近 FP8）记录了这一配方。
- **AI2——OLMo。** **完全开放**的模型（权重、数据、代码、日志）；Tülu 3（开放后训练，提出了 RLVR 一词）；**OLMo 3**（2025-11；7B/32B 的 Base 和 Think 版，Dolma 3 约 6T token，Apache 2.0）和 OLMo 3.1（约 2025-12）——可复现科学的参照标杆。
- **Thinking Machines Lab。** 由 Mira Murati 于 2025 年创立；发表了颇具影响的后训练研究博客——**"LoRA Without Regret"**（2025-09：全层 LoRA 可匹敌全量微调（fine-tuning）；对强化学习而言秩 1 即足够，因为策略梯度每个回合只吸收约 1 bit 信息）和 **"On-Policy Distillation"**（2025-10：学生自行采样，教师给出逐 token 的反向 KL 奖励；以约 1/10 的强化学习 GPU 时数在 AIME24 上达到约 70%）。有报道称其推出了微调 API（Tinker），但此处未经核实。
- **Microsoft——phi / MAI。** phi-1→phi-4 证明了小模型的"教科书级"合成数据（synthetic data）论点；**MAI-1-preview**（2025-08）是 Microsoft 首个完全自研的基础模型，据报道还发布了一个推理模型（MAI-Thinking-1，2026-06）——以对冲对 OpenAI 的依赖。
- **Cohere。** 聚焦企业/检索增强生成（RAG）（Command R/R+、Command A），多语言（Aya）；面向企业而非消费者。
- **其他：** 01.AI（Yi）、百度（ERNIE）；Google 的 **Gemma 4** 和 OpenAI 的 **gpt-oss** 分别在各自实验室的文件中介绍。

## 关键挑战（共性）
- 相对于签约数十吉瓦的美国实验室存在**算力劣势**；对中国实验室而言，还有**出口管制（export controls）**以及向国产加速器的迁移。
- **与受限的美国前沿模型之间的差距**（Mythos/Fable、GPT-6 Astra）：大约一个发布周期；开放检查点有时在发布时移除了部分能力（K3.1 开放权重中的网络安全数据）。
- 开放模型的**变现**——因此出现带 MaaS 或营收例外条款的"MIT，但……"式许可（Kimi、Qwen3.8、Mistral Large 4），以及先托管、后放权重的发布方式。
- **服务万亿参数 MoE** 需要机架级 NVLink 域（NVIDIA 为 Qwen3.8 发布了 GB300 NVL72 部署指南），实际上抬高了"开放"的门槛。

## 各实验室观点（彼此差异）
- **Mistral：** 开放 + 高效 + 欧洲主权；如今达到约 1T 规模，并采用模式合并的检查点。
- **Qwen：** 覆盖所有规模的广度、线性注意力混合架构（Gated DeltaNet），以及开放 MoE 中最多的专家数。
- **Kimi：** 优化器与注意力创新（MuonClip、KDA、AttnRes），以及最大的开放模型（K3，2.8T）。
- **GLM：** 采用 DSA、异步智能体强化学习（slime），据报道使用全国产训练栈。
- **MiniMax：** 在注意力上反其道而行——M2 用全注意力，M3 用可学习的块稀疏注意力。
- **NVIDIA：** 完全开放的混合架构，用以展示 NVFP4 预训练及其硬件。
- **xAI：** 算力优先、快速扩张、与 X 集成。
- **OLMo / Thinking Machines：** 开放、可复现的科学与后训练方法。

## 最新进展（2023–2026年11月）
2023 年至 2025 年中，开放模型（Qwen、Mistral、DeepSeek、OLMo）缩小了与闭源前沿的大部分差距；**推理**扩散到开放发布中（QwQ、Magistral、R1）。xAI 的 **Colossus** 表明算力可以极快地集结起来，**Muon** 的采用（Kimi）也由此开始。

**2025年中 → 2026年11月更新：**
- **万亿参数开放 MoE（2025-07 → 2026-10）：** 最大的开放权重从 1T（Kimi K2）→ 1.6T（DeepSeek V4-Pro）→ 2.8T（Kimi K3）→ 2.4T（Qwen3.8）→ 约 1.05T（Mistral Large 4，首个西方模型）；激活参数保持在约 13–104B，因此稀疏度持续上升（K3：896 个专家中激活 16 个）。
- **次二次复杂度注意力成为中国旗舰的标配：** 约 3:1 的线性/全注意力混合（Qwen3-Next/3.5 使用 Gated DeltaNet；Kimi Linear/K3 使用 KDA + MLA）、可学习的 top-k 稀疏注意力（GLM-5 的 DSA；MiniMax M3 的 MSA），以及 SWA 混合（MiMo）——而 MiniMax M2 回归全注意力是一个反例。（arXiv:2510.26692）
- **1M token 上下文成为默认规格：** Kimi K3/K3.1、MiniMax M3、MiMo-V2.5 和 Nemotron 3 均支持 1M；卖点在于降低 KV/FLOPs，而非窗口大小本身。
- **4 bit 训练：** K2 Thinking（INT4 QAT）、K3（MXFP4 权重）、Nemotron 3 Super/Ultra（**NVFP4 预训练**；Hadamard 变换、随机舍入、末端层保留 BF16）。（arXiv:2509.25149）
- **Muon 进入生产：** MuonClip 让 K2 以零损失尖峰训练了 15.5T token；据报道 GLM-5 和 DeepSeek-V4 也使用了 Muon；稳定性工作的重心转向稀疏度（K3 的"Quantile Balancing"）。（arXiv:2507.20534）
- **发布模式：** 先托管预览，1.5–4 周后再放出权重（K3、Qwen3.8、Mistral Large 4），开放检查点有时与托管产品不同（Qwen3.8 仅文本；K3.1 移除了网络安全数据）。
- **来自小型实验室的后训练配方：** 同策略（on-policy）蒸馏与全层 LoRA（Thinking Machines），以及完全开放的强化学习流水线（OLMo 3 Think），成为标准参考。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"Mistral 7B 为什么令人印象深刻，什么是滑动窗口注意力？"** 强大的小型开放模型；局部窗口注意力的成本为 $O(n\cdot w)$，并结合 GQA（见[注意力](../00_foundations/01_attention_mechanisms.md)）。
- **"Mixtral 证明了 MoE 的什么——此后开放 MoE 又如何演进？"** 从 8 选 2 → 512–896 个细粒度专家加共享专家，激活比例约 2–5%。
- **"比较 Gated DeltaNet/KDA 混合架构与 DSA/MSA 式稀疏注意力。为什么 MiniMax M2 回到了全注意力？"** 循环状态容量 vs 可学习的 top-k 检索；多跳推理以及对低精度状态的敏感性。
- **"什么是 MuonClip / QK-clip，为什么需要它？"** 大规模使用 Muon 会导致注意力 logit 爆炸；按头重新缩放 Q/K 权重可限制最大 logit（见[优化器选择](../02_pretraining/03_optimizer_choices.md)）。
- **"你会如何用 NVFP4 进行预训练？"** 块缩放、对 Wgrad 输入做随机 Hadamard 变换、随机舍入、敏感层保持高精度。
- **"OLMo 有何不同？"** 数据 + 代码 + 权重完全开放 → 可复现的科学。

## 开放问题 / 关注事项
开放实验室能否在长时程智能体上追平受限的美国前沿模型；在 1M 上下文下约 3:1 混合架构与压缩稀疏注意力谁会胜出；万亿参数开放权重的经济性；许可碎片化（"MIT，但……"）；中国开放模型在西方部署中的地缘政治问题及其硬件来源；Mistral 能否为下一个规模台阶筹集资金；以及 xAI 的算力优先押注能否转化为可验证的前沿领先。

## 参考文献
- Jiang, A. et al. (2023). *Mistral 7B.* arXiv:2310.06825.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Bai, J. et al. (2023). *Qwen Technical Report.* arXiv:2309.16609.
- Yang, A. et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- GLM-5 Team (2026). *GLM-5.* arXiv:2602.15763.
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149.
- Groeneveld, D. et al. (2024). *OLMo.* arXiv:2402.00838.
- Abdin, M. et al. (2024). *Phi-3.* arXiv:2404.14219.
