# 知识蒸馏（Knowledge Distillation）

> **最后更新：** 2026-11-04
> **相关文件：** [合成数据生成](../01_data/04_synthetic_data_generation.md) · [投机解码](04_speculative_decoding.md) · [推理模型](../04_reasoning_and_agents/01_reasoning_models_o1_r1.md)
> **关键论文：** Hinton et al. 2015 Distilling the Knowledge in a NN（[arXiv:1503.02531](https://arxiv.org/abs/1503.02531)）· Sanh et al. 2019 DistilBERT（[arXiv:1910.01108](https://arxiv.org/abs/1910.01108)）· Kim & Rush 2016 Sequence-Level KD（[arXiv:1606.07947](https://arxiv.org/abs/1606.07947)）· Agarwal et al. 2023 GKD（同策略）（[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)）· Thinking Machines 2025 On-Policy Distillation（博客）

## 概述
知识蒸馏（knowledge distillation，KD）把一个大型、昂贵的**教师**模型的能力迁移到一个更小、更便宜、近似教师行为的**学生**模型中。KD 是业界打造"以小博大"小模型的方式（DistilBERT、TinyLlama、Gemma、DeepSeek-R1-Distill 系列），并在推理时代愈发核心：强教师（o1/R1 及其后继者）的推理轨迹被蒸馏进小模型，使其以极低的推断成本继承大部分推理能力。自 2025 年末以来，**同策略蒸馏（on-policy distillation）**本身已成为标准的后训练阶段——介于 SFT 与 RL 之间的稠密奖励中间路线——并被用于把多个领域专家合并为一个模型。KD 同样是投机解码草稿模型和端侧部署的基础。

## 核心概念
**经典（logit）蒸馏。** Hinton et al.：训练学生去匹配教师的**软概率分布**（而不仅仅是硬标签）。教师经温度 $T$ 软化后的 logit 携带了"**暗知识（dark knowledge）**"——错误类别之间的相对概率编码了硬标签所缺少的相似性结构。损失：
$$\mathcal{L}=\alpha\, T^2\, D_{KL}\!\big(p_T^{(T)}\,\|\,p_S^{(T)}\big) + (1-\alpha)\,\mathcal{L}_{CE}(y, p_S)$$
其中 $p^{(T)}=\text{softmax}(z/T)$，$T^2$ 用于重新缩放梯度。对 LLM 而言，对应的做法是在每个位置匹配教师完整的**下一个 token 分布**（前向 KL）。

**序列级 KD（Kim & Rush）。** 对于生成任务，在*教师生成的序列*上做蒸馏：生成教师输出，并用标准 CLM 损失在其上训练学生。这实质上就是**在教师生成的数据上训练**——简单、可扩展，是大多数现代 LLM 蒸馏的基础（也即"蒸馏 = 来自教师的合成数据（synthetic data）"，见[合成数据](../01_data/04_synthetic_data_generation.md)）。

**白盒与黑盒。**
- **白盒**（可获得教师 logit）：逐 token 匹配完整分布——信号更丰富，需要访问教师且共享分词器（tokenizer）。
- **黑盒**（只有 API）：在教师*输出*（文本）上训练，可附带 CoT/解释（Orca 风格）。对前沿模型的开源蒸馏大多是黑盒的。

**同策略 / 生成式 KD（GKD）。** 离线序列级 KD 的一个关键缺陷是**分布不匹配**——学生在教师轨迹上训练，推断时面对的却是*自己的*轨迹（暴露偏差）。**同策略 KD**（GKD）让学生生成序列、由教师对其*打分/纠正*（在学生自身的分布上最小化散度），从而修复不匹配并提升效果。散度的选择（前向 vs 反向 KL、JSD）很重要：反向 KL 是寻模（mode-seeking）的（学生更聚焦）。用 RL 的视角看，学生采样 $y\sim\pi_\theta$，并获得一个稠密的逐 token 奖励
$$r_t = \log \pi_T(y_t\mid y_{<t}) - \log \pi_\theta(y_t\mid y_{<t}),$$
其期望即为逐 token 反向 KL 的相反数——兼具 RL 式的同策略性和 SFT 式的稠密监督。

**推理蒸馏。** 从推理型教师中蒸馏*思维链（CoT）轨迹*。DeepSeek 表明，**把 R1 的推理能力蒸馏进小型稠密模型**（1.5B–70B）往往**胜过直接在小模型上做 RL**——这是一个重要且出人意料的结果，重塑了小型推理模型的构建方式。

## 关键挑战
- **暴露偏差 / 分布不匹配。** 离线 KD 在教师输出上训练，但学生解码的是自己的输出——降低质量（由同策略 KD 修复）。
- **容量差距。** 过小的学生无法表示大得多的教师的函数；差距过大会损害迁移（有时一个中等规模的"助教"模型会有帮助）。
- **分词器/词表不匹配。** 白盒 logit 匹配要求共享分词器；跨分词器蒸馏很难。
- **继承缺陷。** 学生会复制教师的偏见、幻觉（hallucination）和错误；上限就是教师。
- **反向 KL 的失效模式。** 寻模行为可能导致多样性坍缩，而教师信号在学生的长前缀或分布外前缀上会减弱。
- **法律/服务条款与访问。** 蒸馏闭源 API 模型可能违反服务条款并引发知识产权问题；前沿实验室如今正主动防御蒸馏。

## 解决方案与当前最佳实践
对 LLM 而言，**序列级 / 数据蒸馏**（在经过筛选/验证的高质量教师生成内容上训练）是冷启动的主力；在可获得教师 logit 时，接着做**同策略蒸馏**（学生采样，教师给出逐 token 反向 KL 奖励）——要达到教师水平，它比 RL 便宜得多，还能修复领域中期训练中丢失的能力。用**多教师 OPD** 合并多个领域专家检查点。对于**推理**，从强教师那里蒸馏**经过验证的 CoT 轨迹**（保留正确的——STaR/拒绝采样（rejection sampling））；这如今是小型推理模型的标准做法。在可获得 logit 且分词器匹配时，使用**温度/软标签**匹配。部署时把 KD 与**剪枝**和**量化**结合（小且量化的学生）。要验证学生不只是在死记硬背。

## 各实验室观点
- **Google：**DistilBERT 一脉、GKD（同策略 KD）、由更大的 Gemini 蒸馏而来的 Gemini **Flash** 模型；Gemma 蒸馏。
- **DeepSeek：****R1-Distill** 系列——把推理能力蒸馏进 Qwen/Llama 基座；主张从成本效率看蒸馏优于小模型 RL。
- **Qwen / KAT-Coder 团队：**使用多教师同策略蒸馏来合并领域专家。
- **NVIDIA：**Minitron 式的**先剪枝后蒸馏**（Nemotron Nano 2：12B → 9B）。
- **Thinking Machines Lab：**把同策略蒸馏推广为 RL 的低成本替代方案。
- **OpenAI/Anthropic：**推出蒸馏而来的"mini/haiku/flash"级模型（旗舰模型更小、更便宜的版本）；OpenAI 提供**蒸馏 API**；Anthropic 如今会把在其顶级模型上检测到的蒸馏尝试路由到较弱的模型，并限制推理轨迹的暴露。
- **Meta/开源社区：**TinyLlama、MiniLM 以及无数蒸馏出来的开放模型。

## 最新进展（2023–2026年11月）
**推理蒸馏**成为打造小型推理模型的主要方式（R1-Distill、Qwen-distill），往往胜过直接对小模型做 RL。**同策略 KD**（GKD、使用反向 KL 的 MiniLLM）解决了暴露偏差。**蒸馏缩放定律**（2024–2025）刻画了蒸馏何时优于从零训练（在教师强而学生算力有限时帮助最大）。前沿实验室把**蒸馏流水线**和小型/快速模型层级产品化为核心产品。

**2025年中 → 2026年11月更新：**
- **On-Policy Distillation（2025-10，Thinking Machines）。** 以学生采样的轨迹配合逐 token 反向 KL 的教师奖励，以约为 RL 十分之一的 GPU 时数在 AIME24 上达到约 70%，并能恢复领域中期训练中丢失的指令遵循能力。
- **OPD 成为标准阶段（2026）。** Qwen3 和 KAT-Coder-V2.5 通过多教师 OPD 合并领域专家（[arXiv:2607.05471](https://arxiv.org/abs/2607.05471)）；OPD 位于 SFT 冷启动与（可选的）RL 之间。
- **修复反向 KL 的失效模式（2026-03 → 2026-05）。** 熵门控的前向/反向 KL 切换（EOPD，[arXiv:2603.07079](https://arxiv.org/abs/2603.07079)）、截断的 top-$K$ 反向 KL（[arXiv:2603.25562](https://arxiv.org/abs/2603.25562)），以及针对长前缀上教师信号衰减的校正（[arXiv:2605.30833](https://arxiv.org/abs/2605.30833)）。
- **蒸馏为更快的解码器（2026-02）。** 在线自蒸馏把预训练的 NTP 模型转换为独立的多 token 预测解码器，报告在 GSM8K 上解码加速超过 3 倍、准确率损失低于 5%（[arXiv:2602.06019](https://arxiv.org/abs/2602.06019)）。
- **面向混合架构的剪枝 + 蒸馏（2025-08）。** Nemotron Nano 2（Mamba-2/Transformer 混合）通过 Minitron 剪枝与蒸馏从 12B 压缩到 9B（[arXiv:2508.14444](https://arxiv.org/abs/2508.14444)）。
- **蒸馏防御（2026）。** Anthropic 的 Fable 5 分类器会检测蒸馏请求并将其路由到 Opus 4.8；"保留思考（preserved thinking）"阻止 API 用户通过编辑上下文来提取推理；Anthropic 发表了关于防御蒸馏攻击的文章（2026-10）。对前沿推理轨迹做黑盒蒸馏正变得越来越难。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么用软标签 / 什么是'暗知识'？"** 教师的完整分布编码了硬标签所缺失的类间相似性；梯度信号更丰富。
- **"LLM 用序列级蒸馏还是 logit 蒸馏？"** 在教师生成内容上训练（可扩展、黑盒）vs 匹配逐 token 分布（更丰富，需要 logit + 共享词表）。
- **"什么是分布不匹配问题，同策略 KD 如何修复？"** 学生在教师轨迹上训练却解码自己的输出；GKD 在由教师打分的学生生成序列上训练。
- **"为什么把 R1 蒸馏进小模型，而不是直接对小模型做 RL？"** 经验上更便宜也更好——小模型继承了教师通过昂贵 RL 发现的推理能力。
- **"为什么同策略蒸馏达到教师水平比 RL 便宜约 10 倍？"** RL 每回合只得到约 1 个标量奖励；OPD 在学生自己的状态上得到稠密的逐 token 信号——同样的同策略性，每个样本携带的比特多得多。失效模式：寻模坍缩，以及在分布外前缀上教师信号偏弱。

## 开放问题
**容量差距**的极限（多小的学生能学到给定的教师）缺乏清晰的理论。**跨分词器**的白盒蒸馏、蒸馏**智能体/工具使用**能力，以及学生能否**超越**教师（而非被教师封顶）都是开放问题。蒸馏与 RL 的相互作用（先蒸馏后 RL vs 先 RL 后蒸馏 vs 用 OPD 取代 RL）、无干扰的多教师合并，以及反蒸馏防御对开源生态的长期影响，都在积极研究中。

## 参考文献
- Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
- Sanh, V. et al. (2019). *DistilBERT.* arXiv:1910.01108.
- Kim, Y., Rush, A. (2016). *Sequence-Level Knowledge Distillation.* arXiv:1606.07947.
- Agarwal, R. et al. (2023). *GKD (On-Policy Distillation).* arXiv:2306.13649.
- DeepSeek-AI (2025). *DeepSeek-R1 (distillation).* arXiv:2501.12948.
- Thinking Machines Lab (2025). *On-Policy Distillation.* Blog post.
- NVIDIA (2025). *NVIDIA Nemotron Nano 2.* arXiv:2508.14444.
