# DeepSeek——实验室路线图（DeepSeek — Lab Roadmap）

> **最后更新：** 2026-11-04
> **相关文件：** [GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [混合专家](../06_efficiency/05_mixture_of_experts.md) · [混合精度与低精度训练](../02_pretraining/04_mixed_precision_and_quantization.md) · [注意力机制](../00_foundations/01_attention_mechanisms.md)
> **关键论文：** DeepSeek-AI 2024 DeepSeek-V3（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）· DeepSeek-AI 2025 DeepSeek-R1（[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）· DeepSeek-AI 2024 DeepSeek-V2（MLA）（[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)）· Shao et al. 2024 DeepSeekMath/GRPO（[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)）· DeepSeek-AI 2025 DeepSeek-V3.2（DSA）（[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）

## 概述
DeepSeek 是一家中国实验室（脱胎于量化基金幻方 High-Flyer），在 2024 年末至 2025 年成为业界最大的颠覆者：它发布的**开放权重（open weights）**模型以**极低的训练成本**达到或接近前沿（GPT-4o / o1）水平，并公开了详尽的方法。**DeepSeek-V3**（671B 混合专家（MoE），据报道最终训练成本约 560 万美元）和 **DeepSeek-R1**（可与 o1 匹敌的开放推理模型）引发了"DeepSeek 时刻"（2025年1月）。此后 DeepSeek 不再那么追求重磅发布，而是专注于**百万 token 上下文的注意力效率**：混合思考（V3.1）、可学习稀疏注意力（sparse attention）（**DSA**，V3.2），以及 **DeepSeek-V4** 中的压缩稀疏注意力（2026年4月；总参数 1.6T / 激活 49B，1M 上下文，MIT）。在**出口管制（export controls）带来的硬件约束**下，它仍是"效率胜过算力"的标杆，尽管到 2026 年它已与 Qwen、Kimi 和 GLM 共享中国开放权重前沿。

## 模型谱系与关键创新
- **DeepSeek LLM / Coder / Math（2023–2024）：** 强大的开放基座/代码/数学模型；**DeepSeekMath** 提出了 **GRPO**（见 [GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md)）。
- **DeepSeek-V2（2024）：** **多头潜在注意力（Multi-head Latent Attention，MLA）**——低秩 KV 压缩——外加 **DeepSeekMoE**（细粒度 + 共享专家）。
- **DeepSeek-V3（2024）：** 671B 参数的 **MoE**（激活 37B），以 **FP8** 训练约 14.8T token，采用 **DualPipe**、**无辅助损失负载均衡（load balancing）**和**多 token 预测（MTP）**。
- **DeepSeek-R1 / R1-Zero（2025-01）：** 通过**带可验证奖励的纯强化学习（GRPO）**训练的开放**推理**模型；R1-Zero 表明推理能力*可在无 SFT 的情况下从强化学习中涌现*；并蒸馏到较小的稠密模型（dense model）中。
- **NSA（2025-02）：** 原生稀疏注意力（Native Sparse Attention）——硬件对齐、可训练的稀疏注意力，包含压缩/选择/滑动三路分支（ACL 2025 最佳论文）。
- **DeepSeek-V3.1（2025-08-21，MIT）：** 一个**带 Think / Non-Think 两种模式的混合模型**（取代独立的 R1 产品线）；**UE8M0 FP8** 缩放格式（仅指数、2 的幂次块缩放因子），"为下一代国产芯片量身定制"。
- **DeepSeek-V3.2-Exp（2025-09-29）→ V3.2 / V3.2-Speciale（2025-12-01）：** **DeepSeek 稀疏注意力（DSA）**——一个 FP8 "闪电索引器（lightning indexer）"为历史 token 打分，MLA 只关注 top-k；训练配置与 V3.1-Terminus 对齐以隔离 DSA 的影响（质量大致持平），API 价格降低 50% 以上。Speciale 声称在 IMO、CMO、ICPC 世界总决赛和 IOI 2025 上达到金牌水平。
- **DeepSeekMath-V2（2025-11）：** 训练一个 LLM 证明验证器，作为生成器的奖励，激励生成器发现并修正自身错误；IMO 2025 金牌水平（5/6），Putnam 2024 118/120（自报）。
- **mHC（2025-12）：** 流形约束超连接（Manifold-Constrained Hyper-Connections）——拓宽残差流，并把混合矩阵投影到双随机矩阵上（Sinkhorn-Knopp），恢复稳定的信号传播；据报道用于 V4。
- **DeepSeek-V4 Preview（2026-04-24，MIT）：** **V4-Pro 总参数 1.6T / 激活 49B** 和 **V4-Flash 284B / 激活 13B**，均为 1M 上下文，具备 Thinking/Non-Thinking 模式。V4-Flash 于 2026-07-31 正式发布，V4-Pro 约在 8 月中旬正式发布。
- **DeepSeek-V4.1-Flash（2026-09-10，MIT）：** **552B** MoE，具备**原生视觉**（文本+图像输入），1M 上下文；取代 V4-Flash 和视觉实验版，同时 V4-Pro 正在逐步下线。**R2 尚未发布**。

## 技术差异化
- **MLA → DSA → CSA/HCA**——一条持续降低 KV 与 FLOPs 的注意力设计路线；DSA 已扩散到 DeepSeek 之外（GLM-5 采用了它）。
- **DeepSeekMoE**——细粒度 + 共享专家，**无辅助损失**均衡。
- **GRPO + 可验证奖励强化学习（RLVR）**——无评论家（critic）的推理强化学习；o1 级推理的开放蓝图，并扩展到自我验证（DeepSeekMath-V2）。
- **极致训练效率与低精度**——FP8（V3）、面向国产芯片的 UE8M0 FP8（V3.1）、对路由专家进行 **FP4 量化感知训练**（V4），以及以 **Muon** 作为主优化器（optimizer）（V4）。
- **彻底的开放**——MIT 许可的权重 + 异常详尽的技术报告和开源算子（FlashMLA、DeepGEMM）。

## 已发表研究重点
高效注意力（MLA、NSA、DSA、CSA/HCA）、MoE、低精度训练（FP8、FP4 QAT）、系统（DualPipe、算子）、推理强化学习（GRPO、RLVR、R1）、自我验证的数学（DeepSeekMath-V2）、残差流架构（mHC）、多 token 预测以及蒸馏（distillation）。其报告详尽到可被广泛复现——MTP 模块和 DSA 式索引器如今已成为其他中国开放模型的标配。

## 算力与基础设施
在**美国出口管制**下运营，历史上使用互联带宽受限的 **NVIDIA H800**（及更早的 A100）GPU——正是这一约束*倒逼*出 FP8、DualPipe、MLA 和通信重叠。自 2025 年起，它明确**为国产加速器协同设计数值格式**（UE8M0 FP8）。V4 的训练硬件**存在争议**（说法从华为昇腾，到 NVIDIA 训练加昇腾推断，再到美国官员指控使用走私的 Blackwell）；DeepSeek 均未确认。背后有幻方的 GPU 储备支持。

## 公开战略方向
推动**开放**、**高效**的前沿模型；通过算法/系统效率而非蛮力算力追求 AGI；让**百万 token 上下文变得负担得起**（据报道，V4 在 1M 长度下每 token FLOPs 约为 V3.2 的 27%，KV 缓存（KV cache）约为 10%）；并以 MIT 许可公开分享方法。其定位是证明前沿并不仅仅取决于算力预算——这一叙事对"算力护城河"论断具有重大的地缘政治意义。

## 已知优势与劣势
- **优势：** 注意力效率领先、成本/质量比、开放的推理与数学能力（Speciale、Math-V2）、架构创新（MLA、DSA、CSA/HCA、mHC）、宽松的 MIT 许可、可复现的透明度。
- **劣势：** 硬件获取受出口管制限制（且来源不明）、2026 年发布节奏慢于 Qwen/Kimi 并失去"唯一开放前沿"的地位、多模态起步晚（直到 V4.1 才有原生视觉）、安全研究存在感较弱，以及地缘政治/监管阻力（在部分司法辖区受到禁令/审查）。

## 公开招聘重点
（根据产出和报告推断）系统/效率工程（低精度、并行、算子）、注意力与 MoE 架构、推理强化学习（GRPO/RLVR、验证器），以及数据/数学/代码整理。强调用受限算力做更多事——一种独特的工程文化。

## 最新进展（2023–2026年11月）
2023 年至 2025 年中产出了 GRPO、MLA、DeepSeekMoE、V3 的 FP8 配方、R1 和 NSA。此后 DeepSeek 集中精力降低长上下文成本，并为国产硬件做设计。

**2025年中 → 2026年11月更新：**
- **2025-08——V3.1 混合思考 + UE8M0 FP8：** 一个检查点（checkpoint）同时服务两种模式（API 中分别为 `deepseek-chat` 与 `deepseek-reasoner`）；仅指数的块缩放因子牺牲了缩放因子中的尾数精度，换取对国产芯片更简单的硬件实现。
- **2025-09/12——DSA：** 闪电索引器（少量头、ReLU、FP8）为 MLA 选出 top-k token；分两阶段训练（先稠密预热，用 KL 使索引器与主注意力分布对齐，再进行稀疏持续训练）。索引器仍是 $O(L^2)$ 但开销很小，因此主注意力成本变为 $O(Lk)$。（arXiv:2512.02556）
- **2025-11——DeepSeekMath-V2：** 以验证器作为奖励，并对难证明扩展验证计算——把 RLVR 推广到没有廉价真值检查器的领域。（arXiv:2511.22570）
- **2025-12——mHC：** 把超连接的混合矩阵投影到 Birkhoff 多面体上，恢复了普通超连接所破坏的近似恒等映射；在 27B 规模上稳定。（arXiv:2512.24880）
- **2026-04——V4（CSA + HCA）：** CSA 把每个 KV 块压缩成一个条目，再由索引器选出 top-k 压缩条目；HCA 则在压缩程度高得多的 KV 上做稠密注意力。据报道，在先以稠密方式完成 4K→16K 训练后，于 64K 阶段开启稀疏注意力；使用 Muon 优化器（嵌入/输出头/归一化层仍用 AdamW）；对路由专家做 FP4 QAT；约 32T token（二手来源）。报告编号列为 arXiv:2606.19348（HF 页面上的编号/日期不一致——需核实）。
- **2026-09——V4.1-Flash：** 一个带原生视觉的 552B 模型同时取代 V4-Flash 和 V4-Pro——收敛为一个高效的多模态主力模型，而非旗舰阶梯。

## 面试要点
> 💡 **实验室实际会问的问题（与 DeepSeek 相关）：**
- **"解释 GRPO 及其与 PPO 的区别。"** 无评论家、组内相对归一化优势（见 [GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md)）。
- **"什么是 MLA，它为何有用？"** 低秩潜在 KV 压缩 → KV 缓存很小，质量接近 MHA。
- **"DeepSeek-V3 为何能如此低成本训练？"** 在 H800 上结合 FP8 + DualPipe + 无辅助损失 MoE + MTP + 通信重叠。
- **"R1-Zero 证明了什么？"** 推理能力可以在无 SFT 的情况下，从带可验证奖励的纯强化学习中涌现。
- **"讲讲 DSA 和 V4 的 CSA/HCA。在 1M token 下成本花在哪里？"** 索引器是二次的但很便宜；主注意力作用于 top-k（压缩）条目；KV 缓存大小和索引器 K 缓存主导服务端设计。
- **"实验室为什么会选择 UE8M0（2 的幂）缩放因子？"** 在硬件中应用缩放因子成本低、与国产加速器兼容，代价是缩放更粗糙。

## 开放问题 / 关注事项
DeepSeek 能否在出口管制下跟上下一个规模台阶（并澄清其硬件来源）；CSA/HCA 式压缩在 1M 上下文下能否保住多跳推理能力；R2 或下一代推理旗舰；把效率优势扩展到多模态智能体（V4.1）；建立安全研究形象；以及在美国实验室签下数十吉瓦算力的情况下，"效率胜过算力"是否依然成立。

## 参考文献
- DeepSeek-AI (2024). *DeepSeek-V3.* arXiv:2412.19437.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- DeepSeek-AI (2024). *DeepSeek-V2 (MLA).* arXiv:2405.04434.
- Shao, Z. et al. (2024). *DeepSeekMath (GRPO).* arXiv:2402.03300.
- Yuan, J. et al. (2025). *Native Sparse Attention.* arXiv:2502.11089.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
- DeepSeek-AI (2025). *DeepSeekMath-V2.* arXiv:2511.22570.
- DeepSeek-AI (2025). *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880.
- DeepSeek-AI (2026). *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* arXiv:2606.19348 (verify ID).
