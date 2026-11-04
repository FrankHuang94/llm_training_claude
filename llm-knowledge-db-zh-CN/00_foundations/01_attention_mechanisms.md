# 注意力机制（Attention Mechanisms）

> **最后更新：** 2026-11-04
> **相关文件：** [Transformer 架构](00_transformer_architecture.md) · [上下文窗口扩展](../06_efficiency/06_context_window_extension.md) · [计算与内存效率](../02_pretraining/05_compute_and_memory_efficiency.md)
> **关键论文：** Dao et al. 2022 FlashAttention（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）· Dao 2023 FlashAttention-2（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）· Shazeer 2019 MQA（[arXiv:1911.02150](https://arxiv.org/abs/1911.02150)）· Ainslie et al. 2023 GQA（[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)）· Liu et al. 2023 Ring Attention（[arXiv:2310.01889](https://arxiv.org/abs/2310.01889)）

## 概述
自注意力（self-attention）的表达力是有代价的：时间与内存随序列长度 $n$ 按 $O(n^2)$ 增长。随着上下文窗口（context window）从 512（BERT）增长到 128K–10M token（2024–2026），注意力成为最主要的系统瓶颈——既体现在训练时的分数矩阵上，也体现在自回归推断（inference）时的键/值（KV）缓存上。业界的应对分为两条互补的路线：一是**精确的、IO 感知的内核**（FlashAttention），数学上完全不变，只是避免在慢速内存中物化 $n\times n$ 矩阵；二是**架构层面的近似**（MQA/GQA、滑动窗口、线性/稀疏注意力），通过改变计算内容来降低内存或 FLOPs。

理解这些机制至关重要，因为决定一个长上下文（long context）模型能否训练、能否部署的，往往是内存墙而不是原始 FLOPs。尤其是 KV 缓存，在大规模下主导了推断的内存与延迟（latency）。

## 核心概念
**FlashAttention（IO 感知）。** 核心洞见是：标准注意力是*内存受限*而非计算受限的——在 GPU HBM 上读写 $n\times n$ 分数矩阵占据了大部分实际耗时。FlashAttention 把 $Q,K,V$ 切成能放入高速 SRAM 的块，并用**在线 softmax** 计算注意力——维护滑动最大值 $m$ 和归一化因子 $\ell$，从而在单次流式遍历中算出输出，而无需存储完整的分数矩阵：
$$\ell^{(j)} = e^{m^{(j-1)}-m^{(j)}}\ell^{(j-1)} + \sum e^{S_{ij}-m^{(j)}}$$
这把内存从 $O(n^2)$ 降到 $O(n)$，并带来 2–4 倍加速。**FlashAttention-2** 改进了并行性和工作划分（更高的 GPU 占用率、更少的非矩阵乘 FLOPs）；**FlashAttention-3** 利用 Hopper（H100）的异步/FP8 特性获得进一步提升。

**多查询注意力（MQA）。** 所有头共享同一个键头和值头（$h$ 个查询头，1 个 KV 头），使 KV 缓存缩小 $h$ 倍。这能大幅提升解码吞吐量（throughput），但可能降低质量并使训练不稳定。

**分组查询注意力（GQA）。** 介于两者之间：$g$ 个 KV 头由若干组查询头共享（$1 < g < h$）。GQA 只用一小部分 KV 缓存就几乎恢复了 MHA 的全部质量（LLaMA-2 70B、LLaMA-3、Mistral 均采用）。KV 缓存大小为 $2\cdot g\cdot d_k\cdot n\cdot \text{layers}$，因此把 $g$ 从 $h$ 降到例如 8 能节省大量内存。

**滑动窗口注意力（SWA）。** 每个 token 只关注之前的 $w$ 个 token（Mistral 使用 $w=4096$）。堆叠 $L$ 层后，有效感受野为 $L\cdot w$。每层开销降为 $O(n\cdot w)$。

**Ring Attention。** 把序列按环形分布到多个设备上；每个设备持有一块 Q，并让 K/V 块沿环流转，使通信与计算重叠，从而实现几乎无限的上下文，只受设备总内存限制。

**线性注意力与 delta 规则。** 去掉 softmax 后，注意力可以写成在矩阵值状态 $S_t\in\mathbb{R}^{d_v\times d_k}$ 上的递推，每步内存为 $O(1)$。加性线性注意力（linear attention）（$S_t=S_{t-1}+v_tk_t^\top$）只会不断累加，召回能力随之退化。**Gated DeltaNet** 则采用覆写：
$$S_t=\alpha_t\,S_{t-1}\big(I-\beta_t k_tk_t^\top\big)+\beta_t\,v_tk_t^\top,\qquad o_t=S_tq_t$$
其中 $\alpha_t$ 是衰减门，$\beta_t$ 是写入强度——这相当于一步在线回归，替换掉存放在键 $k_t$ 之下的值，使状态保持有界。Kimi Delta Attention（KDA）把衰减从按头改为按通道。

**可学习的 top-$k$ 稀疏注意力。** 一个廉价的打分器为每个查询挑选要关注的历史 token（或 KV 块）；主注意力只在被选中的集合上运行。DeepSeek Sparse Attention（DSA）使用一个少头、FP8、ReLU 的"闪电索引器（lightning indexer）"——仍是 $O(L^2)$，但常数极小——并在 top-$k$ token 上运行 MLA。

## 关键挑战
- **推断时的 KV 缓存内存。** 随上下文 × 批大小 × 层数线性增长；常常是部署长上下文模型时的硬约束。
- **质量与效率的权衡。** MQA/SWA 节省内存，但可能损害长距离召回；GQA 是务实的折中。
- **硬件相关的内核。** FlashAttention 需要针对每一代 GPU 重新调优；移植到 TPU/AMD 并不容易。
- **长程信息丢失。** 滑动窗口和稀疏模式有可能丢掉真正的长程依赖。

## 解决方案与当前最佳实践
稠密模型（dense model）的默认方案仍是 **FlashAttention-2/3 内核 + GQA**，长上下文则配合**滑动窗口或全局/局部混合**模式。对于 2026 年支持 1M token 的开放旗舰模型，选择集中在两条路线之间：(a) **约 3:1 的线性:全注意力混合**（Gated DeltaNet/KDA + 门控注意力或 MLA）；(b) **在（压缩后的）KV 上做可学习的 top-$k$ 稀疏注意力**（DSA → CSA/HCA、MiniMax MSA），通常与 MLA 结合。推断栈（vLLM、TensorRT-LLM）引入了 **PagedAttention**——以类似虚拟内存分页的方式管理 KV 缓存，消除碎片并支持高吞吐批处理。对 KV 缓存做量化（quantization）（FP8/INT8）日益成为标配。DeepSeek 的 **MLA** 把 K/V 压缩为低秩潜变量，以接近 GQA 的内存获得接近 MHA 的质量。

## 各实验室观点
- **Google** 提出了 MQA（Shazeer 2019），并在 TPU 部署中采用激进的 KV 削减；PaLM 使用了 MQA。
- **Meta/Mistral** 在开放模型中把 GQA（以及 Mistral 的 SWA）确立为标准。
- **DeepSeek** 把 MLA 作为设计空间中一个独特的点——做潜在 KV 压缩而不是共享头——随后在其上叠加可学习的稀疏性：NSA（研究）→ DSA（V3.2）→ CSA + HCA（V4，默认 1M 上下文）。
- **Moonshot/Qwen** 押注 delta 规则混合架构（Kimi Linear/K3 中的 KDA + MLA；Qwen3-Next/3.5 中的 Gated DeltaNet + 门控注意力）；Moonshot 还用 MoBA 块稀疏来支撑长上下文服务。
- **MiniMax** 的路径是闪电注意力（lightning attention）混合（M1）→ 全注意力（M2，理由是多跳推理受损以及状态对低精度敏感）→ 块稀疏的 MSA（M3）。
- **Anthropic/OpenAI** 的内核属于专有技术，但长上下文的 Claude/GPT 显然依赖 FlashAttention 级别的内核加上 KV 优化；两家如今都提供约 1M token 的窗口（Opus 4.6 的 1M beta，在 1M 长度的 MRCR v2 8-needle 上达到 76%；GPT-5.4/GPT-6 Astra 约 1.05M）。OpenAI 的开放模型 gpt-oss 让 128 token 的带状层与稠密层交替，配合 GQA（64 个 Q 头 / 8 个 KV 头）和可学习的注意力汇（attention sink）logit。

## 最新进展（2023–2026年11月）
带 FP8 的 FlashAttention-3；**MLA** 经由 DeepSeek-V2/V3 得到普及；**PagedAttention/前缀缓存**成为服务标准；只在少数层使用全注意力的注意力-SSM 混合模型（Jamba、Zamba）；以及对**原生稀疏注意力**（例如 DeepSeek 的 NSA，2025）的研究——这类注意力可以端到端训练，而不是事后施加。

**2025年中 → 2026年11月更新：**
- **可训练稀疏性进入生产（2025-02 → 2025-12）。** NSA（arXiv:2502.11089，ACL 2025 最佳论文）组合了压缩、选择和滑动三条分支；MoBA（arXiv:2502.13189）在 KV 块上施加 MoE 风格的 top-$k$ 门控。DeepSeek-V3.2-Exp/V3.2 中的 **DSA**（arXiv:2512.02556）分两阶段训练——先做稠密预热，用 KL 损失让索引器对齐主注意力分布，再做稀疏的继续训练——其质量与 V3.1-Terminus 持平，同时 API 价格下降超过 50%。GLM-5（2026-02；arXiv:2602.15763）在 744B 参数规模上采用了 DSA。
- **压缩稀疏注意力（2026-04）。** DeepSeek-V4 把每个 $m$-token 的 KV 块压缩成一个条目（CSA），由闪电索引器挑选 top-$k$ 个压缩条目，并额外在压缩程度高得多的 KV 上做稠密注意力（HCA）。据报道，在 1M 上下文下 V4-Pro 每 token 所需 FLOPs 约为 V3.2 的 27%，KV 缓存约为其 10%；另据报道，稀疏注意力是在稠密训练之后、到 64K 阶段才开启的（报告标注为 arXiv:2606.19348；编号与日期不一致——引用前请核实）。
- **delta 规则混合架构（2025-09/10）。** Qwen3-Next（3 Gated DeltaNet : 1 门控注意力）报告在 32K 以上长度时预填充吞吐量是 Qwen3-32B 的 10 倍以上；Kimi Linear（arXiv:2510.26692）报告在 1M 长度下 KV 缓存最多减少 75%、解码吞吐量最高提升 6 倍。反面证据：MiniMax M2 放弃了线性/SWA 变体，因为它们损害多跳推理，且在低精度状态存储下很脆弱。
- **注意力汇已可被工程化处理。** 输出门控（arXiv:2505.06708）消除了首 token 汇（质量占比从 46.7% 降至 4.8%），并使 RULER 提升超过 10 分；gpt-oss 则在 softmax 分母中为每个头学习一个显式的汇 logit。GDN 混合模型中的激活尖峰集中出现在全注意力层之前（arXiv:2608.12149，单一来源）。
- **将预训练模型线性化。** STILL（arXiv:2602.02180）把显著 token 送入 softmax，其余 token 用线性方式汇总，仅需约 0.04B 训练 token；MiniCPM-SALA（arXiv:2602.11761，二手来源）把一个 Transformer 转换为 1:3 的稀疏:线性混合模型，成本约为从头训练的 25%。
- **对服务的影响。** DSA 类索引器额外引入了"索引器 K 缓存"，使预填充/解码之间的 KV 传输更复杂；KV 变换编码（PCA + 自适应量化 + 熵编码，ICLR 2026）报告最高 20 倍压缩；注意力-FFN 解耦部署（arXiv:2605.28302）表明，注意力开销低的模型可以把 ≥90% 的 GPU 分给 FFN 专家。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"为什么 GQA 能降低内存，代价是什么？"** KV 缓存缩小 $h/g$ 倍；质量损失较小，而 MQA 的损失更大。
- **"解释 FlashAttention 如何避免 $O(n^2)$ 内存。"** 在线 softmax + 在 SRAM 中分块；它是 IO 感知、精确的，基于内存受限的分析。
- **"给定模型/上下文，KV 缓存有多大？"** 计算 $2\cdot \text{layers}\cdot g\cdot d_k\cdot n\cdot \text{bytes}$——展示你会估算。
- **"你会如何部署一个 1M token 上下文的模型？"** 环形/序列并行、分页 + 量化的 KV、前缀缓存——以及在 2026 年，一个 KV 实际上呈次线性增长的架构（3:1 线性混合或压缩 top-$k$ 稀疏注意力）。
- **"推导 gated DeltaNet 的更新式，并说明它为什么优于加性线性注意力。"** $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$：它覆写 $k_t$ 对应的槽位而不是累加，因此状态保持有界、召回更好。
- **"DSA 与 3:1 delta 规则混合架构——你会选哪个？"** DSA 在选中子集上保留精确的 softmax（检索能力好，但索引器是 $O(L^2)$ 且多一份缓存）；混合架构提供 $O(1)$ 状态的层和极小的 KV，但存在多跳/召回差距的风险（MiniMax 的论点）。

## 开放问题
次二次方注意力（线性注意力、SSM）能否在检索/上下文任务上完全追平 softmax 注意力仍未有定论——2025–26 年的混合模型在基准（benchmark）上已经持平，但 MiniMax 的多跳结果以及循环状态对精度的敏感性让这个问题依然悬而未决。全局与局部注意力的最优混合比例、能达到稠密质量的可训练稀疏性，都是活跃的研究方向。不损失召回的 KV 缓存压缩尚未解决。

## 参考文献
- Dao, T. et al. (2022). *FlashAttention.* arXiv:2205.14135.
- Dao, T. (2023). *FlashAttention-2.* arXiv:2307.08691.
- Shah, J. et al. (2024). *FlashAttention-3.* arXiv:2407.08608.
- Ainslie, J. et al. (2023). *GQA.* arXiv:2305.13245.
- Liu, H. et al. (2023). *Ring Attention.* arXiv:2310.01889.
- Kwon, W. et al. (2023). *PagedAttention / vLLM.* arXiv:2309.06180.
- Yang, S. et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464.
- Yuan, J. et al. (2025). *Native Sparse Attention.* arXiv:2502.11089.
- Lu, E. et al. (2025). *MoBA: Mixture of Block Attention for Long-Context LLMs.* arXiv:2502.13189.
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692.
- DeepSeek-AI (2025). *DeepSeek-V3.2*（DeepSeek Sparse Attention）. arXiv:2512.02556.
