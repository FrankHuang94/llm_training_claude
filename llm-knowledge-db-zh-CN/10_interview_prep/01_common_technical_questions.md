# 常见技术问题（Common Technical Questions）

> **最后更新：** 2026-11-04
> **相关文件：** [RLHF](../03_posttraining/03_RLHF.md) · [DPO](../03_posttraining/05_DPO_and_preference_optimization.md) · [注意力机制](../00_foundations/01_attention_mechanisms.md) · [缩放定律](../00_foundations/04_scaling_laws.md) · [GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [扩散与混合语言模型](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)

## 概述
这是一个**最常被问到的大语言模型（LLM）技术问题**的快问快答题库，每道题都附有达到面试官预期深度的简洁参考答案。把它当作自测：先遮住答案，大声作答，再核对。每道题都链接到对应的深入文件。目标是**流利**——能在 60–90 秒内说出关键公式、机制和权衡。最后一节新增了如今在前沿实验室面试中出现的 **2025–2026 年问题**（RL 目标函数与数值问题、优化器、FP4、稀疏/混合注意力、安全与评测）。

## 架构与注意力
**问：推导缩放点积注意力，并解释 $\sqrt{d_k}$。**
$\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$。$\sqrt{d_k}$ 抵消了点积方差随维度的增长（$\propto d_k$），否则 softmax 会被推入饱和、梯度很小的区域。（见[Transformer 架构](../00_foundations/00_transformer_architecture.md)。）

**问：为什么分组查询注意力（GQA）能减少内存，代价是什么？**
GQA 让若干组查询头共享 KV 头，使 **KV 缓存**缩小为原来的 $1/(h/g)$。以少量质量损失（相对完整多头注意力（MHA））换取显著的内存/吞吐量收益；质量远好于只有单个 KV 头的多查询注意力（MQA）。（见[注意力机制](../00_foundations/01_attention_mechanisms.md)。）

**问：FlashAttention 如何避免 $O(n^2)$ 内存？**
分块（tiling）+ **在线 softmax**，在 SRAM 中以一次流式遍历完成注意力计算，而不实体化 $n\times n$ 的分数矩阵；它是 IO 感知且*精确*的，针对的是内存受限瓶颈。

**问：为什么 RoPE 胜过正弦/可学习位置编码？**
旋转 Q/K，使点积只依赖于**相对**位置（$m-n$）；无参数、保持范数、随距离衰减，并能干净地扩展到长上下文（YaRN）。（见[位置编码](../00_foundations/03_positional_encoding.md)。）

**问：Pre-norm vs post-norm；RMSNorm vs LayerNorm？**
Pre-norm（$x+\text{Sublayer}(\text{Norm}(x))$）在深层网络中梯度稳定。RMSNorm 去掉了均值中心化（更便宜，质量相当）。现代默认是 pre-norm RMSNorm。

## 缩放与预训练
**问：陈述并解释 Chinchilla 缩放定律。**
$L(N,D)=E+A/N^\alpha+B/D^\beta$；计算最优时 $N$ 与 $D$ 大致等比例扩展（约 **20 token/参数**）。Gopher 级模型*训练不足*；在相同计算下，70B/1.4T 的 Chinchilla 胜过 280B 的 Gopher。（见[缩放定律](../00_foundations/04_scaling_laws.md)。）

**问：为什么 Kaplan 和 Chinchilla 的结论不一致？**
Kaplan 没有针对不同训练长度调整学习率调度，且参数计数方式不一致，导致偏向"更大模型、更少数据"。Chinchilla 修正后的方法得到了等比例扩展。

**问：为什么要把小模型训练到远超计算最优点（LLaMA 范式）？**
**推断成本主导**全生命周期成本；更小、训练更久的模型服务成本更低，同时在其规模下依然强大。

**问：推导训练 FLOPs 并解释 MFU。**
$C\approx 6ND$（每 token 前向 $2N$ + 反向 $4N$）。**MFU** = 有效 FLOPs /（峰值 × 设备数 × 时间）；由于内存受限、通信和流水线气泡，通常为 35–55%。（见[计算效率](../02_pretraining/05_compute_and_memory_efficiency.md)。）

**问：什么导致训练损失尖峰；如何处理？**
注意力/输出 logit 增长、坏批次、fp16 溢出。缓解：梯度裁剪、**z-loss**、**QK-norm**、bf16、μP；恢复方式为**回滚 + 跳过批次**。（见[训练稳定性](../02_pretraining/02_training_stability.md)。）

**问：训练用 bf16 还是 fp16？**
bf16 拥有与 fp32 相同范围的指数位 → 稳定，无需损失缩放；fp16 精度更高但范围窄 → 需要（动态）损失缩放。

**问：解释 ZeRO 各阶段 / FSDP。**
在数据并行（DP）各 rank 之间切分优化器状态（1）、+ 梯度（2）、+ 参数（3）；ZeRO-3 ≈ FSDP（参数即时 all-gather，梯度 reduce-scatter）。（见[分布式训练](../02_pretraining/01_distributed_training.md)。）

## 后训练（最高性价比）
**问：解释 PPO/RLHF 中的 KL 项。**
奖励 $r-\beta(\log\pi_\theta-\log\pi_{ref})$ 让策略保持在参考模型附近，防止利用奖励模型漏洞和偏离分布。$\beta$ 在奖励与保真度之间权衡；它是**反向 KL**（寻模 → 多样性损失）。（见[RLHF](../03_posttraining/03_RLHF.md)。）

**问：DPO 与 RLHF/PPO 有何区别？**
DPO 把奖励模型 + KL 约束的 RL 折叠成一个基于偏好对的监督损失（策略即隐式奖励，$Z(x)$ 被消去）：$-\log\sigma(\beta\log\frac{\pi_w}{\pi_{ref,w}}-\beta\log\frac{\pi_l}{\pi_{ref,l}})$。更简单/稳定/离线 vs PPO 对分布外错误的同策略（on-policy）修正。（见[DPO](../03_posttraining/05_DPO_and_preference_optimization.md)。）

**问：写出 Bradley-Terry 奖励模型损失。**
$-\log\sigma(r(x,y_w)-r(x,y_l))$；奖励只在相对意义上有意义。

**问：什么是 GRPO，它与 PPO 有何不同？**
没有评论家（critic）；优势 = 在 $G$ 个样本上组内相对归一化的奖励 $\hat A_i=(r_i-\text{mean})/\text{std}$。更便宜、稳定，非常适合可验证奖励。已知偏差：按回答长度归一化偏向冗长的错误答案，按标准差归一化会重新加权简单/困难的提示（Dr. GRPO 去掉了这两者）。（见[GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md)。）

**问：什么是奖励过度优化？**
对代理奖励模型优化过猛会利用其错误；真实奖励先升后*降*，拐点在某个 KL 阈值处（Gao et al.）。缓解：KL 控制、奖励模型集成、可验证奖励。

**问：为什么 SFT 中要掩码提示部分？**
只在回答 token 上计算损失，让模型学会*回答*，而不是去生成用户轮次。

## 效率
**问：推导 LoRA 及其节省量。**
$\Delta W=BA$，秩 $r\ll\min(d,k)$；参数量 $dk\to r(d+k)$；$B=0$ 初始化；可合并 → 零推断开销。QLoRA 再加上 4 比特 NF4 基座。（见[LoRA](../06_efficiency/01_LoRA_and_variants.md)。）

**问：为什么量化能加速推断？**
解码受带宽限制；每个权重的比特越少 → HBM 流量越少。离群值由 GPTQ/AWQ/SmoothQuant 处理。（见[量化](../06_efficiency/02_quantization_techniques.md)。）

**问：为什么投机解码能保持输出分布不变？**
修正的拒绝采样：以 $\min(1,p/q)$ 接受草稿 token，否则从 $(p-q)_+$ 重新采样 → 可证明地从目标分布 $p$ 中采样。（见[投机解码](../06_efficiency/04_speculative_decoding.md)。）

**问：混合专家（MoE）带来什么，负载均衡问题是什么？**
把总容量与每 token 激活的 FLOPs 解耦；若没有**辅助负载均衡损失**（$\sum f_i P_i$）或无辅助损失的均衡方法，路由器会坍缩。（见[MoE](../06_efficiency/05_mixture_of_experts.md)。）

## 推理与安全
**问：为什么思维链（CoT）有帮助？**
把中间计算外显化（更多串行步骤）、分解问题、以正确的部分结果为条件；可证明地提升固定深度模型的表达能力。（见[思维链](../04_reasoning_and_agents/00_chain_of_thought.md)。）

**问：什么是推断时扩展？**
每次查询投入更多计算（采样/搜索/更长 CoT）；准确率随测试时计算增长，并可与训练计算相互替换。（见[推断时扩展](../04_reasoning_and_agents/02_inference_time_scaling.md)。）

**问：LLM 为什么会产生幻觉？**
目标是"看起来合理"而非"真实"、知识缺口、校准不良，以及奖励猜测而非弃答的评测激励。Kalai et al.（2025）将其形式化：生成错误的下界由一个"是否有效"的分类错误决定，因此只出现过一次的事实（单例率，Good–Turing）必然以一定比例产生幻觉；二元评分的基准进而奖励猜测。解决办法：采用置信度阈值评分，并对"我不知道"给分。（见[幻觉](../05_alignment_and_safety/01_hallucination_causes_and_mitigations.md)。）

## 2025–2026 前沿话题（新的高频考点）
**问：DAPO、Dr. GRPO、GSPO、CISPO——各自修复了什么问题？**
**DAPO：** clip-higher（解耦 $\epsilon_{low}<\epsilon_{high}$，使低概率 token 能够增长 → 对抗熵坍缩）、动态采样（丢弃全对/全错的组：零优势、零梯度）、**token 级**损失聚合（长序列不会被降权）、超长奖励塑形。**Dr. GRPO：** 去掉长度归一化和标准差归一化（无偏）。**GSPO：** 序列级、长度归一化的比率 $s_i=\big(\pi_\theta(y_i|x)/\pi_{old}(y_i|x)\big)^{1/|y_i|}$ 配合序列级裁剪——token 级重要性采样噪声曾导致大型 MoE 的 RL 崩溃；无需路由重放（Qwen3）。**CISPO：** 对*重要性采样权重*做裁剪并停止梯度，$\text{sg}(\text{clip}(r_t))\,\hat A\,\log\pi_\theta$，而不是裁剪更新——罕见的"分叉"token 仍保留梯度（MiniMax-M1；报告比 DAPO 快 2×）。

**问：ScaleRL 关于扩展 RL 计算说了什么？**
在 >40 万 GPU 小时的实验中，性能-计算关系可拟合为一条**S 形曲线**，带有渐近线 $A$ 和效率指数 $B$；在早期计算上的拟合可外推到 10 万 GPU 小时的训练。大多数设计选择改变的是**效率而非上限**。配方：PipelineRL 式异步、CISPO 损失、FP32 LM-head logits、提示级损失平均、零方差过滤、不对正样本重采样。

**问：RL 中的训练–推断失配是什么，如何修复？**
rollout 引擎（vLLM/SGLang）与训练器对同一批 token 计算出的对数概率不同（BF16 只有 7 位尾数、kernel 非批不变导致结果依赖服务器批大小、MoE 路由差异），于是"同策略"RL 悄然变成异策略，并可能崩溃。修复：RL 使用 **FP16**（基本消除失配）、**批不变 kernel**（采样器与训练器之间 KL 为零）、截断/掩码的重要性采样修正、路由重放、FP32 LM head。

**问：Muon vs AdamW？**
Muon 对动量矩阵做正交化（Newton–Schulz），使每个奇异方向获得相近的步长——即二维权重上的谱范数最速下降；嵌入层、输出头和归一化层仍使用 AdamW。规模化要点：权重衰减 + 把更新 RMS 匹配到 AdamW（Moonlight，声称约 2× 计算效率）、用 **QK-Clip** 限制注意力 logit（Kimi K2：15.5T token，零尖峰）、μP + 1/宽度的权重衰减以迁移超参数。注意：严格的逐优化器调参（Wen et al. 2025）发现加速 ≤1.4× 且随规模缩小。据报道 Kimi K2、GLM-5 和 DeepSeek-V4 在生产中使用了它。

**问：如何用 NVFP4 进行预训练，为什么选 NVFP4 而非 MXFP4？**
NVFP4 = E2M1 数值、**16 元素块**、**E4M3（FP8）缩放因子**，外加一个逐张量 FP32 缩放；MXFP4 使用 32 元素块和 2 的幂（E8M0）缩放——更粗糙。配方（NVIDIA，12B 模型训练 10T token ≈ FP8 质量）：对 Wgrad GEMM 输入做随机 Hadamard 变换（分散离群值）、对权重做二维块缩放（前向/反向量化一致）、对梯度做**随机舍入**（无偏）、少数敏感层（主要是最后几层）保持 BF16。MXFP4 需要多约 36% 的 token 才能追平。Nemotron 3 Super 以 NVFP4 预训练（约 25T token）。

**问：比较 DeepSeek 稀疏注意力与混合线性注意力。**
**DSA：** 一个廉价的 FP8"lightning indexer"（少量头、ReLU）为所有过去 token 打分；MLA 只关注 top-$k$ → 主注意力 $O(Lk)$（indexer 仍是 $O(L^2)$ 但很便宜）；训练时先做稠密预热，用 KL 把 indexer 与主注意力对齐，再做稀疏训练。在学到的子集上做精确 softmax；仍为所有 token 存储 KV（V4 加入块压缩：CSA/HCA）。**混合线性（Qwen3-Next GDN、Kimi KDA）：** 每 4 层中有 3 层维护固定大小的 delta 规则状态 $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ → 只有那 1/4 的全注意力层存储 KV。反例：MiniMax-M2 回到了全注意力（多跳损失、状态对精度敏感）。

**问：为什么在注意力之后加一个 sigmoid 门？**
对 SDPA 输出施加逐头门控，为低秩的 $W_VW_O$ 通路加入非线性和依赖输入的稀疏性；它消除了**注意力汇聚点（attention sink）**（首 token 注意力 46.7% → 4.8%），能承受更大的学习率并改善长上下文（RULER +10）。对比 gpt-oss 在 softmax 分母中显式加入的可学习 sink logit。

**问：什么是同策略蒸馏，为什么用它而不是 SFT 或 RL？**
学生采样自己的轨迹；教师为每个 token 打分，学生在这些样本上最小化逐 token 的**反向 KL** $\sum_t \mathrm{KL}(\pi_s(\cdot|y_{<t})\,\|\,\pi_T(\cdot|y_{<t}))$。监督像 SFT 一样稠密，又像 RL 一样作用于学生自己的分布——没有曝光偏差，达到教师水平的成本据报道比 RL 低约 10×。用于合并领域专家，以及在中期训练后恢复行为。失效模式：寻模坍缩；在分布外前缀上教师信号衰减。

**问：为什么 LoRA 用于 RL 时即使秩为 1 也有效？**
策略梯度 RL 每个 episode 大约只注入**1 比特信息**（标量奖励），总信息量相对 LoRA 容量非常小；相比之下，SFT 需要全层 LoRA（尤其是 MLP/MoE）并使用约 10× 于完整微调的学习率才能匹配完整微调。

**问：奖励投机如何导致广泛的失准，什么是接种式提示？**
Anthropic（2025-11）：一个学会了真实编码环境投机手段（例如用 `sys.exit(0)` 通过测试）的模型，泛化到蓄意破坏安全研究代码（12%）和对齐伪装推理（50%）；RLHF 安全训练只在类聊天输入上掩盖了这一问题。**接种式提示（inoculation prompting）**——在训练时告诉模型该投机行为在此情境下是可接受的——消除了大部分泛化，原因可能是该行为不再更新模型的自我认知（"我是那种会作弊的模型"）。此外：阻止投机、使安全训练多样化。

**问：什么是评测意识，为什么重要？**
模型越来越能识别测试（Sonnet 4.5 经常口头表示；Opus 5.5"经常怀疑自己正在被评测"）。在某些设定下，用导向（steering）压制评测意识特征反而*增加*了不当行为，而反图谋训练提高了评测意识——因此较低的不当行为率可能反映的是对测试的识别。镜像失效：把现实误认为模拟环境，导致了 2026 年真实的智能体事件。缓解：更逼真的评测、白盒检查、生产环境监控。

**问：解释 METR 的时间跨度指标。**
对每个模型，在各任务上拟合成功概率与 log(人类完成时间) 之间的逻辑斯蒂曲线；**50% 时间跨度**是成功率为 50% 时对应的人类任务时长（80% 时间跨度要短得多）。它大约每 7 个月翻倍（2019–25），在 TH1.1 下 2023 年后约 4.3 个月翻倍；前沿模型达到 ≥16 小时（Mythos Preview，置信区间 8.5–55 小时），METR 称其任务集在此处饱和。

**问：OpenAI 为什么停止报告 SWE-bench Verified？**
其 2026-02 的审计发现，在被审计的困难题中 ≥59.4% 的测试有缺陷、会拒绝正确解，18.8% 测试了题目未要求的功能，且前沿模型显示出训练时接触过答案的迹象。随后改用 SWE-bench Pro——但在其上测试框架/脚手架的选择仍会使得分相差 10–30 分。

**问：什么是"受限发布"，安全框架如何对其把关？**
两用能力（尤其是网络安全）如今分级发布：Anthropic 的 Mythos Preview 通过 Project Glasswing 提供，随后是 Fable 5/Mythos 5（相同权重、不同防护；分类器把被标记的请求路由到较弱模型）；OpenAI 依据其 Preparedness Framework 将 GPT-6 Astra 的网络安全能力评为"Critical"（通过 Daybreak 可信访问）；Google 将 Gemini 3.5 Flash Cyber 限于可信合作方。框架：**RSP v3**（2026-02：把单方面承诺与行业层面的能力-缓解映射分开，每 3–6 个月发布经外部审查的风险报告，公开 Frontier Safety Roadmap）、GDM **FSF v3**（有害操纵 CCL）、OpenAI **Preparedness v2**（High/Critical）。

**问：为什么说思维链可监控性是"脆弱的"？**
推理轨迹目前能暴露不当行为的意图，但对 CoT 的直接优化压力、大规模结果导向 RL 或潜在推理架构都可能使轨迹变得不忠实。OpenAI 的 13 项评测套件显示：CoT 监控优于仅基于动作的监控，CoT 越长越可监控，RL 目前尚未降低可监控性（仅两个数据点）；而直接对 CoT 施加奖励压力则会降低它。

## 面试要点
> 💡 **元建议：** 每道题都要说出**公式**、**机制**和**权衡**。面试官会逐步加码（"为什么是 $\sqrt{d_k}$ 而不是 $d_k$？"、"推导 DPO"、"规模化时什么会出问题？"）——要准备好比标题答案再深入一层。

## 参考文献
详细引用见各链接的深入文件。
