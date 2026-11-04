# 面向 LLM 工程师的编程练习（Coding Exercises for LLM Engineers）

> **最后更新：** 2026-11-04
> **相关文件：** [Transformer 架构](../00_foundations/00_transformer_architecture.md) · [分词](../00_foundations/02_tokenization.md) · [LoRA 及其变体](../06_efficiency/01_LoRA_and_variants.md) · [RLHF](../03_posttraining/03_RLHF.md) · [GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [扩散与混合语言模型](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)

## 概述
大语言模型（LLM）工程面试经常包含**实现类练习**：用 PyTorch 写注意力、构建 BPE 分词器（tokenizer）、注入 LoRA、勾勒一个 PPO/DPO 训练步——而在 2026 年，越来越多地要求写一个 **GRPO/DAPO 式的强化学习（RL）损失**或一个**稀疏注意力玩具实现**。这些题检验你是否在*操作层面*理解数学——形状、掩码、数值稳定性——而不仅是概念上理解。本文件提供**参考实现**，供学习并凭记忆复现。每个都要练习从零写出；面试官会关注形状是否正确、因果掩码、$\sqrt{d_k}$ 缩放以及数值上的谨慎。

## 1. 缩放点积注意力与多头注意力
```python
import torch, torch.nn as nn, torch.nn.functional as F, math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q,K,V: (B, H, T, d_k)
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # (B,H,T,T)
    if mask is not None:                                  # causal: upper-tri = -inf
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V                                       # (B,H,T,d_k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, causal=True):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # each (B,H,T,d_k)
        mask = torch.tril(torch.ones(T, T, device=x.device)) if causal else None
        o = scaled_dot_product_attention(q, k, v, mask)  # (B,H,T,d_k)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(o)
```
**讲解要点：** $\sqrt{d_k}$ 缩放；在 softmax 之前用 `masked_fill(-inf)` 实现因果掩码；多头的 reshape/permute；为什么不加 bias（现代惯例）。要准备好扩展到**分组查询注意力（GQA）**（更少的 KV 头：把 K、V 投影到 `g*d_k`，再 repeat-interleave 到 `h`）。

## 2. 字节对编码（BPE）训练器
```python
from collections import Counter

def get_pair_freqs(corpus):                 # corpus: list[list[str]] of symbols
    pairs = Counter()
    for word in corpus:
        for a, b in zip(word[:-1], word[1:]):
            pairs[(a, b)] += 1
    return pairs

def merge(corpus, pair):
    a, b = pair; new = a + b; out = []
    for word in corpus:
        i, merged = 0, []
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                merged.append(new); i += 2
            else:
                merged.append(word[i]); i += 1
        out.append(merged)
    return out

def train_bpe(words, num_merges):
    corpus = [list(w) + ['</w>'] for w in words]   # char-level + end marker
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_freqs(corpus)
        if not pairs: break
        best = max(pairs, key=pairs.get)           # most frequent adjacent pair
        corpus = merge(corpus, best)
        merges.append(best)
    return merges                                  # ordered merge rules
```
**讲解要点：** 贪心地合并出现频率最高的相邻对；编码时按学到的顺序应用合并规则；字节级变体在字节上操作（零 OOV）；与 WordPiece（基于似然）/ Unigram（自顶向下剪枝）对比。（见[分词](../00_foundations/02_tokenization.md)。）

## 3. LoRA 权重注入
```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base                            # frozen
        for p in self.base.parameters(): p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)  # init A ~ small
        self.B = nn.Parameter(torch.zeros(d_out, r))        # init B = 0 -> ΔW=0
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scale
    def merge(self):                                # for zero-latency inference
        self.base.weight.data += (self.B @ self.A) * self.scale
```
**讲解要点：** $\Delta W=BA$，$B=0$ 初始化使训练从预训练模型出发，$\alpha/r$ 缩放，**可合并** → 无推断开销；QLoRA = 把 `base` 量化为 4 比特 NF4。（见[LoRA](../06_efficiency/01_LoRA_and_variants.md)。）

## 4. DPO 损失（以及 PPO 训练步草图）
```python
def dpo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    # logps_*: sum of log-probs of the chosen/rejected response under policy/ref
    pi_logratio  = pi_logps_w  - pi_logps_l
    ref_logratio = ref_logps_w - ref_logps_l
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
```
```python
# PPO clipped policy loss (per-token), advantages A_t precomputed (GAE)
def ppo_policy_loss(logp, logp_old, A, eps=0.2):
    ratio = torch.exp(logp - logp_old)
    return -torch.min(ratio * A, torch.clamp(ratio, 1-eps, 1+eps) * A).mean()
```
**讲解要点：** DPO 的隐式奖励 $\beta\log\pi/\pi_{ref}$ 以及 $Z(x)$ 的消去；PPO 的裁剪作为廉价的信任域；加到奖励上的对参考模型的 KL 惩罚；GRPO 用组内归一化的优势 $\hat A_i=(r_i-\text{mean})/\text{std}$ 取代价值函数/评论家（critic）。（见[DPO](../03_posttraining/05_DPO_and_preference_optimization.md)、[RLHF](../03_posttraining/03_RLHF.md)。）

## 5. GRPO / DAPO 损失（token 级归一化 + clip-higher）与 CISPO
```python
def grpo_dapo_loss(logp, logp_old, rewards, mask, group_size,
                   eps_low=0.2, eps_high=0.28, std_norm=True, token_level=True):
    # logp, logp_old: (N, T) per-token log-probs of the sampled tokens under the
    #                 current policy / the behavior (rollout) policy
    # rewards: (N,) one scalar per response; N = num_prompts * group_size, grouped contiguously
    # mask:    (N, T) 1.0 on response tokens, 0.0 on prompt/padding
    r = rewards.view(-1, group_size)                          # (P, G)
    adv = r - r.mean(dim=1, keepdim=True)                     # group-relative baseline, no critic
    if std_norm:                                              # GRPO/DAPO; Dr. GRPO: std_norm=False
        adv = adv / (r.std(dim=1, keepdim=True) + 1e-6)
    # dynamic sampling (DAPO): all-correct / all-wrong groups carry zero signal -> drop them
    # (DAPO actually over-samples until the batch is full; masking is the toy version)
    keep = (r.std(dim=1) > 0).float().repeat_interleave(group_size)   # (N,)
    adv = adv.reshape(-1, 1)                                  # (N, 1) broadcast over tokens
    m = mask * keep[:, None]                                  # (N, T)

    ratio = torch.exp(logp - logp_old)                        # (N, T) importance ratio
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)   # clip-higher: eps_high > eps_low
    per_tok = -torch.min(ratio * adv, clipped * adv)          # (N, T) PPO-style pessimistic bound

    if token_level:   # DAPO: every valid token in the batch has equal weight
        return (per_tok * m).sum() / m.sum().clamp(min=1.0)
    # original GRPO: mean over tokens within each sequence, then over sequences
    # -> tokens of long responses are down-weighted (length bias)
    seq = (per_tok * m).sum(1) / m.sum(1).clamp(min=1.0)
    return seq[keep > 0].mean() if keep.any() else seq.sum() * 0
```
```python
def cispo_loss(logp, logp_old, adv, mask, eps_high=0.28):
    # CISPO: clip the IS *weight* and stop its gradient; every token keeps a gradient through logp
    w = torch.clamp(torch.exp(logp - logp_old), max=1 + eps_high).detach()   # (N, T)
    per_tok = -w * adv * logp                                 # REINFORCE with truncated IS
    return (per_tok * mask).sum() / mask.sum().clamp(min=1.0)
```
**讲解要点：** 组内相对优势取代了评论家；**clip-higher**（$\epsilon_{high}>\epsilon_{low}$）让具有正优势的低概率 token 得以增长，从而对抗熵坍缩；**token 级**聚合 vs GRPO 的逐序列均值（后者会对长回答的每个 token 降权，并且在配合标准差归一化时偏向冗长的错误答案以及简单/困难的提示——见 Dr. GRPO）；动态采样移除零优势的组；**CISPO** 截断并分离（detach）重要性采样权重，使被裁剪的"分叉"token 仍能获得梯度；**GSPO** 则会把逐 token 比率替换为序列比率 `exp(((logp-logp_old)*mask).sum(1)/mask.sum(1))` 并按序列裁剪。要提到 `logp_old` 必须来自 *rollout 引擎*的数值计算，否则会出现训练–推断失配（FP16、批不变 kernel、截断重要性采样）。（见[GRPO 与 RLVR](../03_posttraining/06_GRPO_and_RLVR.md)。）

## 6. Top-k 稀疏注意力玩具实现（DSA 式 lightning indexer）
```python
def topk_sparse_attention(q, k, v, q_idx, k_idx, w_idx, top_k):
    # q, k, v: (B, H, T, d)   main attention (MQA/MLA in DSA; plain MHA here)
    # q_idx:   (B, Hi, T, di) indexer queries (few heads, small dim, FP8 in DSA)
    # k_idx:   (B, T, di)     indexer keys, shared across indexer heads
    # w_idx:   (B, Hi, T)     per-query, per-indexer-head weights
    B, H, T, d = q.shape
    # 1) lightning indexer: I[t,s] = sum_j w[t,j] * ReLU(q_idx[t,j] . k_idx[s])
    idx = torch.einsum('bjtd,bsd->bjts', q_idx, k_idx).relu()          # (B, Hi, T, T)
    idx = (w_idx.unsqueeze(-1) * idx).sum(1)                           # (B, T, T)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    idx = idx.masked_fill(~causal, float('-inf'))
    # 2) top-k keys per query, shared by all heads
    kk = min(top_k, T)
    top = idx.topk(kk, dim=-1).indices                                  # (B, T, kk)
    sel = torch.zeros(B, T, T, dtype=torch.bool, device=q.device)
    sel.scatter_(-1, top, True)
    sel &= causal          # rows t < kk picked some -inf (future) slots; drop them
    # 3) attention restricted to the selected keys
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)                   # toy: dense O(T^2)
    scores = scores.masked_fill(~sel[:, None], float('-inf'))           # (B, H, T, T)
    return F.softmax(scores, dim=-1) @ v                                # (B, H, T, d)
    # a real kernel gathers only the kk selected K/V rows: O(T*kk) instead of O(T^2)
```
**讲解要点：** indexer 很便宜（少量头、小维度、ReLU，在 DSA 中用 FP8），但仍是 $O(T^2)$；一旦实现为 gather 操作，主注意力就变为 $O(Tk)$；top-$k$ 在各头之间共享（DSA 以 MQA 模式运行 MLA）；当 `top_k >= T` 时输出等于稠密因果注意力（很好的单元测试）；indexer 先通过**稠密预热**训练，用 KL 把其分布与（各头求和后的）主注意力对齐，再进行稀疏训练；DeepSeek-V4 的 CSA 在**块压缩**后的 KV 上做同样的选择。与线性混合（固定大小的 delta 规则状态）以及 MiniMax 的块级 MSA 进行对比。（见[扩散与混合语言模型](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)。）

## 7. 其他可练习的题目
- 从零实现 **RMSNorm / SwiGLU**（各几行代码）。
- **RoPE**：构造旋转并作用于 Q/K。
- **KV 缓存**：用不断增长的缓存实现增量解码。
- **Top-k / top-p（nucleus）采样**以及温度。
- **自洽性（self-consistency）**：采样 N 条推理链，对答案多数投票。
- **MoE 路由**：top-2 门控 + 负载均衡损失。
- **投机解码**：草稿循环 + 以 $\min(1,p/q)$ 接受/拒绝。
- **Gated DeltaNet 递推**：先用 Python 循环实现 $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$，再实现分块版本。
- **NVFP4 伪量化**：16 元素块、E2M1 数值、每块一个 FP8 缩放因子；加入随机舍入。
- **同策略蒸馏损失**：在学生样本上计算学生与教师之间的逐 token 反向 KL。
- **掩码扩散语言模型损失**：采样 $t$、掩码 token、在被掩码位置上计算按 $1/t$ 加权的交叉熵。

## 面试技巧
- **从形状写起**（在注释里写出张量维度）——减少 bug，也体现能力。
- **处理好掩码与数值稳定性**（对 `-inf` 做 softmax、在对数空间计算、`logsumexp`）。
- **说明假设**（批维在前、因果 vs 双向）以及每一行背后的**数学**。
- **了解常见坑**：SFT 中对提示做损失掩码、$\sqrt{d_k}$、LoRA 的 $B=0$ 初始化、PPO 中 KL 的符号、GRPO 家族损失的聚合层级（token vs 序列）、稀疏掩码中整行为 `-inf` 的情况（始终至少保留一个有效键）。

## 参考文献
- Vaswani, A. et al. (2017). *Attention Is All You Need.* arXiv:1706.03762.
- Sennrich, R. et al. (2016). *BPE.* arXiv:1508.07909.
- Hu, E. et al. (2021). *LoRA.* arXiv:2106.09685.
- Rafailov, R. et al. (2023). *DPO.* arXiv:2305.18290.
- Schulman, J. et al. (2017). *PPO.* arXiv:1707.06347.
- Yu, Q. et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476.
- Liu, Z. et al. (2025). *Understanding R1-Zero-Like Training (Dr. GRPO).* arXiv:2503.20783.
- MiniMax (2025). *MiniMax-M1 (CISPO).* arXiv:2506.13585.
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071.
- DeepSeek-AI (2025). *DeepSeek-V3.2 (DeepSeek Sparse Attention).* arXiv:2512.02556.
