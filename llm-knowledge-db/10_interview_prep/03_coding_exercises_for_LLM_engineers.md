# Coding Exercises for LLM Engineers

> **Last Updated:** 2026-11-04
> **Related Files:** [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Tokenization](../00_foundations/02_tokenization.md) · [LoRA & Variants](../06_efficiency/01_LoRA_and_variants.md) · [RLHF](../03_posttraining/03_RLHF.md) · [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md) · [Diffusion & Hybrid LMs](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md)

## Overview
LLM engineering interviews frequently include **implementation exercises**: write attention in PyTorch, build a BPE tokenizer, inject LoRA, sketch a PPO/DPO step — and, increasingly in 2026, write a **GRPO/DAPO-style RL loss** or a **sparse-attention toy**. These test whether you understand the math *operationally* — shapes, masking, numerical stability — not just conceptually. This file provides **reference implementations** to study and reproduce from memory. Practice writing each from scratch; interviewers watch for correct shapes, causal masking, the $\sqrt{d_k}$ scale, and numerical care.

## 1. Scaled Dot-Product & Multi-Head Attention
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
**Talking points:** $\sqrt{d_k}$ scaling, causal mask via `masked_fill(-inf)` before softmax, head reshape/permute, why no bias (modern convention). Be ready to extend to **GQA** (fewer KV heads: project K,V to `g*d_k` and repeat-interleave to `h`).

## 2. Byte-Pair Encoding (BPE) Trainer
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
**Talking points:** greedy most-frequent-pair merging; encoding applies merges in learned order; byte-level variant operates on bytes (zero OOV); contrast with WordPiece (likelihood) / Unigram (top-down pruning). (See [Tokenization](../00_foundations/02_tokenization.md).)

## 3. LoRA Weight Injection
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
**Talking points:** $\Delta W=BA$, $B=0$ init so training starts at the pretrained model, $\alpha/r$ scaling, **mergeable** → no inference cost; QLoRA = quantize `base` to 4-bit NF4. (See [LoRA](../06_efficiency/01_LoRA_and_variants.md).)

## 4. DPO Loss (and a PPO step sketch)
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
**Talking points:** DPO's implicit reward $\beta\log\pi/\pi_{ref}$ and the $Z(x)$ cancellation; PPO's clip as a cheap trust region; the KL-to-reference penalty added to the reward; GRPO replaces the value/critic with a group-normalized advantage $\hat A_i=(r_i-\text{mean})/\text{std}$. (See [DPO](../03_posttraining/05_DPO_and_preference_optimization.md), [RLHF](../03_posttraining/03_RLHF.md).)

## 5. GRPO / DAPO Loss (token-level normalization + clip-higher) and CISPO
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
**Talking points:** group-relative advantage replaces the critic; **clip-higher** ($\epsilon_{high}>\epsilon_{low}$) lets low-probability tokens with positive advantage grow, countering entropy collapse; **token-level** aggregation vs GRPO's per-sequence mean (which down-weights every token of long responses and, with std-normalization, biases toward long wrong answers and easy/hard prompts — Dr. GRPO); dynamic sampling removes zero-advantage groups; **CISPO** truncates and detaches the IS weight so clipped "fork" tokens still get gradient; **GSPO** would replace the per-token ratio with the sequence ratio `exp(((logp-logp_old)*mask).sum(1)/mask.sum(1))` and clip per sequence. Mention that `logp_old` must come from the *rollout engine's* numerics or you get training–inference mismatch (FP16, batch-invariant kernels, truncated IS). (See [GRPO & RLVR](../03_posttraining/06_GRPO_and_RLVR.md).)

## 6. Top-k Sparse Attention Toy (DSA-style lightning indexer)
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
**Talking points:** the indexer is cheap (few heads, small dim, ReLU, FP8 in DSA) but still $O(T^2)$; main attention becomes $O(Tk)$ once implemented as a gather; top-$k$ is shared across heads (DSA runs MLA in MQA mode); with `top_k >= T` the output equals dense causal attention (good unit test); the indexer is trained first with a **dense warm-up** that KL-aligns its distribution to the (head-summed) main attention, then sparse training; DeepSeek-V4's CSA applies the same selection to **block-compressed** KV. Contrast with linear hybrids (fixed-size delta-rule state) and MiniMax's block-level MSA. (See [Diffusion & Hybrid LMs](../09_emerging_frontiers/06_diffusion_and_hybrid_language_models.md).)

## 7. Other Exercises to Practice
- **RMSNorm / SwiGLU** from scratch (a few lines each).
- **RoPE**: build the rotation and apply to Q/K.
- **KV cache**: implement incremental decoding with a growing cache.
- **Top-k / top-p (nucleus) sampling** and temperature.
- **Self-consistency**: sample N chains, majority-vote the answer.
- **MoE routing**: top-2 gating + load-balancing loss.
- **Speculative decoding**: draft loop + accept/reject with $\min(1,p/q)$.
- **Gated DeltaNet recurrence**: $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ in a Python loop, then a chunked version.
- **NVFP4 fake-quant**: 16-element blocks, E2M1 values, FP8 scale per block; add stochastic rounding.
- **On-policy distillation loss**: per-token reverse KL between student and teacher on student samples.
- **Masked-diffusion LM loss**: sample $t$, mask tokens, $1/t$-weighted cross-entropy on masked positions.

## Interview Tips
- **Start with shapes** (write tensor dims in comments) — reduces bugs and signals competence.
- **Handle masking and numerical stability** (softmax over `-inf`, log-space, `logsumexp`).
- **State assumptions** (batch-first, causal vs bidirectional) and the **math** behind each line.
- **Know the gotchas**: prompt loss-masking in SFT, $\sqrt{d_k}$, $B=0$ LoRA init, KL sign in PPO, loss aggregation level (token vs sequence) in GRPO-family losses, `-inf` rows in sparse masks (always keep at least one valid key).

## References
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
