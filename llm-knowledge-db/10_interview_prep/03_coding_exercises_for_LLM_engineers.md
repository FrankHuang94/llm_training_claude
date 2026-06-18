# Coding Exercises for LLM Engineers

> **Last Updated:** 2026-06-18
> **Related Files:** [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Tokenization](../00_foundations/02_tokenization.md) · [LoRA & Variants](../06_efficiency/01_LoRA_and_variants.md) · [RLHF](../03_posttraining/03_RLHF.md)

## Overview
LLM engineering interviews frequently include **implementation exercises**: write attention in PyTorch, build a BPE tokenizer, inject LoRA, or sketch a PPO/DPO step. These test whether you understand the math *operationally* — shapes, masking, numerical stability — not just conceptually. This file provides **reference implementations** to study and reproduce from memory. Practice writing each from scratch; interviewers watch for correct shapes, causal masking, the $\sqrt{d_k}$ scale, and numerical care.

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

## 5. Other Exercises to Practice
- **RMSNorm / SwiGLU** from scratch (a few lines each).
- **RoPE**: build the rotation and apply to Q/K.
- **KV cache**: implement incremental decoding with a growing cache.
- **Top-k / top-p (nucleus) sampling** and temperature.
- **Self-consistency**: sample N chains, majority-vote the answer.
- **MoE routing**: top-2 gating + load-balancing loss.
- **Speculative decoding**: draft loop + accept/reject with $\min(1,p/q)$.

## Interview Tips
- **Start with shapes** (write tensor dims in comments) — reduces bugs and signals competence.
- **Handle masking and numerical stability** (softmax over `-inf`, log-space, `logsumexp`).
- **State assumptions** (batch-first, causal vs bidirectional) and the **math** behind each line.
- **Know the gotchas**: prompt loss-masking in SFT, $\sqrt{d_k}$, $B=0$ LoRA init, KL sign in PPO.

## References
- Vaswani, A. et al. (2017). *Attention Is All You Need.* arXiv:1706.03762.
- Sennrich, R. et al. (2016). *BPE.* arXiv:1508.07909.
- Hu, E. et al. (2021). *LoRA.* arXiv:2106.09685.
- Rafailov, R. et al. (2023). *DPO.* arXiv:2305.18290.
- Schulman, J. et al. (2017). *PPO.* arXiv:1707.06347.
