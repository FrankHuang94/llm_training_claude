# Pretraining Objectives

> **Last Updated:** 2026-06-18
> **Related Files:** [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Information Theory Basics](../00_foundations/06_information_theory_basics.md) · [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md)
> **Key Papers:** Radford et al. 2018 GPT (CLM) · Devlin et al. 2018 BERT (MLM) ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805)) · Raffel et al. 2020 T5 (span corruption) ([arXiv:1910.10683](https://arxiv.org/abs/1910.10683)) · Tay et al. 2022 UL2 ([arXiv:2205.05131](https://arxiv.org/abs/2205.05131)) · Bavarian et al. 2022 FIM ([arXiv:2207.14255](https://arxiv.org/abs/2207.14255))

## Overview
The pretraining objective defines *what signal* the model extracts from unlabeled text. The field has overwhelmingly converged on **causal language modeling (CLM)** — autoregressive next-token prediction — for generative LLMs, because it is a universal self-supervised task that scales cleanly, supports in-context learning, and unifies pretraining with generation. But the design space (masked LM, prefix-LM, span corruption, fill-in-the-middle, multi-token prediction) is worth understanding both historically and because elements are re-entering frontier recipes.

## Core Concepts
**Causal LM (CLM).** Factorize the joint via the chain rule and maximize log-likelihood:
$$\mathcal{L}_{CLM}=-\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i})$$
This is exactly cross-entropy / minimizing KL to the data (see [Information Theory](../00_foundations/06_information_theory_basics.md)). Every token is both an input and a label, giving dense supervision; combined with causal masking it enables single-pass parallel training over the sequence.

**Masked LM (MLM).** BERT masks ~15% of tokens and predicts them from *bidirectional* context: $\mathcal{L}_{MLM}=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})$. Excellent for representations/understanding but (a) wastes supervision (only masked tokens contribute loss) and (b) is not directly generative. MLM lost to CLM for LLMs but remains relevant for encoders/embeddings.

**Prefix-LM.** A hybrid: bidirectional attention over a prefix, causal generation of the suffix (used in UL2, some Gemini-era encoders). Bridges understanding and generation.

**Span corruption (T5).** Mask contiguous spans, replace with sentinel tokens, and have a decoder generate the missing spans. Effective for encoder-decoder transfer learning.

**UL2 / Mixture-of-Denoisers.** Tay et al. (2022) train one model on a *mixture* of objectives — short-span (R-denoising), long-span (X-denoising), and prefix-LM (S-denoising) — switching via mode tokens, aiming to combine the strengths of each.

**Fill-in-the-Middle (FIM).** Bavarian et al. (2022): randomly split a document into prefix/middle/suffix and reorder as `<PRE> prefix <SUF> suffix <MID> middle`, so a *causal* model learns infilling without architectural change. Essential for code completion (IDE insert-in-the-middle). Now standard in code models with a typical 50–90% FIM rate during pretraining.

**Multi-Token Prediction (MTP).** Predict the next $k$ tokens (extra heads) as an auxiliary objective; improves data efficiency and yields a built-in draft model for speculative decoding (DeepSeek-V3, Gloeckle et al. 2024).

## Key Challenges
- **Supervision efficiency.** MLM trains on only the masked fraction; CLM trains on every token — a major reason CLM scales better.
- **Exposure bias.** Teacher-forcing trains on ground-truth prefixes but inference conditions on the model's own (possibly erroneous) outputs.
- **Objective–capability mismatch.** Next-token prediction optimizes local likelihood, not long-horizon coherence, planning, or factuality.
- **Infilling vs left-to-right.** Pure CLM can't natively infill; FIM must be baked in at pretraining (hard to add later cheaply).

## Solutions & Current Best Practices
**CLM is the universal default** for generative LLMs, augmented with **FIM** for code and increasingly **MTP** as an auxiliary head. Encoder/embedding models still use MLM-style objectives (or contrastive). Document **packing** (concatenating documents to fill the context, with appropriate attention/document masking) maximizes token utilization. Z-loss (penalizing the log-partition) stabilizes the softmax (see [Training Stability](02_training_stability.md)).

## Lab Perspectives
- **OpenAI** standardized CLM (GPT line) and pioneered FIM for Codex/code models.
- **Google DeepMind** explored the broadest objective space (T5 span corruption, UL2, prefix-LM) and folds elements into Gemini.
- **Meta** uses CLM + FIM (Code Llama) and researched MTP.
- **DeepSeek** adopted MTP as a core training objective in V3 (efficiency + speculative decoding synergy).

## Latest Developments (2023–2026)
**MTP** moved from research to production (DeepSeek-V3). Renewed interest in **objective mixtures** for data efficiency under the data wall. Reasoning-era post-training has shifted the *capability frontier* to RL on top of CLM pretraining, but the **pretraining objective itself remains CLM** — debates continue on whether richer pretraining objectives (infilling, denoising, MTP) materially help at frontier scale.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why did CLM beat MLM for LLMs?"** Dense supervision (every token), native generation, in-context learning, clean scaling.
- **"How does FIM work without changing the architecture?"** Reorder prefix/middle/suffix with sentinels so a causal model learns infilling.
- **"What is multi-token prediction and why is it useful?"** Predict $k$ future tokens; better data efficiency + free draft model for speculative decoding.
- **"What is exposure bias?"** Train on gold prefixes, infer on own outputs → compounding errors.

## Open Problems
Whether next-token prediction is *sufficient* for robust reasoning/planning (vs requiring RL or new objectives) is the field's central debate. The value of objective mixtures and MTP at the largest scales is not fully settled, and there is no accepted objective that directly targets long-horizon coherence or factuality during pretraining.

## References
- Devlin, J. et al. (2018). *BERT.* arXiv:1810.04805.
- Raffel, C. et al. (2020). *T5.* arXiv:1910.10683.
- Tay, Y. et al. (2022). *UL2.* arXiv:2205.05131.
- Bavarian, M. et al. (2022). *Efficient Training of LMs to Fill in the Middle.* arXiv:2207.14255.
- Gloeckle, F. et al. (2024). *Better & Faster LLMs via Multi-Token Prediction.* arXiv:2404.19737.
