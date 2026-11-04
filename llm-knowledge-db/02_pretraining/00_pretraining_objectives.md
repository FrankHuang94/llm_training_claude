# Pretraining Objectives

> **Last Updated:** 2026-11-04
> **Related Files:** [Transformer Architecture](../00_foundations/00_transformer_architecture.md) · [Information Theory Basics](../00_foundations/06_information_theory_basics.md) · [Model Architecture Variants](../00_foundations/05_model_architecture_variants.md)
> **Key Papers:** Radford et al. 2018 GPT (CLM) · Devlin et al. 2018 BERT (MLM) ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805)) · Raffel et al. 2020 T5 (span corruption) ([arXiv:1910.10683](https://arxiv.org/abs/1910.10683)) · Tay et al. 2022 UL2 ([arXiv:2205.05131](https://arxiv.org/abs/2205.05131)) · Bavarian et al. 2022 FIM ([arXiv:2207.14255](https://arxiv.org/abs/2207.14255))

## Overview
The pretraining objective defines *what signal* the model extracts from unlabeled text. The field has overwhelmingly converged on **causal language modeling (CLM)** — autoregressive next-token prediction — for generative LLMs, because it is a universal self-supervised task that scales cleanly, supports in-context learning, and unifies pretraining with generation. But the design space (masked LM, prefix-LM, span corruption, fill-in-the-middle, multi-token prediction, discrete diffusion) is worth understanding both historically and because elements are re-entering frontier recipes.

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

**Masked (discrete) diffusion LM.** Sample a mask ratio $t\sim U(0,1)$, mask each token independently with probability $t$ to get $x_t$, and train a *bidirectional* Transformer to recover the masked tokens: $\mathcal{L}=-\mathbb{E}_{t,x_t}\big[\tfrac{1}{t}\sum_{i:\,x_t^i=[\text{M}]}\log p_\theta(x_0^i\mid x_t)\big]$, an upper bound on the negative log-likelihood. Generation iteratively unmasks many tokens in parallel per step. It is a re-weighted, multi-ratio generalization of MLM that *is* generative.

## Key Challenges
- **Supervision efficiency.** MLM trains on only the masked fraction; CLM trains on every token — a major reason CLM scales better.
- **Exposure bias.** Teacher-forcing trains on ground-truth prefixes but inference conditions on the model's own (possibly erroneous) outputs.
- **Objective–capability mismatch.** Next-token prediction optimizes local likelihood, not long-horizon coherence, planning, or factuality.
- **Infilling vs left-to-right.** Pure CLM can't natively infill; FIM must be baked in at pretraining (hard to add later cheaply).
- **Parallel decoding vs likelihood.** Diffusion LMs decode many tokens per step but still trail AR models in likelihood at equal compute, and their best-likelihood variant is not always their best-reasoning variant.

## Solutions & Current Best Practices
**CLM is the universal default** for generative LLMs, augmented with **FIM** for code and **MTP** as an auxiliary head — DeepSeek-V3-style MTP modules (training signal + speculative draft) are now standard in Qwen3-Next, GLM-4.5/5 and similar open models. Encoder/embedding models still use MLM-style objectives (or contrastive). Document **packing** (concatenating documents to fill the context, with appropriate attention/document masking) maximizes token utilization. Z-loss (penalizing the log-partition) stabilizes the softmax (see [Training Stability](02_training_stability.md)). Diffusion LMs are increasingly obtained by **converting a pretrained AR model** rather than training from scratch.

## Lab Perspectives
- **OpenAI** standardized CLM (GPT line) and pioneered FIM for Codex/code models.
- **Google DeepMind** explored the broadest objective space (T5 span corruption, UL2, prefix-LM) and folds elements into Gemini; it showed an experimental text-diffusion model (Gemini Diffusion, 2025-05), reportedly at Gemini 2.0 Flash-Lite level, with no public update since.
- **Meta** uses CLM + FIM (Code Llama) and researched MTP.
- **DeepSeek** adopted MTP as a core training objective in V3 (efficiency + speculative decoding synergy); **Qwen** and **Zhipu** followed in Qwen3-Next and GLM-4.5/5.
- **Inception Labs** and **inclusionAI (Ant)** are the most visible bets on diffusion LMs at commercial / 100B scale.

## Latest Developments (2023–Nov 2026)
**MTP** moved from research to production (DeepSeek-V3). Renewed interest in **objective mixtures** for data efficiency under the data wall. Reasoning-era post-training has shifted the *capability frontier* to RL on top of CLM pretraining, but the **pretraining objective itself remains CLM** — debates continue on whether richer pretraining objectives (infilling, denoising, MTP) materially help at frontier scale.

**Mid-2025 → Nov 2026 update:**
- **MTP becomes default plumbing (2025–26).** DeepSeek-V3-style sequential MTP modules ship in Qwen3-Next and GLM-4.5/5, doing double duty as a denser training signal and as a native speculative draft (see [Speculative Decoding](../06_efficiency/04_speculative_decoding.md)).
- **Beyond next-$k$-token targets (2025-10).** *Future Summary Prediction* (ICLR 2026) trains an auxiliary head to predict a compact summary of the long-range future rather than the next $k$ tokens, reporting gains over both NTP and MTP at 3B/8B on math, reasoning and code ([arXiv:2510.14751](https://arxiv.org/abs/2510.14751)).
- **MTP by self-distillation (2026-02).** A pretrained NTP model is converted into a standalone MTP decoder with an online distillation objective, reporting >3× faster decoding on GSM8K at <5% accuracy loss ([arXiv:2602.06019](https://arxiv.org/abs/2602.06019)) — MTP can be retrofitted, not only pretrained.
- **Diffusion LMs reach commercial scale (2025-06 →).** Mercury (Inception Labs) ships diffusion LMs predicting many tokens per step ([arXiv:2506.17298](https://arxiv.org/abs/2506.17298)); Mercury 2 (2026-02) is pitched as a diffusion *reasoning* LM at a self-reported ~1,000 tok/s on Blackwell.
- **AR→diffusion conversion at 100B (2025-12).** LLaDA2.0 converts pretrained AR MoE models (16B mini, 100B flash) with a block-diffusion schedule: growing block size (warm-up) → full-sequence diffusion (stable) → small blocks (decay), plus an auxiliary confidence loss for aggressive parallel decoding ([arXiv:2512.15745](https://arxiv.org/abs/2512.15745)).
- **Diffusion scaling laws (2026-02).** Comparing masked, uniform-state and interpolating diffusion against AR: masked diffusion scales best in likelihood among diffusion variants, but uniform-state diffusion does better on GSM8K despite worse perplexity — likelihood is an imperfect proxy across objective families ([arXiv:2602.15014](https://arxiv.org/abs/2602.15014)).
- **Midtraining as an objective-level lever (2025-10 / 2026-03).** The timing of specialized data interacts with its mixture weight ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865)), and PRISM finds science data at midtraining unlocks +17–28 GPQA-Diamond points after RL while RL data-mix changes move results <2 points ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)) — the CLM phase, not RL, largely sets the ceiling.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why did CLM beat MLM for LLMs?"** Dense supervision (every token), native generation, in-context learning, clean scaling.
- **"How does FIM work without changing the architecture?"** Reorder prefix/middle/suffix with sentinels so a causal model learns infilling.
- **"What is multi-token prediction and why is it useful?"** Predict $k$ future tokens; better data efficiency + free draft model for speculative decoding.
- **"What is exposure bias?"** Train on gold prefixes, infer on own outputs → compounding errors.
- **"AR vs masked-diffusion LMs — what do you trade?"** Diffusion: parallel, bidirectional decoding and native infilling, but a looser likelihood bound and quality that degrades as more tokens are committed per step; AR: exact likelihood, KV caching, mature serving. Conversion from AR checkpoints (LLaDA2.0) is how diffusion LMs currently reach scale.

## Open Problems
Whether next-token prediction is *sufficient* for robust reasoning/planning (vs requiring RL or new objectives) is the field's central debate. The value of objective mixtures and MTP at the largest scales is not fully settled, and there is no accepted objective that directly targets long-horizon coherence or factuality during pretraining. Whether diffusion LMs can match AR quality on long reasoning (rather than only speed), and how much of capability is set by midtraining data versus the objective itself, are open.

## References
- Devlin, J. et al. (2018). *BERT.* arXiv:1810.04805.
- Raffel, C. et al. (2020). *T5.* arXiv:1910.10683.
- Tay, Y. et al. (2022). *UL2.* arXiv:2205.05131.
- Bavarian, M. et al. (2022). *Efficient Training of LMs to Fill in the Middle.* arXiv:2207.14255.
- Gloeckle, F. et al. (2024). *Better & Faster LLMs via Multi-Token Prediction.* arXiv:2404.19737.
- Nie, S. et al. (2025). *Large Language Diffusion Models (LLaDA).* arXiv:2502.09992.
- Inception Labs (2025). *Mercury: Ultra-Fast Language Models Based on Diffusion.* arXiv:2506.17298.
- inclusionAI (2025). *LLaDA2.0.* arXiv:2512.15745.
