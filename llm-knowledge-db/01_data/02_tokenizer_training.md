# Tokenizer Training

> **Last Updated:** 2026-06-18
> **Related Files:** [Tokenization](../00_foundations/02_tokenization.md) · [Data Collection & Curation](00_data_collection_and_curation.md) · [Data Mixing & Curriculum](03_data_mixing_and_curriculum.md)
> **Key Papers:** Sennrich et al. 2016 BPE ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) · Kudo 2018 Unigram ([arXiv:1804.10959](https://arxiv.org/abs/1804.10959)) · Kudo & Richardson 2018 SentencePiece ([arXiv:1808.06226](https://arxiv.org/abs/1808.06226)) · Dagan et al. 2024 "Getting the most out of your tokenizer" ([arXiv:2402.01035](https://arxiv.org/abs/2402.01035))

## Overview
While [Tokenization](../00_foundations/02_tokenization.md) covers the *algorithms*, this file covers the *engineering decisions* in actually training a production tokenizer: vocabulary size, training-corpus composition, digit/whitespace handling, special tokens, and multilingual/code coverage. The tokenizer is trained **once, before the model**, on a representative sample of the pretraining mixture, and then frozen for the model's lifetime — so mistakes are expensive and nearly irreversible. A well-designed tokenizer improves effective context length, inference cost, and downstream performance on math and code "for free."

## Core Concepts
**Vocabulary size.** The central hyperparameter. Larger vocab → shorter sequences (less compute/longer effective context) but a bigger embedding + output softmax (more params, more under-trained rare tokens) and a larger memory footprint. Frontier multilingual models trend large: GPT-2 (50K) → LLaMA-2 (32K) → LLaMA-3 (128K) → GPT-4o (~200K) → some models 256K. There is a (weak) scaling intuition that larger models warrant larger vocabularies.

**Training corpus composition.** The tokenizer inherits the *biases of its training sample*. If trained on 95% English, non-English fertility (tokens/word) balloons. Best practice: sample the tokenizer-training corpus to **match the intended pretraining mixture** across languages, code, and math — possibly upweighting under-served scripts to equalize fertility.

**Digit handling.** A deliberate choice for arithmetic. Options: split every digit (Llama, PaLM), group fixed 3-digit chunks (some GPT variants), or left-to-right consistency. Single-digit tokenization gives the most consistent number representation and best arithmetic, at the cost of longer sequences for numbers.

**Whitespace & pre-tokenization.** GPT-style byte-level BPE uses a regex pre-tokenizer (the `tiktoken` pattern) to prevent merges across word boundaries and to handle whitespace/punctuation consistently. SentencePiece encodes whitespace as `▁`, avoiding language-specific pre-tokenization (vital for Chinese/Japanese/Thai).

**Special / reserved tokens.** Chat roles (`<|im_start|>`, `[INST]`), BOS/EOS, padding, tool-call delimiters, and FIM sentinels (`<|fim_prefix|>` etc.) are reserved. Reserving spare "unused" slots lets you add control tokens later without retraining the embedding table awkwardly.

## Key Challenges
- **Frozen-artifact problem.** The tokenizer can't adapt as the data distribution or target languages shift post-launch.
- **Multilingual fertility imbalance.** English-centric training penalizes other languages on cost, latency, and quality.
- **Code/math sensitivity.** Indentation, operators, and numbers are tokenized in ways that materially affect coding and arithmetic ability.
- **Glitch/under-trained tokens.** Rare tokens (e.g., scraped usernames like `SolidGoldMagikarp`) appear in the vocab but barely in training, producing bizarre behavior.

## Solutions & Current Best Practices
Train **byte-level BPE (tiktoken)** or **Unigram (SentencePiece)** on a **mixture-matched** corpus; choose vocab 64K–256K based on model scale and multilinguality; **split digits**; reserve ample special tokens including FIM and chat control tokens; **audit per-language fertility** and prune/avoid glitch tokens by filtering the tokenizer-training data. Dagan et al. (2024) show tokenizer choices measurably affect both inference cost and accuracy, recommending compression-aware evaluation before freezing.

## Lab Perspectives
- **OpenAI** ships `tiktoken`; the GPT-4o tokenizer markedly improved non-English efficiency over GPT-4's `cl100k`.
- **Meta** quadrupled Llama 3's vocabulary to 128K for efficiency and multilingual coverage.
- **Google** uses SentencePiece (Gemma/Gemini) with large multilingual vocabularies and byte fallback.
- **DeepSeek/Qwen/Mistral** train code-heavy, multilingual byte-level BPE tokenizers; Qwen notably emphasizes large multilingual vocab.

## Latest Developments (2023–2026)
Vocabulary expansion is now standard for new model generations. **Per-language fertility audits** and **compression-ratio benchmarks** are part of pre-training hygiene. The frontier explores **tokenizer-free / byte-level** models (Meta BLT, MambaByte) to escape the frozen-artifact problem, and **vocabulary scaling laws** (relating optimal vocab to model size, Tao et al. 2024) emerged to formalize the size choice.

## Interview Angles
> 💡 **What labs actually ask:**
- **"How would you choose vocabulary size?"** Sequence-compression vs embedding/softmax cost and rare-token under-training; scale with model size and multilinguality.
- **"How do you build a fair multilingual tokenizer?"** Mixture-matched (or fertility-equalized) training corpus; measure tokens/word per language.
- **"Why split digits?"** Consistent number representation → better arithmetic.
- **"What causes glitch tokens and how do you avoid them?"** Under-trained rare tokens from unfiltered scrape data; clean the tokenizer corpus.

## Open Problems
Optimal vocabulary size lacks a robust, universally accepted scaling law. Adapting or extending a frozen tokenizer to new domains/languages without destabilizing the model is unsolved. Whether tokenizer-free models can match subword efficiency and quality at frontier scale remains the key open question.

## References
- Sennrich, R. et al. (2016). *BPE.* arXiv:1508.07909.
- Kudo, T. (2018). *Subword Regularization (Unigram).* arXiv:1804.10959.
- Kudo, T., Richardson, J. (2018). *SentencePiece.* arXiv:1808.06226.
- Dagan, G. et al. (2024). *Getting the Most out of Your Tokenizer.* arXiv:2402.01035.
- Tao, C. et al. (2024). *Scaling Laws with Vocabulary.* arXiv:2407.13623.
