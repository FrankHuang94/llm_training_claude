# Tokenization

> **Last Updated:** 2026-06-18
> **Related Files:** [Tokenizer Training](../01_data/02_tokenizer_training.md) · [Transformer Architecture](00_transformer_architecture.md) · [Information Theory Basics](06_information_theory_basics.md)
> **Key Papers:** Sennrich et al. 2016 BPE ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) · Kudo 2018 Unigram LM ([arXiv:1804.10959](https://arxiv.org/abs/1804.10959)) · Kudo & Richardson 2018 SentencePiece ([arXiv:1808.06226](https://arxiv.org/abs/1808.06226)) · Radford et al. 2019 GPT-2 (byte-level BPE) · Xue et al. 2022 ByT5 ([arXiv:2105.13626](https://arxiv.org/abs/2105.13626))

## Overview
Tokenization is the lossy-but-reversible interface between raw bytes and the integer sequences a Transformer consumes. It determines the model's effective vocabulary, sequence length (and thus compute cost), and a surprising amount of downstream behavior — arithmetic, spelling, multilingual fairness, and code handling all hinge on tokenizer design. Because the LM operates over tokens, *the tokenizer defines the units over which the probability model is factorized*, and many "model" failures are really tokenizer artifacts.

Modern LLMs use **subword** tokenization, a compromise between character-level (long sequences, no OOV, weak units) and word-level (huge vocab, OOV problems). Subword methods learn a vocabulary that balances sequence compression against vocabulary size, typically 32K–256K tokens.

## Core Concepts
**Byte-Pair Encoding (BPE).** Starts from characters/bytes and greedily merges the most frequent adjacent pair, repeating until the target vocab size is reached. Encoding applies the learned merge rules in order. BPE is deterministic and frequency-driven. **Byte-level BPE** (GPT-2 onward) operates on raw UTF-8 bytes, guaranteeing zero OOV for any input (any Unicode string is representable) at the cost of multi-token characters for non-Latin scripts.

**WordPiece** (BERT) is BPE-like but merges the pair maximizing likelihood of the training corpus under a unigram model rather than raw frequency: choose the merge maximizing $\frac{\text{count}(xy)}{\text{count}(x)\text{count}(y)}$.

**Unigram LM** (Kudo 2018) takes the opposite, top-down approach: start with a large candidate vocabulary and iteratively prune tokens that least hurt corpus likelihood under a unigram model, using EM. It yields a probabilistic segmentation, enabling **subword regularization** (sampling alternative segmentations during training for robustness).

**SentencePiece** is an implementation (not an algorithm) wrapping BPE/Unigram that operates directly on raw text with no pre-tokenization, treating whitespace as a meta-symbol (`▁`). This makes it language-agnostic — crucial for languages without whitespace (Chinese, Japanese, Thai).

**Fertility** = average tokens per word (or per byte). High fertility means a language is "expensive" — more tokens per unit of meaning, higher cost, shorter effective context. English typically ~1.3 tokens/word; many low-resource languages see 2–4× higher fertility under English-centric tokenizers, a fairness and cost issue.

## Key Challenges
- **Multilingual inequity.** English-optimized vocabularies over-fragment other scripts, inflating cost and latency for non-English users and degrading quality.
- **Arithmetic/spelling brittleness.** Inconsistent number splitting (e.g., "327" as one token vs "3","27") harms arithmetic; character-level tasks are hard because tokens hide characters.
- **The "tokenization is a bottleneck" debate.** Tokenizers are frozen artifacts trained before the model, create glitch tokens (e.g., `SolidGoldMagikarp`), and complicate domain adaptation.
- **Vocabulary–sequence tradeoff.** Larger vocab compresses sequences (less compute) but enlarges the embedding/softmax and risks under-trained rare tokens.

## Solutions & Current Best Practices
Current practice: **byte-level BPE or Unigram via SentencePiece/`tiktoken`**, vocab 100K–256K for frontier multilingual models (GPT-4o uses ~200K; Llama 3 uses 128K, up from Llama 2's 32K — a deliberate efficiency/multilingual gain). Digits are often split into individual tokens or fixed 3-digit chunks to aid arithmetic. Reserved/special tokens encode chat roles and tool calls. Tokenizer training data should mirror the pretraining mixture to balance fertility across languages and code.

## Lab Perspectives
- **OpenAI** ships `tiktoken` (byte-level BPE); GPT-4o's larger multilingual vocab notably cut non-English token counts.
- **Meta** expanded Llama 3's vocab to 128K, citing efficiency and multilingual coverage.
- **Google** uses SentencePiece (Gemini/Gemma) with large multilingual vocabularies.
- **DeepSeek/Mistral** use byte-level BPE tuned for code-heavy and multilingual mixtures.
- A research undercurrent across labs (Meta's BLT, DeepMind) pushes toward **tokenizer-free / byte-level** models.

## Latest Developments (2023–2026)
**Byte Latent Transformer (BLT)** (Meta, 2024) replaces fixed tokens with dynamically-sized byte patches based on entropy, aiming to eliminate the tokenizer entirely while matching token-level efficiency. **MegaByte**, **MambaByte**, and entropy-based patching explore the same frontier. Multilingual vocab expansion and per-language fertility audits are now standard pre-training hygiene.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Compare BPE, WordPiece, and Unigram."** Bottom-up frequency vs likelihood-merge vs top-down pruning; which enables subword regularization.
- **"Why is byte-level BPE used, and what does it cost?"** Zero OOV / any-Unicode vs multi-token non-Latin characters.
- **"Why are LLMs bad at arithmetic/spelling?"** Tokenization hides characters and splits numbers inconsistently.
- **"How would you make a tokenizer fairer across languages?"** Balance training mixture, measure fertility, expand vocab, consider byte-level.

## Open Problems
Whether end-to-end **tokenizer-free** models can match subword efficiency at frontier scale is unresolved. Optimal vocabulary size as a function of model/data scale lacks a clean scaling law. Adapting a frozen tokenizer to new domains/languages without retraining remains awkward, and glitch-token robustness is not fully understood.

## References
- Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units (BPE).* arXiv:1508.07909.
- Kudo, T. (2018). *Subword Regularization (Unigram LM).* arXiv:1804.10959.
- Kudo, T., Richardson, J. (2018). *SentencePiece.* arXiv:1808.06226.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2).*
- Pagnoni, A. et al. (2024). *Byte Latent Transformer.* arXiv:2412.09871.
