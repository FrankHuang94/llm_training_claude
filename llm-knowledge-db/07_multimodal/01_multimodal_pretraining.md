# Multimodal Pretraining

> **Last Updated:** 2026-06-18
> **Related Files:** [Vision-Language Models](00_vision_language_models.md) · [Cross-Modal Alignment](02_cross_modal_alignment.md) · [Data Collection & Curation](../01_data/00_data_collection_and_curation.md)
> **Key Papers:** Jia et al. 2021 ALIGN ([arXiv:2102.05918](https://arxiv.org/abs/2102.05918)) · Schuhmann et al. 2022 LAION-5B ([arXiv:2210.08402](https://arxiv.org/abs/2210.08402)) · Google 2023 Gemini ([arXiv:2312.11805](https://arxiv.org/abs/2312.11805)) · Team Chameleon 2024 (early-fusion) ([arXiv:2405.09818](https://arxiv.org/abs/2405.09818))

## Overview
Multimodal pretraining is where a model first learns the joint structure of multiple modalities at scale. There are two broad philosophies: **late-fusion / connector** approaches that pretrain unimodal encoders (CLIP-style) and stitch them together (see [VLMs](00_vision_language_models.md)), and **early-fusion / native** approaches that train a single Transformer on **interleaved multimodal token streams from scratch** (Gemini, Chameleon). The shift toward native multimodal pretraining — treating images, audio, and video as just more tokens — is one of the defining architectural trends of 2024–2026, enabling **any-to-any** understanding and generation. The fuel is web-scale paired data (image-text, video-text, audio-text).

## Core Concepts
**Data sources.**
- **Image-text:** **LAION-5B** (5.8B CLIP-filtered web image-text pairs — the open backbone of CLIP/diffusion training), **ALIGN** (1.8B noisy alt-text pairs, showing scale beats heavy curation), **CC12M**, **DataComp**, **COYO**. Interleaved image-text documents (MMC4, OBELICS) provide *document-level* multimodal context (vital for few-shot, Flamingo-style).
- **Video-text:** WebVid, HowTo100M, InternVid; video adds the **temporal** dimension and huge token counts.
- **Audio:** large speech corpora (LibriSpeech, Common Voice) and web audio; Whisper used 680K hours of weakly-labeled web audio.

**Tokenizing non-text modalities.** To put a modality into a Transformer you need *tokens*:
- **Continuous embeddings** (ViT patches, audio encoder features) fed via a connector — preserves information, used for *understanding*.
- **Discrete tokens** via a **VQ-VAE / VQ-GAN / tokenizer** (e.g., image → codebook indices, audio → SoundStream/EnCodec codes) — lets the model *generate* the modality autoregressively with the same next-token objective (Chameleon, audio LMs).

**Fusion strategies.**
- **Late fusion (connector):** separate encoders + projection/cross-attention; modular, reuses strong unimodal pretraining, easier to build.
- **Early fusion (native):** one Transformer over mixed-modality token sequences from the start (Gemini, Chameleon's "mixed-modal early fusion"). Tighter integration and native cross-modal generation, but harder to train and stabilize (competing modalities, balancing).

**Objectives.** Contrastive (CLIP/ALIGN — alignment), generative/autoregressive (next-token over multimodal tokens — understanding + generation), masked modeling (BEiT/MAE), and combinations. Native models typically use the **autoregressive** objective over interleaved tokens.

## Key Challenges
- **Data quality & alignment noise.** Web alt-text is noisy; filtering (CLIP-score, dedup) is essential but imperfect; high-quality paired data is scarcer than text.
- **Modality imbalance / interference.** Modalities differ in token counts, information density, and difficulty; naive joint training lets one dominate or causes negative transfer.
- **Token explosion.** Images (high-res) and especially video produce enormous token counts → compute/context pressure.
- **Tokenizer quality (for generation).** Discrete visual/audio tokenizers lose information; reconstruction quality bounds generation quality.
- **Training stability.** Early-fusion multimodal training is less stable than text-only (Chameleon documented special norms/init).

## Solutions & Current Best Practices
**Late-fusion connector** training remains the practical default for *understanding*-focused VLMs (cheaper, modular, leverages CLIP/SigLIP + a strong LLM). **Native early-fusion** is the frontier for **any-to-any** models (Gemini, GPT-4o), using interleaved multimodal data, careful **modality balancing**, and stability tricks (QK-norm, dropout, modality-specific norms). **Heavy data filtering** (CLIP-score, dedup, DataComp-style) and **interleaved document data** for in-context multimodality. **Discrete tokenizers** (VQ/EnCodec) where cross-modal *generation* is needed. Quality classifiers and synthetic captions (re-captioning with a VLM) boost data quality.

## Lab Perspectives
- **Google DeepMind:** **Gemini** is natively multimodal (text/image/audio/video) from pretraining — the flagship early-fusion model; long-context video.
- **OpenAI:** **GPT-4o** "omni" — single model across text/image/audio with native generation (images, voice); CLIP/Whisper/DALL·E lineage.
- **Meta:** **Chameleon** (open early-fusion mixed-modal), ImageBind (binding 6 modalities), LLaMA vision; strong open multimodal research.
- **Open community:** LAION (data), OpenCLIP, Qwen-VL/InternVL, Emu/Janus (unified understanding+generation).

## Latest Developments (2023–2026)
**Native any-to-any** models (GPT-4o, Gemini) handling and *generating* multiple modalities in one Transformer. **Unified understanding + generation** (Chameleon, Emu3, Janus) via discrete multimodal tokens. **Re-captioning** web images with VLMs to improve data quality (DALL·E 3, PixArt). **Video** as a first-class modality (long-context Gemini, video tokenizers). **Audio-native** models (real-time voice, GPT-4o voice). Debate over early- vs late-fusion and over discrete-vs-continuous representations for unified models.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Early fusion vs late fusion — tradeoffs?"** Native joint training (tight integration, any-to-any generation, harder/unstable) vs connector on pretrained encoders (modular, cheaper, understanding-focused).
- **"How do you put an image/audio into a Transformer?"** Continuous encoder features (understanding) or discrete VQ/EnCodec tokens (generation).
- **"Why did ALIGN/LAION work with noisy data?"** Scale compensates for noise (with CLIP-score filtering); contrastive learning is robust to some noise.
- **"Main challenge in joint multimodal pretraining?"** Modality imbalance/interference and token explosion; needs balancing + stability tricks.
- **"How do unified gen+understanding models represent images?"** Discrete codebook tokens enable autoregressive generation alongside text.

## Open Problems
The best **representation** (continuous vs discrete) and **fusion** (early vs late) for unified multimodal models is unsettled. **Modality balancing**, stable large-scale early-fusion training, efficient **video/audio** tokenization, and whether a single model can match specialized unimodal models across all modalities are open. High-quality multimodal **data scarcity** is a growing constraint.

## References
- Jia, C. et al. (2021). *ALIGN.* arXiv:2102.05918.
- Schuhmann, C. et al. (2022). *LAION-5B.* arXiv:2210.08402.
- Gemini Team (2023). *Gemini.* arXiv:2312.11805.
- Chameleon Team (2024). *Chameleon: Mixed-Modal Early-Fusion.* arXiv:2405.09818.
- Radford, A. et al. (2022). *Whisper.* arXiv:2212.04356.
