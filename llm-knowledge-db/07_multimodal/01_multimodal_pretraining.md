# Multimodal Pretraining

> **Last Updated:** 2026-11-04
> **Related Files:** [Vision-Language Models](00_vision_language_models.md) · [Cross-Modal Alignment](02_cross_modal_alignment.md) · [Data Collection & Curation](../01_data/00_data_collection_and_curation.md) · [World Models](../09_emerging_frontiers/02_world_models.md)
> **Key Papers:** Jia et al. 2021 ALIGN ([arXiv:2102.05918](https://arxiv.org/abs/2102.05918)) · Schuhmann et al. 2022 LAION-5B ([arXiv:2210.08402](https://arxiv.org/abs/2210.08402)) · Google 2023 Gemini ([arXiv:2312.11805](https://arxiv.org/abs/2312.11805)) · Team Chameleon 2024 (early-fusion) ([arXiv:2405.09818](https://arxiv.org/abs/2405.09818)) · Assran et al. 2025 V-JEPA 2 ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985))

## Overview
Multimodal pretraining is where a model first learns the joint structure of multiple modalities at scale. There are two broad philosophies: **late-fusion / connector** approaches that pretrain unimodal encoders (CLIP-style) and stitch them together (see [VLMs](00_vision_language_models.md)), and **early-fusion / native** approaches that train a single Transformer on **interleaved multimodal token streams from scratch** (Gemini, Chameleon). The shift toward native multimodal pretraining — treating images, audio, and video as just more tokens — is one of the defining architectural trends of 2024–2026, enabling **any-to-any** understanding and generation; by 2026 it reached trillion-parameter open MoEs (Kimi K3) and omni models that report no modality degradation (Qwen3-Omni). A third, non-generative line — **self-supervised latent prediction** on video (V-JEPA 2) — pretrains perception for planning and robotics. The fuel is web-scale paired data (image-text, video-text, audio-text) plus massive unlabeled video.

## Core Concepts
**Data sources.**
- **Image-text:** **LAION-5B** (5.8B CLIP-filtered web image-text pairs — the open backbone of CLIP/diffusion training), **ALIGN** (1.8B noisy alt-text pairs, showing scale beats heavy curation), **CC12M**, **DataComp**, **COYO**. Interleaved image-text documents (MMC4, OBELICS) provide *document-level* multimodal context (vital for few-shot, Flamingo-style).
- **Video-text:** WebVid, HowTo100M, InternVid; video adds the **temporal** dimension and huge token counts. Unlabeled internet video (>1M hours for V-JEPA 2) is used directly for self-supervised objectives.
- **Audio:** large speech corpora (LibriSpeech, Common Voice) and web audio; Whisper used 680K hours of weakly-labeled web audio.

**Tokenizing non-text modalities.** To put a modality into a Transformer you need *tokens*:
- **Continuous embeddings** (ViT patches, audio encoder features) fed via a connector — preserves information, used for *understanding*.
- **Discrete tokens** via a **VQ-VAE / VQ-GAN / tokenizer** (e.g., image → codebook indices, audio → SoundStream/EnCodec codes) — lets the model *generate* the modality autoregressively with the same next-token objective (Chameleon, audio LMs).

**Fusion strategies.**
- **Late fusion (connector):** separate encoders + projection/cross-attention; modular, reuses strong unimodal pretraining, easier to build.
- **Early fusion (native):** one Transformer over mixed-modality token sequences from the start (Gemini, Chameleon's "mixed-modal early fusion"). Tighter integration and native cross-modal generation, but harder to train and stabilize (competing modalities, balancing).

**Objectives.** Contrastive (CLIP/ALIGN — alignment), generative/autoregressive (next-token over multimodal tokens — understanding + generation), masked modeling (BEiT/MAE), **joint-embedding predictive (JEPA)** — predict the *representation* of masked video regions from visible context, $\min \lVert g_\phi(f_\theta(x_{\text{ctx}}), m) - \bar f(x_{\text{tgt}})\rVert_1$ with an EMA target encoder $\bar f$, avoiding pixel reconstruction — and combinations. Native models typically use the **autoregressive** objective over interleaved tokens.

## Key Challenges
- **Data quality & alignment noise.** Web alt-text is noisy; filtering (CLIP-score, dedup) is essential but imperfect; high-quality paired data is scarcer than text.
- **Modality imbalance / interference.** Modalities differ in token counts, information density, and difficulty; naive joint training lets one dominate or causes negative transfer (the "modality tax" on text quality).
- **Token explosion.** Images (high-res) and especially video produce enormous token counts → compute/context pressure.
- **Tokenizer quality (for generation).** Discrete visual/audio tokenizers lose information; reconstruction quality bounds generation quality.
- **Training stability.** Early-fusion multimodal training is less stable than text-only (Chameleon documented special norms/init).
- **Release divergence.** Open checkpoints may omit modalities that the hosted model has, complicating reproducibility.

## Solutions & Current Best Practices
**Late-fusion connector** training remains the practical default for *understanding*-focused VLMs (cheaper, modular, leverages CLIP/SigLIP + a strong LLM). **Native early-fusion** is the frontier for **any-to-any** models (Gemini, GPT-4o) and now for flagship open MoEs, using interleaved multimodal data, careful **modality balancing**, and stability tricks (QK-norm, dropout, modality-specific norms). **Heavy data filtering** (CLIP-score, dedup, DataComp-style) and **interleaved document data** for in-context multimodality. **Discrete tokenizers** (VQ/EnCodec) where cross-modal *generation* is needed. Quality classifiers and synthetic captions (re-captioning with a VLM) boost data quality. For embodied/planning use, **self-supervised video pretraining** followed by small action-conditioned post-training (V-JEPA 2-AC) is data-efficient.

## Lab Perspectives
- **Google DeepMind:** **Gemini** is natively multimodal (text/image/audio/video) from pretraining — the flagship early-fusion model; long-context video; **Veo 3** added native audio to video generation (2025-05); "Gemini Omni" announced at I/O 2026 (details not verified).
- **OpenAI:** **GPT-4o** "omni" — single model across text/image/audio with native generation (images, voice); CLIP/Whisper/DALL·E lineage; **Sora 2** (2025-09) generates video and audio jointly.
- **Meta:** **Chameleon** (open early-fusion mixed-modal), ImageBind (binding 6 modalities), LLaMA vision, **V-JEPA 2** (self-supervised video world model); the closed Muse Spark (2026-04) claims strong multimodal understanding at far lower compute (vendor).
- **Qwen/Alibaba:** **Qwen3-Omni** (Thinker–Talker MoE) and natively vision-language Qwen3.5; Qwen3.8 open weights text-only.
- **Other open labs:** LAION (data), OpenCLIP, InternVL, Emu/Janus (unified understanding+generation); Kimi K3, DeepSeek V4.1-Flash, MiMo-V2.5 and MiniMax M3 ship native multimodality.

## Latest Developments (2023–Nov 2026)
**Native any-to-any** models (GPT-4o, Gemini) handling and *generating* multiple modalities in one Transformer. **Unified understanding + generation** (Chameleon, Emu3, Janus) via discrete multimodal tokens. **Re-captioning** web images with VLMs to improve data quality (DALL·E 3, PixArt). **Video** as a first-class modality (long-context Gemini, video tokenizers). **Audio-native** models (real-time voice, GPT-4o voice). Debate over early- vs late-fusion and over discrete-vs-continuous representations for unified models.

**Mid-2025 → Nov 2026 update:**
- **V-JEPA 2 (2025-06, arXiv:2506.09985).** Self-supervised JEPA pretraining on >1M hours of internet video reaches 77.3% on Something-Something v2; **V-JEPA 2-AC**, post-trained on <62 hours of unlabeled DROID robot video, plans zero-shot pick-and-place on Franka arms via model-predictive control in latent space — evidence that non-generative video pretraining transfers to control with tiny action data.
- **Qwen3-Omni (2025-09, arXiv:2509.17765).** A 30B-A3B Thinker–Talker MoE jointly trained over text, image, audio and video; the authors report **no modality degradation** versus single-modal Qwen models — a counterexample to the assumed "modality tax" at this scale.
- **Joint audio-video generation (2025-05 → 2025-09).** Veo 3 introduced native audio; Sora 2 generates synchronized dialogue and sound effects with improved physical plausibility — pretraining targets now couple modalities at the sample level, not just in a shared embedding space.
- **Native multimodality reaches trillion-scale open MoEs (2026).** Qwen3.5-397B-A17B (2026-02, native VL, 512 experts); MiMo-V2.5 (2026-04, native vision+audio, 1M context); MiniMax M3 (~2026-06, native multimodality with block-sparse attention); Kimi K3 (2026-07, 2.8T/104B, 896 experts with 16 active, native vision, 1M context); DeepSeek V4.1-Flash (2026-09, 552B, native vision). The 1M-token context that hybrid/sparse attention made affordable is what makes long interleaved multimodal streams practical.
- **Open checkpoints diverge from hosted models (2026-08).** Qwen3.8-2.4T-A95B's open weights are text-only (thinking always on) while the hosted Max-Preview is multimodal — the expensive multimodal stack is sometimes kept proprietary.
- **Efficiency claims shift the frame.** Meta claims Muse Spark matches or beats Llama 4 Maverick-level reasoning with far less compute and is strongest in multimodal understanding (vendor); Gemma 4 brings text/image/audio pretraining to 2B–4B edge models.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Early fusion vs late fusion — tradeoffs?"** Native joint training (tight integration, any-to-any generation, harder/unstable) vs connector on pretrained encoders (modular, cheaper, understanding-focused).
- **"How do you put an image/audio into a Transformer?"** Continuous encoder features (understanding) or discrete VQ/EnCodec tokens (generation).
- **"Why did ALIGN/LAION work with noisy data?"** Scale compensates for noise (with CLIP-score filtering); contrastive learning is robust to some noise.
- **"Main challenge in joint multimodal pretraining?"** Modality imbalance/interference and token explosion; needs balancing + stability tricks.
- **"How do unified gen+understanding models represent images?"** Discrete codebook tokens enable autoregressive generation alongside text.
- **"JEPA latent prediction vs pixel/token generation as a video pretraining objective?"** JEPA predicts representations, ignoring unpredictable pixel detail and spending capacity on dynamics/semantics; it can't generate video but transfers efficiently to planning (V-JEPA 2-AC). Generative objectives give samples and dense supervision at much higher cost.
- **"Is there a 'modality tax' on text ability?"** Historically yes (interference); Qwen3-Omni reports none at 30B-A3B with careful mixing — discuss data ratios, capacity (MoE helps), and evaluation of text-only regressions.

## Open Problems
The best **representation** (continuous vs discrete) and **fusion** (early vs late) for unified multimodal models is unsettled. **Modality balancing**, stable large-scale early-fusion training, efficient **video/audio** tokenization, whether generative or latent-predictive objectives yield better world models, and whether a single model can match specialized unimodal models across all modalities are open. High-quality multimodal **data scarcity** is a growing constraint.

## References
- Jia, C. et al. (2021). *ALIGN.* arXiv:2102.05918.
- Schuhmann, C. et al. (2022). *LAION-5B.* arXiv:2210.08402.
- Gemini Team (2023). *Gemini.* arXiv:2312.11805.
- Chameleon Team (2024). *Chameleon: Mixed-Modal Early-Fusion.* arXiv:2405.09818.
- Radford, A. et al. (2022). *Whisper.* arXiv:2212.04356.
- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985.
- Qwen Team (2025). *Qwen3-Omni Technical Report.* arXiv:2509.17765.
