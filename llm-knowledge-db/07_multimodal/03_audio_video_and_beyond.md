# Audio, Video, and Beyond

> **Last Updated:** 2026-06-18
> **Related Files:** [Multimodal Pretraining](01_multimodal_pretraining.md) · [Vision-Language Models](00_vision_language_models.md) · [World Models](../09_emerging_frontiers/02_world_models.md)
> **Key Papers:** Radford et al. 2022 Whisper ([arXiv:2212.04356](https://arxiv.org/abs/2212.04356)) · Rubenstein et al. 2023 AudioPaLM ([arXiv:2306.12925](https://arxiv.org/abs/2306.12925)) · Défossez et al. 2024 Moshi ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037)) · Zhang et al. 2023 Video-LLaMA ([arXiv:2306.02858](https://arxiv.org/abs/2306.02858))

## Overview
Beyond images, frontier models increasingly handle **audio** (speech and general sound), **video** (the temporal visual stream), and other modalities (3D, depth, sensor data). These modalities push two frontiers: **scale** (video and audio produce enormous token counts) and **real-time interaction** (low-latency speech for natural voice assistants). The trend mirrors vision: pretrained encoders + LLM connectors for *understanding*, and discrete neural codecs for *generation*, converging toward **native any-to-any** models (GPT-4o voice, Gemini video). Audio and video also connect to the **world-model** question — modeling temporal dynamics (see [World Models](../09_emerging_frontiers/02_world_models.md)).

## Core Concepts
**Audio understanding.** **Whisper** (OpenAI) is the canonical speech model: an encoder-decoder Transformer trained on 680K hours of weakly-supervised web audio for robust multilingual ASR and translation, demonstrating that **scale + weak supervision** beats curated speech datasets. For LLM integration, an audio encoder (Whisper/Conformer) feeds features into the LLM via a connector (audio analogue of LLaVA).

**Audio tokenization & generation.** Neural audio codecs (**SoundStream**, **EnCodec**) compress waveforms into **discrete tokens** via residual vector quantization (RVQ), letting an LLM generate audio autoregressively. **AudioLM/AudioPaLM/VALL-E** generate speech as token sequences; this enables a unified text+speech model. **Moshi** (Kyutai) is a full-duplex, real-time speech-to-speech model achieving very low latency by modeling audio and text streams jointly.

**Video understanding.** Video = a sequence of frames + audio + time. Approaches sample frames, encode each with a ViT, and feed the (large) token sequence to the LLM, using **temporal modeling** (temporal attention, Q-Formers, or simply long context). The core problem is **token explosion**: a few minutes of video at reasonable frame rates is hundreds of thousands of tokens. **Long context** (Gemini 1.5/2.5 at 1M+ tokens) is a key enabler for hour-long video; token compression and keyframe selection help.

**Real-time/streaming.** Voice assistants need **low latency** and **full-duplex** (listen while speaking, handle interruptions). Native audio models (GPT-4o voice, Moshi, Gemini Live) generate speech directly rather than the slow ASR→LLM→TTS cascade, capturing prosody, emotion, and timing.

**Beyond.** **ImageBind** binds 6 modalities (image, text, audio, depth, thermal, IMU); robotics/embodied models add proprioception and action; world/video-generation models (Sora, Genie) learn dynamics.

## Key Challenges
- **Token explosion (video/audio).** Temporal data overwhelms context; efficient tokenization/compression and long context are essential.
- **Temporal reasoning.** Understanding events, causality, and order over time (not just per-frame content) is hard.
- **Latency for real-time.** Full-duplex, sub-300ms speech interaction requires streaming architectures and tight engineering.
- **Codec quality vs bitrate.** Discrete audio tokens trade fidelity for sequence length; reconstruction quality bounds generation.
- **Data & alignment.** Paired video/audio-text data is noisier and scarcer; temporal alignment (which words match which moment) is hard.
- **Evaluation.** Long-video and audio understanding/generation benchmarks are immature.

## Solutions & Current Best Practices
**Understanding:** pretrained encoder (Whisper/ViT) + connector + LLM, with **long context** for video and **frame sampling/compression** to manage tokens. **Generation:** discrete **neural codec** tokens (EnCodec/SoundStream) with autoregressive (or diffusion) modeling. **Real-time voice:** native, streaming, full-duplex speech models (avoid the ASR→LLM→TTS cascade). **Native any-to-any** pretraining for unified models (GPT-4o, Gemini). Use re-captioning and synthetic data to improve scarce paired data. Evaluate on Video-MME, EgoSchema (video), and ASR/audio-understanding suites.

## Lab Perspectives
- **OpenAI:** Whisper (open ASR), **GPT-4o** native voice (low-latency, emotional), Sora (video generation as world model).
- **Google DeepMind:** **Gemini** native audio+video understanding with **long context** (hour-long video); AudioPaLM, Lyria (music), Veo (video gen).
- **Meta:** ImageBind, SeamlessM4T (speech translation), Audiobox, MovieGen (video).
- **Kyutai/open:** **Moshi** (real-time full-duplex speech), open audio codecs.
- **Anthropic:** primarily text+vision; less public audio/video.

## Latest Developments (2023–2026)
**Real-time native voice** (GPT-4o, Gemini Live, Moshi) replaced cascaded pipelines, enabling natural spoken interaction with prosody/interruptions. **Long-context video** understanding (Gemini) handles hours of footage. **Unified any-to-any** models generate across modalities. **Video generation** (Sora, Veo, Genie) advanced rapidly, blurring into world models. Active work on **efficient video tokenization**, **temporal reasoning**, and **streaming/duplex** architectures.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why was Whisper significant?"** Scale + weakly-supervised web audio (680K hrs) → robust multilingual ASR, beating curated-data models.
- **"How do you feed audio/video into an LLM?"** Encoder features via a connector (understanding) or discrete codec tokens (generation); long context / frame sampling for video.
- **"Why is native voice better than ASR→LLM→TTS?"** Lower latency, preserves prosody/emotion/timing, enables full-duplex interruption handling.
- **"What's the main bottleneck for video understanding?"** Token explosion over time → long context + compression + temporal modeling.
- **"How does audio generation work autoregressively?"** Neural codec (RVQ) discretizes waveform into tokens an LM predicts.

## Open Problems
Efficient, high-fidelity **video/audio tokenization**, robust **temporal/causal reasoning** over long sequences, and truly **unified any-to-any** models that match specialists are open. **Real-time, full-duplex** multimodal interaction at scale, evaluation for long-video and generative audio, and the link to **embodied/world models** (predicting dynamics) remain active frontiers.

## References
- Radford, A. et al. (2022). *Whisper.* arXiv:2212.04356.
- Rubenstein, P. et al. (2023). *AudioPaLM.* arXiv:2306.12925.
- Défossez, A. et al. (2024). *Moshi.* arXiv:2410.00037.
- Zhang, H. et al. (2023). *Video-LLaMA.* arXiv:2306.02858.
- Défossez, A. et al. (2022). *EnCodec.* arXiv:2210.13438.
