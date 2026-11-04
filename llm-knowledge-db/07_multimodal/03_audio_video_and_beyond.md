# Audio, Video, and Beyond

> **Last Updated:** 2026-11-04
> **Related Files:** [Multimodal Pretraining](01_multimodal_pretraining.md) · [Vision-Language Models](00_vision_language_models.md) · [World Models](../09_emerging_frontiers/02_world_models.md) · [Long Context & Memory](../09_emerging_frontiers/00_long_context_and_memory.md)
> **Key Papers:** Radford et al. 2022 Whisper ([arXiv:2212.04356](https://arxiv.org/abs/2212.04356)) · Rubenstein et al. 2023 AudioPaLM ([arXiv:2306.12925](https://arxiv.org/abs/2306.12925)) · Défossez et al. 2024 Moshi ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037)) · Zhang et al. 2023 Video-LLaMA ([arXiv:2306.02858](https://arxiv.org/abs/2306.02858)) · Wiedemer et al. 2025 Video Models Are Zero-Shot Learners and Reasoners ([arXiv:2509.20328](https://arxiv.org/abs/2509.20328))

## Overview
Beyond images, frontier models increasingly handle **audio** (speech and general sound), **video** (the temporal visual stream), and other modalities (3D, depth, sensor data). These modalities push two frontiers: **scale** (video and audio produce enormous token counts) and **real-time interaction** (low-latency speech for natural voice assistants). The trend mirrors vision: pretrained encoders + LLM connectors for *understanding*, and discrete neural codecs for *generation*, converging toward **native any-to-any** models (GPT-4o voice, Gemini video, Qwen3-Omni). Audio and video also connect to the **world-model** question — modeling temporal dynamics (see [World Models](../09_emerging_frontiers/02_world_models.md)); in 2025–26 video generators began to behave like zero-shot visual reasoners and interactive world simulators (Veo 3, Genie 3), and joint audio-video generation became standard (Veo 3, Sora 2).

## Core Concepts
**Audio understanding.** **Whisper** (OpenAI) is the canonical speech model: an encoder-decoder Transformer trained on 680K hours of weakly-supervised web audio for robust multilingual ASR and translation, demonstrating that **scale + weak supervision** beats curated speech datasets. For LLM integration, an audio encoder (Whisper/Conformer) feeds features into the LLM via a connector (audio analogue of LLaVA).

**Audio tokenization & generation.** Neural audio codecs (**SoundStream**, **EnCodec**) compress waveforms into **discrete tokens** via residual vector quantization (RVQ), letting an LLM generate audio autoregressively. With $Q$ codebooks of size $K$ at frame rate $f$, the bitrate is $f\,Q\log_2 K$ bits/s — the fidelity-vs-sequence-length knob. **AudioLM/AudioPaLM/VALL-E** generate speech as token sequences; this enables a unified text+speech model. **Moshi** (Kyutai) is a full-duplex, real-time speech-to-speech model achieving very low latency by modeling audio and text streams jointly.

**Video understanding.** Video = a sequence of frames + audio + time. Approaches sample frames, encode each with a ViT, and feed the (large) token sequence to the LLM, using **temporal modeling** (temporal attention, Q-Formers, or simply long context). The core problem is **token explosion**: a few minutes of video at reasonable frame rates is hundreds of thousands of tokens. **Long context** (Gemini 1.5/2.5 at 1M+ tokens; by 2026 1M is the default spec for many open models) is a key enabler for hour-long video; token compression and keyframe selection help.

**Real-time/streaming.** Voice assistants need **low latency** and **full-duplex** (listen while speaking, handle interruptions). Native audio models (GPT-4o voice, Moshi, Gemini Live) generate speech directly rather than the slow ASR→LLM→TTS cascade, capturing prosody, emotion, and timing. **Thinker–Talker** designs (Qwen3-Omni) split a reasoning model from a streaming speech generator that conditions on its hidden states.

**Video generation as world modeling.** Large video generators trained on internet video learn physics and object permanence implicitly; "**chain-of-frames**" — solving a visual task by generating the frames that carry out the solution — is the video analogue of chain-of-thought. Interactive world models (Genie 3) add action/prompt conditioning in real time.

**Beyond.** **ImageBind** binds 6 modalities (image, text, audio, depth, thermal, IMU); robotics/embodied models add proprioception and action (VLAs such as Gemini Robotics 2; latent planners such as V-JEPA 2-AC); world/video-generation models (Sora, Veo, Genie) learn dynamics.

## Key Challenges
- **Token explosion (video/audio).** Temporal data overwhelms context; efficient tokenization/compression and long context are essential.
- **Temporal reasoning.** Understanding events, causality, and order over time (not just per-frame content) is hard.
- **Latency for real-time.** Full-duplex, sub-300ms speech interaction requires streaming architectures and tight engineering.
- **Codec quality vs bitrate.** Discrete audio tokens trade fidelity for sequence length; reconstruction quality bounds generation.
- **Long-horizon consistency.** Interactive world models stay consistent for only minutes; memory is emergent rather than an explicit 3D state.
- **Data & alignment.** Paired video/audio-text data is noisier and scarcer; temporal alignment (which words match which moment) is hard.
- **Evaluation.** Long-video and audio understanding/generation benchmarks are immature; physical plausibility of generated video lacks good metrics.

## Solutions & Current Best Practices
**Understanding:** pretrained encoder (Whisper/ViT) + connector + LLM, with **long context** for video and **frame sampling/compression** to manage tokens; increasingly, native multimodal pretraining with sparse/hybrid attention to make 1M-token contexts affordable. **Generation:** discrete **neural codec** tokens (EnCodec/SoundStream, multi-codebook) with autoregressive (or diffusion) modeling; **joint audio-video** generation for synchronized sound. **Real-time voice:** native, streaming, full-duplex speech models (avoid the ASR→LLM→TTS cascade), e.g., Thinker–Talker with codec streaming. **Native any-to-any** pretraining for unified models (GPT-4o, Gemini, Qwen3-Omni). Use re-captioning and synthetic data to improve scarce paired data; use self-supervised video pretraining for embodied perception. Evaluate on Video-MME, EgoSchema (video), SSv2 (motion understanding), and ASR/audio-understanding suites.

## Lab Perspectives
- **OpenAI:** Whisper (open ASR), **GPT-4o** native voice (low-latency, emotional), Sora → **Sora 2** (2025-09; joint video+audio with synchronized dialogue and SFX, better physical plausibility).
- **Google DeepMind:** **Gemini** native audio+video understanding with **long context** (hour-long video); AudioPaLM, Lyria (music), **Veo 3** (2025-05; first with native audio; zero-shot "chain-of-frames" reasoning), **Genie 3** (2025-08; real-time interactive 720p/24fps worlds, limited preview); Gemini Omni announced at I/O 2026; Gemini Robotics 2 VLA.
- **Meta:** ImageBind, SeamlessM4T (speech translation), Audiobox, MovieGen (video), **V-JEPA 2** (self-supervised video world model with zero-shot robot planning).
- **Qwen/Alibaba:** **Qwen3-Omni** (30B-A3B Thinker–Talker, streaming speech).
- **Kyutai/open:** **Moshi** (real-time full-duplex speech), open audio codecs; Xiaomi MiMo-V2.5 and Gemma 4 edge models with native audio input.
- **Anthropic:** primarily text+vision; less public audio/video.

## Latest Developments (2023–Nov 2026)
**Real-time native voice** (GPT-4o, Gemini Live, Moshi) replaced cascaded pipelines, enabling natural spoken interaction with prosody/interruptions. **Long-context video** understanding (Gemini) handles hours of footage. **Unified any-to-any** models generate across modalities. **Video generation** (Sora, Veo, Genie) advanced rapidly, blurring into world models. Active work on **efficient video tokenization**, **temporal reasoning**, and **streaming/duplex** architectures.

**Mid-2025 → Nov 2026 update:**
- **Joint audio-video generation (2025-05 → 2025-09).** Veo 3 introduced native audio; Sora 2 generates video and audio jointly with synchronized dialogue and sound effects and improved physical plausibility — sound is no longer a post-hoc dubbing stage.
- **Video models as zero-shot learners and reasoners (2025-09, arXiv:2509.20328).** Veo 3 solves perception, physics, manipulation and maze-style tasks zero-shot via chain-of-frames, improving sharply over Veo 2 — suggesting large video generators may become general vision foundation models the way LLMs did for text.
- **Interactive world models (2025-08).** Genie 3 generates navigable 720p worlds at 24 fps in real time, consistent for a few minutes, with "promptable world events"; memory is emergent with no explicit 3D representation. Pitched as a training ground for agents (research preview only).
- **Self-supervised video for control (2025-06, arXiv:2506.09985).** V-JEPA 2 pretrains on >1M hours of video (77.3% SSv2); V-JEPA 2-AC, post-trained on <62 h of unlabeled robot video, plans zero-shot pick-and-place via MPC in latent space.
- **Omni streaming speech (2025-09, arXiv:2509.17765).** Qwen3-Omni (30B-A3B MoE) uses a Talker conditioned on audio/visual features rather than Thinker text, streaming speech through a multi-codebook codec with ~234 ms first-packet latency, and reports no modality degradation versus single-modal Qwen models.
- **Native audio spreads to open and edge models (2026-04).** Xiaomi MiMo-V2.5 (MIT, ~310.8B/15B active, native vision+audio, 1M context) and Gemma 4 E2B/E4B (Apache 2.0, text/image/audio on-device).
- **1M-token contexts get cheap (2025-10 → 2026-04).** Kimi Linear's KDA hybrid (up to 75% less KV cache, 6× decode at 1M) and DeepSeek-V4's compressed sparse attention (~10% of V3.2's KV cache at 1M, per reviewers) make long audio/video token streams far more affordable to serve.
- **Embodied VLAs (2026-07).** Gemini Robotics 2 targets whole-body humanoid control, with the ER 2 planner splitting tasks across multiple robots (secondary reports).

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why was Whisper significant?"** Scale + weakly-supervised web audio (680K hrs) → robust multilingual ASR, beating curated-data models.
- **"How do you feed audio/video into an LLM?"** Encoder features via a connector (understanding) or discrete codec tokens (generation); long context / frame sampling for video.
- **"Why is native voice better than ASR→LLM→TTS?"** Lower latency, preserves prosody/emotion/timing, enables full-duplex interruption handling.
- **"What's the main bottleneck for video understanding?"** Token explosion over time → long context + compression + temporal modeling.
- **"How does audio generation work autoregressively?"** Neural codec (RVQ) discretizes waveform into tokens an LM predicts; bitrate $= f\,Q\log_2 K$ trades fidelity for sequence length.
- **"What does 'chain-of-frames' mean, and is a video generator a world model?"** The model solves a task by generating frames that enact the solution (analogous to CoT); it implies learned dynamics, but consistency is short-horizon and there is no explicit state or action interface unless added (Genie 3).
- **"Design a low-latency speech assistant on top of a reasoning LLM."** Thinker–Talker split, Talker conditioned on hidden states, multi-codebook streaming codec, first-packet latency budget (~200–300 ms), barge-in handling.

## Open Problems
Efficient, high-fidelity **video/audio tokenization**, robust **temporal/causal reasoning** over long sequences, and truly **unified any-to-any** models that match specialists are open. **Real-time, full-duplex** multimodal interaction at scale, evaluation for long-video and generative audio, long-horizon **consistency** of interactive world models, and the link to **embodied/world models** (predicting dynamics, planning with them) remain active frontiers.

## References
- Radford, A. et al. (2022). *Whisper.* arXiv:2212.04356.
- Rubenstein, P. et al. (2023). *AudioPaLM.* arXiv:2306.12925.
- Défossez, A. et al. (2024). *Moshi.* arXiv:2410.00037.
- Zhang, H. et al. (2023). *Video-LLaMA.* arXiv:2306.02858.
- Défossez, A. et al. (2022). *EnCodec.* arXiv:2210.13438.
- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985.
- Qwen Team (2025). *Qwen3-Omni Technical Report.* arXiv:2509.17765.
- Wiedemer, T. et al. (2025). *Video Models Are Zero-Shot Learners and Reasoners.* arXiv:2509.20328.
