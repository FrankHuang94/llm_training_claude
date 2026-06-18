# Lab Comparison Matrix

> **Last Updated:** 2026-06-18
> **Related Files:** [OpenAI](00_openai_roadmap.md) · [Anthropic](01_anthropic_roadmap.md) · [Google DeepMind](02_google_deepmind_roadmap.md) · [Meta](03_meta_ai_roadmap.md) · [DeepSeek](04_deepseek_roadmap.md) · [Mistral & Others](05_mistral_and_others.md)

This page distills how the major frontier labs differ across the dimensions that matter for both **technical understanding** and **interview tailoring**. Use it to quickly contrast approaches and to frame "why this lab?" answers.

---

## Master Comparison Table

| Dimension | **OpenAI** | **Anthropic** | **Google DeepMind** | **Meta AI** | **DeepSeek** | **Mistral** |
|---|---|---|---|---|---|---|
| **Flagship models** | GPT-4o, o3/o4 (reasoning), GPT-5-era | Claude 3.x / 4 (Opus/Sonnet/Haiku) | Gemini 2.5 (Ultra/Pro/Flash), Gemma (open) | Llama 3/4 (open) | V3 (MoE), R1 (reasoning) — open | Mistral Large, Mixtral, Magistral (open) |
| **Architecture choices** | Dense + MoE (GPT-4 likely MoE); native multimodal (4o) | Dense transformer (details closed); long-context | **Native multimodal**, MoE, long-context (1–2M) | Dense (L1–3) → **MoE** (L4); RMSNorm/RoPE/SwiGLU standard | **MLA** + fine-grained **MoE** + MTP | **SWA** + GQA + **MoE** (Mixtral); efficiency-first |
| **Training data scale** | Undisclosed (very large, licensed + web) | Undisclosed; heavy curation + CAI data | Huge multimodal (text/image/audio/video) | ~15T tokens (Llama 3), open recipe | ~14.8T tokens (V3); code/math-heavy | Undisclosed; multilingual/code focus |
| **Post-training approach** | **RLHF/PPO** pioneers + large **RLVR** (o-series) | **Constitutional AI / RLAIF** + RLHF | RLHF + **RLAIF** + verifiable reasoning RL | SFT + **rejection sampling** + RLHF/**DPO** | **GRPO + RLVR** (critic-free); R1-Zero pure RL | SFT + DPO/RL; reasoning (Magistral) |
| **Open vs closed** | **Closed** (API/product) | **Closed** (API/product) | **Mixed**: Gemini closed, **Gemma open** | **Open weights** (flagship strategy) | **Open weights** + detailed reports | **Open weights** (Apache) + API |
| **Safety methodology** | Preparedness, Model Spec, deliberative alignment, RBR | **Safety-first**: CAI, interpretability, scalable oversight, RSP/ASL | Frontier Safety Framework, dangerous-cap evals, interp (Gemma Scope) | Open safety tooling (Llama Guard/Purple Llama); openness-as-safety | Standard safety tuning; lighter public safety profile | Standard safety tuning; enterprise controls |
| **Key technical innovations** | RLHF, scaling laws, **inference-time scaling** (o1), native any-to-any (4o) | **Constitutional AI**, **SAEs/monosemanticity**, computer use, MCP | **Transformer**, **Chinchilla**, native multimodal, **1–2M context**, AlphaProof | **LLaMA** open recipe, overtraining paradigm, **PyTorch/FSDP**, BLT | **MLA**, **GRPO**, FP8 training, **DualPipe**, aux-loss-free MoE | **Sliding-window attention**, open MoE (Mixtral) |
| **Inference-time compute philosophy** | **Leader** — o-series scales test-time compute as a core axis | Extended **thinking** with controllable budget; faithfulness focus | "Thinking" (Flash-Thinking/2.5) + heavy **search** (AlphaProof/Code) | Catching up; reasoning gap in open Llama | **Open reasoning** via RLVR; distill to small models | Reasoning entrant (Magistral); efficiency-tuned |
| **Compute / hardware** | Microsoft **Azure** GPUs; Stargate-scale | **AWS Trainium + Google TPU** + GPUs (multi-accelerator) | Custom **TPU** (vertical integration) + JAX/Pathways | Very large **H100** fleets (~16k+ → 350k+) | **H800** (export-constrained) — efficiency-forced | EU **NVIDIA** clusters (smaller budget) |
| **Distribution / moat** | ChatGPT brand + product ecosystem | Enterprise/coding + safety credibility | Google product surface (Search/Android/Cloud) | Open ecosystem + Meta apps + PyTorch | Open + efficiency/cost narrative | EU sovereignty + enterprise/on-prem |
| **Primary weakness** | Closed/opaque, governance turbulence | Smaller consumer footprint, limited gen-media | Slower to productize, org complexity | Reasoning gap; openness-at-frontier tension | Hardware ceiling (export controls) | Compute budget; research breadth |

---

## Strategic Archetypes (one-line summaries)

- **OpenAI — "Frontier + product, reasoning-first."** Scale training *and* inference compute; ship fast; closed.
- **Anthropic — "Frontier as a means to safety."** Lead on alignment/interpretability while shipping top coding/agentic models.
- **Google DeepMind — "Breadth + vertical integration."** Native multimodal, longest context, custom TPUs, scientific reasoning.
- **Meta — "Commoditize via openness."** Set the open standard; win the ecosystem; integrate into products.
- **DeepSeek — "Efficiency beats compute."** Open, cheap, innovative architecture/systems under hardware constraints.
- **Mistral — "Open, efficient, European."** Sovereignty + efficiency as the differentiator.

## Axes of Divergence (what they actually disagree about)

1. **Open vs closed weights.** Meta/DeepSeek/Mistral (open) vs OpenAI/Anthropic (closed) vs Google (hybrid). Bets on ecosystem/safety-scrutiny vs control/monetization/misuse-prevention.
2. **Compute-maximalism vs efficiency.** OpenAI/xAI/Google (scale compute) vs DeepSeek/Mistral (algorithmic & systems efficiency). DeepSeek's "frontier on H800s" challenged the compute-moat thesis.
3. **Safety posture.** Anthropic (safety-first, catastrophic-risk-serious) ↔ OpenAI/Google (frameworks + capability) ↔ Meta (openness-as-safety, skeptical of x-risk framing) ↔ efficiency labs (lighter public safety research).
4. **Multimodality.** Native-from-pretraining (Google/OpenAI-4o) vs connector/late-fusion or text-first (Anthropic, much of open ecosystem).
5. **Reasoning route.** Trained internal long-CoT via RLVR (OpenAI o-series, DeepSeek R1, Google) vs catching up (Meta open). Neural-PRM vs rule-based verifiable rewards (DeepSeek argued rule-based).
6. **Post-training method.** PPO/RLHF (OpenAI) vs CAI/RLAIF (Anthropic) vs rejection-sampling+DPO (Meta) vs GRPO (DeepSeek).

## How to Use This for "Why This Lab?" Interview Answers

- **Anthropic:** emphasize alignment/interpretability interest, CAI, scalable oversight, careful reasoning.
- **OpenAI:** emphasize RL/post-training, reasoning, inference-time scaling, shipping at scale.
- **Google DeepMind:** emphasize TPU/JAX, multimodal, long context, RL/search, scientific reasoning.
- **Meta:** emphasize open research, PyTorch/systems, efficient architectures, scale.
- **DeepSeek/Mistral:** emphasize efficiency, systems (FP8/MoE/MLA, SWA), and open reproducibility.

## Caveats

This matrix is a **strategic abstraction** as of mid-2026. Model details for closed labs (OpenAI, Anthropic, and Gemini internals) are inferred from public reports and may be imprecise (e.g., GPT-4-as-MoE is widely believed but unconfirmed). Labs converge over time — most now do some MoE, some reasoning RL, and some long context — so treat the columns as *emphases and origins*, not rigid boundaries.
