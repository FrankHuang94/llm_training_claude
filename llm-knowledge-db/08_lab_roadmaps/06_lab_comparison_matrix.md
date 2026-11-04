# Lab Comparison Matrix

> **Last Updated:** 2026-11-04
> **Related Files:** [OpenAI](00_openai_roadmap.md) · [Anthropic](01_anthropic_roadmap.md) · [Google DeepMind](02_google_deepmind_roadmap.md) · [Meta](03_meta_ai_roadmap.md) · [DeepSeek](04_deepseek_roadmap.md) · [Mistral & Others](05_mistral_and_others.md)

This page distills how the major frontier labs differ across the dimensions that matter for both **technical understanding** and **interview tailoring**. Use it to quickly contrast approaches and to frame "why this lab?" answers. As of Nov 2026 the picture has shifted substantially since early 2025: Meta has closed its frontier weights, the open-weight frontier is led by Chinese labs, all three US closed labs gate their most dangerous capabilities behind tiered access, and compute is contracted in gigawatts across multiple chip vendors.

---

## Master Comparison Table

| Dimension | **OpenAI** | **Anthropic** | **Google DeepMind** | **Meta AI** | **DeepSeek** | **Chinese open-weight labs (Qwen / Kimi / GLM)** | **Mistral** |
|---|---|---|---|---|---|---|---|
| **Flagship models (Nov 2026)** | GPT-6 Astra; GPT-6 / 6.1 Sol, Luna; gpt-oss (open) | Opus / Sonnet / Haiku 5.5; Fable 5.1 / Mythos 5.1 (Mythos-class) | Gemini 3.1 Pro, 3.5 / 3.6 Flash; Gemini 4 in post-training; Gemma 4 (open) | Muse Spark, Muse Wonder / Full (closed); Llama 4 (legacy open) | V4.1-Flash (552B, vision); V4-Pro (1.6T, phasing out) | Qwen3.8-2.4T-A95B; Kimi K3 (2.8T) / K3.1 Pro; GLM-5 / 5.1 (744B) | Large 4 (~1.05T, open); Medium 3.5 (128B dense) |
| **Architecture choices** | Closed details; gpt-oss: top-4 MoE, sliding/dense + attention sinks, MXFP4 | Closed details; 1M context (Opus 4.6+) | **Native multimodal**, MoE, long context; Gemma 4 MoE + PLE edge models | Llama: RMSNorm/RoPE/SwiGLU → MoE (L4); Muse undisclosed | **MLA → DSA → CSA/HCA** + fine-grained MoE + MTP; mHC | Linear hybrids (**Gated DeltaNet**, **KDA** ~3:1) or **DSA**; 512–896 experts | Granular MoE (Large 3/4); dense 128B; SWA heritage |
| **Training data scale** | Undisclosed; Astra framed as largest pretraining run | Undisclosed; constitution-driven synthetic data | Huge multimodal (text/image/audio/video) | ~15T (Llama 3); Muse undisclosed | ~14.8T (V3); ~32T reported (V4) | 36T (Qwen3-Max); 15.5T (K2); 28.5T (GLM-5) | Undisclosed |
| **Post-training approach** | RLHF pioneers → large-scale **RL for long-horizon agents**; routed thinking (GPT-5) | **CAI / RLAIF** + RL for agents; inoculation prompting; adaptive / always-on thinking | RLHF + RLAIF + reasoning RL; Deep Think parallel reasoning | SFT + rejection sampling + RLHF/DPO (Llama); Muse undisclosed | **GRPO + RLVR**; hybrid think / non-think; verifier-as-reward (Math-V2) | GRPO-family RL; async agent RL (GLM slime); INT4 QAT in post-training (K2 Thinking) | Mode-merged instruct + reasoning + code checkpoints |
| **Open vs closed** | **Closed** frontier; **gpt-oss open** (Apache 2.0) | **Closed** | **Mixed**: Gemini closed, **Gemma 4 Apache 2.0** | **Closed frontier since Apr 2026** (Muse); legacy Llama open | **Open** (MIT) + detailed reports | **Open**, "MIT, but" custom licenses; some flagships hosted-only (Qwen3-Max) | **Open** (Apache 2.0 / modified MIT, MaaS carve-out) + API |
| **Restricted / tiered release** | **Daybreak** Blue / Red; GPT-5.6-Cyber; Astra "Critical" cyber | **Fable vs Mythos** (same weights); Glasswing; Cyber & Life Sciences Verification Programs; classifier routing | **3.5 Flash Cyber** for governments / trusted partners | Private-preview API for partners | None (open weights) | Capabilities removed from open checkpoints (K3.1 cyber data); hosted-first | None reported |
| **Safety methodology** | Preparedness Framework v2 (High / Critical), CoT-monitorability evals, anti-scheming, confessions | **RSP v3** (Risk Reports, public roadmap), CAI, interpretability, CC++ classifiers, capability shaping in training | **FSF v3** (manipulation CCL, shutdown resistance; reported v3.1 tracked levels), restricted cyber model | Llama Guard / Purple Llama; declined EU GPAI Code | Standard safety tuning; light public safety profile | Standard safety tuning; data removal in open releases | Standard safety tuning; enterprise controls |
| **Key technical innovations** | RLHF, scaling laws, **inference-time scaling**, model **router**, native computer use (5.4) | **Constitutional AI**, SAEs / attribution graphs, MCP, effort control, same-weights tiering | **Transformer**, **Chinchilla**, native multimodal, Genie 3 world model, TPU 8t / 8i | **LLaMA** open recipe, overtraining, **PyTorch / FSDP**, BLT | **MLA**, **GRPO**, FP8, DualPipe, **DSA**, CSA/HCA, UE8M0 | **MuonClip**, KDA, AttnRes (Kimi); Qwen3-Next hybrid; slime (GLM) | SWA, open MoE (Mixtral), first Western ~1T open MoE |
| **Inference-time compute philosophy** | Routed fast vs thinking; tokens-per-task as a metric | **Effort levels** (low → max); thinking always on (Opus 5.5) | Thinking + Deep Think; Flash-tier efficiency | Undisclosed (Muse) | Hybrid think / non-think in one checkpoint | Thinking modes; Qwen3.8 open weights thinking-always-on | Reasoning merged into general checkpoints |
| **Compute / hardware** | ~26 GW across NVIDIA, AMD (MI450) and Broadcom custom; Stargate; Cerebras for low latency | **TPU** (up to 1M chips) + **Trainium** (Rainier) + NVIDIA / Azure + AMD (2 GW), each ≥1 GW scale | Own **TPU** (Ironwood GA; 8t / 8i) + JAX; sells TPUs externally | NVIDIA + up to **6 GW AMD**; gigawatt campuses | H800 legacy; domestic-chip formats; V4 hardware disputed | Largely undisclosed; GLM-5 reportedly all-Ascend; Qwen3.8 served on GB300 NVL72 | EU NVIDIA clusters (smaller budget) |
| **Distribution / moat** | ChatGPT brand + Codex + API | Enterprise / agentic coding (Claude Code) + safety credibility | Google surface (Search AI Mode / Android / Cloud) + TPU | Meta apps + devices + PyTorch | Open + efficiency / cost narrative | Open-ecosystem ubiquity; price | EU sovereignty + enterprise / on-prem |
| **Primary weakness** | Closed; governance and capital intensity | Deployment friction; eval incidents; small consumer footprint | Uneven shipping (3.5 Pro delay); reorg | Lost open leadership; reorgs; unproven closed frontier | Hardware ceiling; slower 2026 cadence | Export controls; licensing fragmentation; trust in the West | Compute budget; trails Chinese open models |

---

## Strategic Archetypes (one-line summaries)

- **OpenAI — "Frontier + product, scale everything."** Re-scaled pretraining *and* long-horizon RL, gigawatt multi-vendor compute, fast tiered releases, trusted access for Critical capabilities.
- **Anthropic — "Frontier as a means to safety, shipped in tiers."** Lead on alignment/interpretability and agentic coding; same weights under different safeguard levels.
- **Google DeepMind — "Breadth + vertical integration."** Native multimodal, TPUs (now also a product), Flash-tier efficiency, world models and robotics.
- **Meta — "From commoditize-via-openness to closed superintelligence."** Muse marks a pivot to closed frontier models.
- **DeepSeek — "Efficiency beats compute."** Open, MIT-licensed, attention-efficiency innovation under hardware constraints.
- **Chinese open-weight labs — "Open frontier at trillion scale."** Fast cadence, extreme sparsity, sub-quadratic attention, 4-bit training, custom licenses.
- **Mistral — "Open, efficient, European."** Sovereignty + openness, now at ~1T scale.

## Axes of Divergence (what they actually disagree about)

1. **Open vs closed weights.** Open frontier: DeepSeek, Qwen/Kimi/GLM/MiniMax, Mistral; selective open: OpenAI (gpt-oss), Google (Gemma); closed: Anthropic, and since 2026 **Meta**. Bets on ecosystem/safety-scrutiny vs control, monetization, misuse and distillation prevention.
2. **Who gets the most capable model.** General availability vs **restricted tiers** (Mythos, Daybreak Red, Flash Cyber) vs open weights with capabilities removed — deployment is now base model + classifiers + routing + identity verification.
3. **Compute-maximalism vs efficiency.** OpenAI/Anthropic/Google/Meta contract **multi-gigawatt, multi-vendor** compute; DeepSeek and Chinese labs push sparsity, compressed attention and 4-bit formats, increasingly designed for domestic chips.
4. **Safety posture.** Anthropic (RSP v3, capability shaping, incident disclosure) ↔ OpenAI/Google (Preparedness / FSF thresholds + restricted release) ↔ Meta (declined EU Code; less safety publication since the closed pivot) ↔ open-weight labs (lighter public safety research, data removal).
5. **Attention for long context.** Dense + undisclosed tricks (US closed) vs **learned sparse** (DSA/CSA, MSA) vs **linear hybrids** (Gated DeltaNet, KDA) vs **full attention** (MiniMax M2's argument).
6. **Reasoning exposure.** Visible or summarized CoT vs **preserved/account-bound thinking** and distillation classifiers (Anthropic), router-hidden depth (OpenAI) vs fully open traces (open models).
7. **Post-training method.** RL for long-horizon agents everywhere; differing emphases: CAI/constitution (Anthropic), routed reasoning (OpenAI), GRPO + verifiers (DeepSeek), async agent RL (GLM).

## Latest Developments (2023–Nov 2026)
Through mid-2025 the matrix was organized around open (Meta, DeepSeek, Mistral) vs closed (OpenAI, Anthropic) vs hybrid (Google), and around who had reasoning models. Those boundaries have since moved.

**Mid-2025 → Nov 2026 update:**
- **2025-08 → 2026-04 — Open/closed realignment:** OpenAI released gpt-oss (Apache 2.0), Google moved Gemma 4 to Apache 2.0, and Meta closed its frontier with Muse Spark — leaving the open frontier to DeepSeek, Qwen, Kimi, GLM and Mistral.
- **2026-04 → 09 — Tiered release becomes standard:** Mythos Preview/Glasswing, Fable vs Mythos 5, Gemini 3.5 Flash Cyber, Daybreak and GPT-6 Astra's "Critical" cyber rating; capability is also shaped *in training* (Opus 4.7/5).
- **2025-09 → 2026-07 — Gigawatt multi-vendor compute:** OpenAI ~26 GW (NVIDIA/AMD/Broadcom), Anthropic across TPU/Trainium/NVIDIA/AMD, Meta 6 GW AMD; TPUs sold externally; chips split into training vs inference parts (TPU 8t/8i, Rubin CPX).
- **2025-07 → 2026-10 — Open MoEs at 1–2.8T:** sub-quadratic attention, 1M context and 4-bit training became default in Chinese flagships; Mistral Large 4 is the first Western ~1T open model.
- **Cadence and price:** ~4–8 weeks between releases per lab; tiered families (Sol/Terra/Luna, Opus/Sonnet/Haiku, Pro/Flash/Lite) with 40–90% price drops at constant capability.
- **Governance:** US export controls briefly pulled Fable/Mythos 5 (2026-06); EU AI Act GPAI duties apply (Meta declined the Code of Practice); safety frameworks shifted from thresholds to periodic reporting (RSP v3 Risk Reports, FSF tracked levels).

## How to Use This for "Why This Lab?" Interview Answers

- **Anthropic:** alignment/interpretability, CAI and the constitution, automated alignment research, safeguards engineering (classifiers, tiered release), long-horizon agentic coding.
- **OpenAI:** RL/post-training at scale, reasoning and routing, computer-use agents, monitorability, shipping at massive scale.
- **Google DeepMind:** TPU/JAX, multimodal, world models and robotics, RL/search, scientific reasoning.
- **Meta:** large-scale systems/PyTorch, multimodal, agent memory, building a closed frontier line from strong infrastructure.
- **DeepSeek / Chinese open-weight labs / Mistral:** efficiency (sparse and linear attention, MoE, FP8/FP4, Muon), open reproducibility, serving trillion-parameter MoEs.

## Caveats

This matrix is a **strategic abstraction** as of Nov 2026. Model details for closed labs (OpenAI, Anthropic, Gemini and Muse internals) are inferred from public reports and system cards and may be imprecise; many 2026 benchmark and pricing figures are vendor-reported, and some items (e.g., GLM-5's all-Ascend training, DeepSeek V4's hardware, Kimi K3.1 open weights) are reported but unconfirmed. Labs converge over time — nearly all now do MoE, long-horizon RL, 1M context and tiered model families — so treat the columns as *emphases and origins*, not rigid boundaries.
