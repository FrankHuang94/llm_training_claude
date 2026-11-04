# Instruction Tuning

> **Last Updated:** 2026-11-04
> **Related Files:** [Supervised Fine-Tuning](01_supervised_finetuning_SFT.md) · [Synthetic Data Generation](../01_data/04_synthetic_data_generation.md) · [Post-Training Overview](00_posttraining_overview.md)
> **Key Papers:** Wei et al. 2021 FLAN ([arXiv:2109.01652](https://arxiv.org/abs/2109.01652)) · Sanh et al. 2021 T0 ([arXiv:2110.08207](https://arxiv.org/abs/2110.08207)) · Chung et al. 2022 Scaling Instruction-Finetuning ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416)) · Wang et al. 2022 Super-NaturalInstructions ([arXiv:2204.07705](https://arxiv.org/abs/2204.07705))

## Overview
Instruction tuning is a specific, influential form of SFT: fine-tuning on a *large, diverse collection of NLP tasks reformatted as natural-language instructions*, with the explicit goal of **zero-shot generalization to unseen tasks**. The landmark result (FLAN, Wei et al. 2021) is that instruction-tuning a model on many tasks dramatically improves its ability to follow instructions for *tasks it never saw during tuning* — instruction-following is a meta-skill that transfers. This is distinct from chat-style SFT (which targets conversational behavior); instruction tuning targets *task generalization*. The two have since merged in practice, but the conceptual distinction and the FLAN line of work are frequent interview material. By 2026, instruction-formatted data has migrated *upstream* into midtraining and even pretraining (structured rephrasing), and *downstream* into RL (instance-specific rubrics as rewards).

## Core Concepts
**Task-as-instruction reformatting.** Take existing supervised datasets (sentiment, NLI, QA, summarization, translation) and wrap each in natural-language **instruction templates** ("Classify the sentiment of this review: ..."). A model trained on hundreds-to-thousands of such tasks learns the abstraction "read instruction → perform task."

**Held-out task evaluation.** The key methodology: cluster tasks, hold out entire *clusters*, instruction-tune on the rest, and evaluate **zero-shot** on the held-out clusters. Improvement there demonstrates genuine generalization, not memorization.

**Scaling findings (Chung et al. 2022).** Three axes improve instruction tuning: **(1) more tasks** (scaling to 1.8k tasks), **(2) larger models**, and **(3) including chain-of-thought** data (CoT instruction tuning unlocks reasoning generalization). Flan-PaLM/Flan-T5 showed large zero-shot and CoT gains; crucially, instruction tuning + CoT data preserved/improved reasoning rather than harming it.

**Datasets.** FLAN collection, T0's P3 (Public Pool of Prompts), Super-NaturalInstructions (1.6k tasks, community-authored), OpenAssistant (human conversations), Dolly, ShareGPT (real ChatGPT logs), and synthetic instruction sets (Self-Instruct, Evol-Instruct). Modern mixes blend task-instruction, conversational, reasoning, agentic/tool-use, and safety data.

**Instruction hierarchy and persistent instructions.** Modern assistants must follow instructions from several principals with explicit precedence (system/developer > user > tool output), and agents follow long-lived, file-based instructions (AGENTS.md, SKILL.md folders). Instruction-following is therefore trained *with priorities*, not as a flat skill.

## Key Challenges
- **Task diversity vs format diversity.** Generalization needs diverse *tasks* and diverse *phrasings*; narrow templates cause overfitting to template surface form.
- **Negative transfer / capability regression.** Naive instruction tuning can hurt open-ended generation or reasoning if the mix is skewed (early FLAN hurt some few-shot abilities; CoT data fixed reasoning).
- **Annotation cost & noise.** Human task collections are expensive; reformatting introduces template artifacts.
- **Eval leakage.** With thousands of tasks, held-out integrity is hard to guarantee (see [Data Contamination](../01_data/05_data_contamination_and_evaluation_leakage.md)).
- **Conflicting instructions.** Injected instructions in tool outputs or documents must be *ignored*; naive instruction-following is an attack surface (prompt injection).

## Solutions & Current Best Practices
**Mix many tasks + many templates + CoT + conversational + safety data**, balanced to avoid skew. Include **reasoning (CoT) exemplars** to preserve/boost reasoning. Combine **academic task collections** (FLAN/SNI) with **real and synthetic conversational** data (ShareGPT/OpenAssistant/Evol-Instruct) — academic tasks give breadth, conversational data gives natural assistant behavior. Deduplicate and decontaminate. Instruction tuning is now typically folded into the broader SFT stage before preference optimization, with instruction-like structured data introduced earlier (midtraining) and instruction adherence reinforced later with **rubric-based rewards**.

## Lab Perspectives
- **Google** originated and scaled instruction tuning (FLAN, T0 collaboration, Flan-T5/PaLM) — the foundational line.
- **OpenAI** framed it as part of InstructGPT's demonstration data and chat alignment; the Model Spec and AGENTS.md formalize instruction precedence and repository-level instructions.
- **Anthropic** places helpfulness (including instruction-following) *last* in an explicit priority order in its 2026 constitution (broadly safe > broadly ethical > guideline-compliant > genuinely helpful).
- **AllenAI/community** drove open instruction datasets (Tülu, SNI, Open-Instruct) and systematic mixture studies; OLMo 3 (2025-11) released fully open Instruct and Think variants with data and checkpoints.
- **Meta/Mistral/DeepSeek/Qwen** blend instruction, conversational, and synthetic reasoning data in their SFT stages.

## Latest Developments (2023–Nov 2026)
The frontier shifted from human task collections to **synthetic, evolved, and reasoning-rich** instruction data (Evol-Instruct, Orca, long-CoT distillation). **Tülu** (AllenAI) provided open, rigorous studies of instruction-tuning mixtures and their interaction with DPO. The line between "instruction tuning" and "chat SFT" blurred; the durable insight — **instruction-following generalizes across tasks and benefits from CoT** — underpins all modern assistants.

**Mid-2025 → Nov 2026 update:**
- **Structured rephrasing moves into pretraining (2026-04).** FinePhrase ([arXiv:2604.13977](https://arxiv.org/abs/2604.13977)) studied >1T generated tokens: rephrasing web data into structured targets (tables, math problems, FAQs, tutorials) beats curated web data and prior synthetic methods; generators larger than ~1B add nothing; 486B tokens released.
- **RL-trained rephrasers (2026-05).** SynPro ([arXiv:2605.17849](https://arxiv.org/abs/2605.17849)) rewards rephrasers for quality, faithfulness, and data influence, reporting 3.7–5.2× the effective tokens of plain repetition in a 10%-of-Chinchilla data regime — the "data wall" is partly an instruction-reformatting problem.
- **Midtraining timing (2025-10).** Introducing specialized/instruction-like data early tolerates a high mixture weight, late requires a low one ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865)).
- **Instructions become rewards (2025-07).** Rubrics as Rewards ([arXiv:2507.17746](https://arxiv.org/abs/2507.17746)) converts each instruction into an instance-specific checklist scored by an LLM judge: up to +31% relative on HealthBench over Likert-style judging.
- **Repository-level instructions standardized (2025-12 → 2026-10).** AGENTS.md (OpenAI) and Agent Skills (SKILL.md folders with progressive disclosure) joined the Linux Foundation's Agentic AI Foundation, making "follow the project's instruction files" a trained behavior for coding agents (see [Agentic Coding & Computer Use](../04_reasoning_and_agents/06_agentic_coding_and_computer_use.md)).
- **Recovering instruction-following after specialization.** On-policy distillation restored instruction-following lost during domain midtraining (Thinking Machines, 2025-10).

## Interview Angles
> 💡 **What labs actually ask:**
- **"What did FLAN show and why was it important?"** Multi-task instruction tuning → zero-shot generalization to unseen tasks; instruction-following is a transferable meta-skill.
- **"How do you evaluate instruction-tuning generalization?"** Held-out *task clusters*, zero-shot.
- **"What three things scale instruction tuning?"** More tasks, bigger models, CoT data (Chung et al.).
- **"Instruction tuning vs RLHF — what does each provide?"** IT gives task-following breadth via imitation; RLHF optimizes preference beyond demonstrations.
- **"Is instruction tuning still a separate stage in 2026?"** Mostly no: instruction-formatted data appears in pretraining (structured rephrasing), midtraining, SFT, and RL (rubric rewards); the remaining distinct problem is instruction *hierarchy* under injection.

## Open Problems
Why instruction-following generalizes so well (the mechanism of cross-task transfer) is not fully understood. Optimal task/format/CoT mixture composition, avoiding negative transfer to open-ended abilities, and whether massive instruction tuning is still needed given strong base models and RL are open questions. Robustly distinguishing legitimate instructions from injected ones in long agentic contexts is unsolved.

## References
- Wei, J. et al. (2021). *Finetuned LMs Are Zero-Shot Learners (FLAN).* arXiv:2109.01652.
- Sanh, V. et al. (2021). *Multitask Prompted Training (T0).* arXiv:2110.08207.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned LMs.* arXiv:2210.11416.
- Wang, Y. et al. (2022). *Super-NaturalInstructions.* arXiv:2204.07705.
- Ivison, H. et al. (2023). *Tülu / Camels in a Changing Climate.* arXiv:2311.10702.
- Scale AI (2025). *Rubrics as Rewards.* arXiv:2507.17746.
- FinePhrase authors (2026). *FinePhrase.* arXiv:2604.13977.
