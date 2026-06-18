# Instruction Tuning

> **Last Updated:** 2026-06-18
> **Related Files:** [Supervised Fine-Tuning](01_supervised_finetuning_SFT.md) · [Synthetic Data Generation](../01_data/04_synthetic_data_generation.md) · [Post-Training Overview](00_posttraining_overview.md)
> **Key Papers:** Wei et al. 2021 FLAN ([arXiv:2109.01652](https://arxiv.org/abs/2109.01652)) · Sanh et al. 2021 T0 ([arXiv:2110.08207](https://arxiv.org/abs/2110.08207)) · Chung et al. 2022 Scaling Instruction-Finetuning ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416)) · Wang et al. 2022 Super-NaturalInstructions ([arXiv:2204.07705](https://arxiv.org/abs/2204.07705))

## Overview
Instruction tuning is a specific, influential form of SFT: fine-tuning on a *large, diverse collection of NLP tasks reformatted as natural-language instructions*, with the explicit goal of **zero-shot generalization to unseen tasks**. The landmark result (FLAN, Wei et al. 2021) is that instruction-tuning a model on many tasks dramatically improves its ability to follow instructions for *tasks it never saw during tuning* — instruction-following is a meta-skill that transfers. This is distinct from chat-style SFT (which targets conversational behavior); instruction tuning targets *task generalization*. The two have since merged in practice, but the conceptual distinction and the FLAN line of work are frequent interview material.

## Core Concepts
**Task-as-instruction reformatting.** Take existing supervised datasets (sentiment, NLI, QA, summarization, translation) and wrap each in natural-language **instruction templates** ("Classify the sentiment of this review: ..."). A model trained on hundreds-to-thousands of such tasks learns the abstraction "read instruction → perform task."

**Held-out task evaluation.** The key methodology: cluster tasks, hold out entire *clusters*, instruction-tune on the rest, and evaluate **zero-shot** on the held-out clusters. Improvement there demonstrates genuine generalization, not memorization.

**Scaling findings (Chung et al. 2022).** Three axes improve instruction tuning: **(1) more tasks** (scaling to 1.8k tasks), **(2) larger models**, and **(3) including chain-of-thought** data (CoT instruction tuning unlocks reasoning generalization). Flan-PaLM/Flan-T5 showed large zero-shot and CoT gains; crucially, instruction tuning + CoT data preserved/improved reasoning rather than harming it.

**Datasets.** FLAN collection, T0's P3 (Public Pool of Prompts), Super-NaturalInstructions (1.6k tasks, community-authored), OpenAssistant (human conversations), Dolly, ShareGPT (real ChatGPT logs), and synthetic instruction sets (Self-Instruct, Evol-Instruct). Modern mixes blend task-instruction, conversational, reasoning, and safety data.

## Key Challenges
- **Task diversity vs format diversity.** Generalization needs diverse *tasks* and diverse *phrasings*; narrow templates cause overfitting to template surface form.
- **Negative transfer / capability regression.** Naive instruction tuning can hurt open-ended generation or reasoning if the mix is skewed (early FLAN hurt some few-shot abilities; CoT data fixed reasoning).
- **Annotation cost & noise.** Human task collections are expensive; reformatting introduces template artifacts.
- **Eval leakage.** With thousands of tasks, held-out integrity is hard to guarantee (see [Data Contamination](../01_data/05_data_contamination_and_evaluation_leakage.md)).

## Solutions & Current Best Practices
**Mix many tasks + many templates + CoT + conversational + safety data**, balanced to avoid skew. Include **reasoning (CoT) exemplars** to preserve/boost reasoning. Combine **academic task collections** (FLAN/SNI) with **real and synthetic conversational** data (ShareGPT/OpenAssistant/Evol-Instruct) — academic tasks give breadth, conversational data gives natural assistant behavior. Deduplicate and decontaminate. Instruction tuning is now typically folded into the broader SFT stage before preference optimization.

## Lab Perspectives
- **Google** originated and scaled instruction tuning (FLAN, T0 collaboration, Flan-T5/PaLM) — the foundational line.
- **OpenAI** framed it as part of InstructGPT's demonstration data and chat alignment.
- **AllenAI/community** drove open instruction datasets (Tülu, SNI, Open-Instruct) and systematic mixture studies.
- **Meta/Mistral/DeepSeek** blend instruction, conversational, and synthetic reasoning data in their SFT stages.

## Latest Developments (2023–2026)
The frontier shifted from human task collections to **synthetic, evolved, and reasoning-rich** instruction data (Evol-Instruct, Orca, long-CoT distillation). **Tülu** (AllenAI) provided open, rigorous studies of instruction-tuning mixtures and their interaction with DPO. The line between "instruction tuning" and "chat SFT" blurred; the durable insight — **instruction-following generalizes across tasks and benefits from CoT** — underpins all modern assistants.

## Interview Angles
> 💡 **What labs actually ask:**
- **"What did FLAN show and why was it important?"** Multi-task instruction tuning → zero-shot generalization to unseen tasks; instruction-following is a transferable meta-skill.
- **"How do you evaluate instruction-tuning generalization?"** Held-out *task clusters*, zero-shot.
- **"What three things scale instruction tuning?"** More tasks, bigger models, CoT data (Chung et al.).
- **"Instruction tuning vs RLHF — what does each provide?"** IT gives task-following breadth via imitation; RLHF optimizes preference beyond demonstrations.

## Open Problems
Why instruction-following generalizes so well (the mechanism of cross-task transfer) is not fully understood. Optimal task/format/CoT mixture composition, avoiding negative transfer to open-ended abilities, and whether massive instruction tuning is still needed given strong base models and RL are open questions.

## References
- Wei, J. et al. (2021). *Finetuned LMs Are Zero-Shot Learners (FLAN).* arXiv:2109.01652.
- Sanh, V. et al. (2021). *Multitask Prompted Training (T0).* arXiv:2110.08207.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned LMs.* arXiv:2210.11416.
- Wang, Y. et al. (2022). *Super-NaturalInstructions.* arXiv:2204.07705.
- Ivison, H. et al. (2023). *Tülu / Camels in a Changing Climate.* arXiv:2311.10702.
