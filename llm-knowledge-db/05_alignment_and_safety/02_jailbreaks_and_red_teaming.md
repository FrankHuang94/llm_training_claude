# Jailbreaks and Red-Teaming

> **Last Updated:** 2026-06-18
> **Related Files:** [Alignment Overview](00_alignment_overview.md) · [Constitutional AI & RLAIF](../03_posttraining/07_constitutional_AI_and_RLAIF.md) · [Tool Use & Function Calling](../04_reasoning_and_agents/03_tool_use_and_function_calling.md)
> **Key Papers:** Zou et al. 2023 Universal Adversarial Attacks (GCG) ([arXiv:2307.15043](https://arxiv.org/abs/2307.15043)) · Wei et al. 2023 "Jailbroken: How Does LLM Safety Training Fail?" ([arXiv:2307.02483](https://arxiv.org/abs/2307.02483)) · Perez et al. 2022 Red Teaming LMs with LMs ([arXiv:2202.03286](https://arxiv.org/abs/2202.03286)) · Greshake et al. 2023 Indirect Prompt Injection ([arXiv:2302.12173](https://arxiv.org/abs/2302.12173))

## Overview
**Jailbreaking** is eliciting behavior a model was trained to refuse (e.g., instructions for weapons, malware, disallowed content) by crafting adversarial inputs that bypass its safety training. **Red-teaming** is the proactive practice of finding such failures before adversaries (or accidents) do. As models gain dangerous capabilities and are wired into tools and autonomous systems, the security stakes rise sharply — and a new class of attacks (**prompt injection**) targets *applications* built on LLMs, not just the chat interface. This is a fast-moving adversarial domain and a core competency for safety/security roles.

## Core Concepts
**Why safety training fails (Wei et al. 2023).** Two failure modes: **competing objectives** (the model's helpfulness/instruction-following drive conflicts with its safety training, and clever prompts amplify the former) and **mismatched generalization** (safety training doesn't cover the input distribution the attack uses — e.g., rare encodings, languages, or formats the refusal behavior never generalized to).

**Jailbreak techniques.**
- **Persona/roleplay** ("DAN," "you are an unfiltered AI"), hypothetical/fiction framing.
- **Instruction obfuscation:** base64/ROT13/leetspeak, low-resource languages, token-splitting, payload in code/markdown.
- **Prefix injection / refusal suppression:** force the model to start with "Sure, here's..." or forbid refusal phrases.
- **Many-shot jailbreaking** (Anthropic 2024): fill a long context with many fake examples of the assistant complying with harmful requests — exploits in-context learning and long context windows.
- **Crescendo / multi-turn:** escalate gradually over a conversation so no single turn trips the filter.
- **Automated/optimized attacks (GCG):** gradient-based search for an adversarial **suffix** that universally induces compliance; transfers across models (white-box → black-box transfer). Tree-of-attacks/PAIR use an attacker LLM to iteratively refine jailbreaks.

**Prompt injection (the app-layer threat).** Distinct from jailbreaks: malicious instructions hidden in **data the model processes** (a web page, email, PDF, tool output) hijack an agent's behavior. **Indirect prompt injection** (Greshake et al.) is especially dangerous for tool-using agents — e.g., a webpage tells the agent to exfiltrate the user's data. This is widely regarded as the top *security* problem for LLM applications.

**Red-teaming.** Manual (human experts, domain specialists for bio/cyber), **automated** (an attacker LLM generates test cases — Perez et al.), and **algorithmic** (GCG, fuzzing). Outputs feed safety training (adversarial examples) and deployment decisions.

## Key Challenges
- **Adversarial arms race.** Every defense invites new attacks; robustness is not a solved, static property.
- **Transferability.** White-box attacks transfer to black-box models; open weights enable attack development.
- **Capability–safety tension.** More capable, instruction-following models are often easier to manipulate (competing objectives).
- **Prompt injection has no clean fix.** Because LLMs don't reliably separate "instructions" from "data," injection is hard to eliminate — a fundamental architecture issue.
- **Evaluation.** Measuring "robustness" is hard; attack success rates depend heavily on the attack set.

## Solutions & Current Best Practices
**Defense-in-depth, not a single fix:** (1) **safety training** with adversarial/red-team data (RLHF/CAI, refusal training, circuit-breakers/representation engineering that disrupt harmful internal states); (2) **input/output classifiers and guardrails** (Llama Guard, moderation APIs, **constitutional classifiers** — Anthropic 2025 — that robustly catch jailbreak attempts); (3) **system-level controls** for agents: privilege separation, sandboxing, human approval for risky actions, treating tool outputs as untrusted, and instruction/data separation; (4) **monitoring + rapid patching**; (5) **dangerous-capability evals** and refusing to deploy past risk thresholds. For prompt injection specifically: least-privilege, output filtering, and not letting untrusted content issue privileged actions.

## Lab Perspectives
- **Anthropic:** extensive red-teaming, **Constitutional Classifiers** (2025, strong jailbreak resistance), many-shot jailbreak discovery, public jailbreak bounties; ties to Responsible Scaling Policy.
- **OpenAI:** Preparedness red-teaming, external red-team network, moderation tooling, deliberative alignment for refusal robustness.
- **Google DeepMind:** Frontier Safety Framework, automated red-teaming, security research on injection.
- **Meta:** **Purple Llama / Llama Guard** open safety classifiers; open-weights raise distinct red-teaming responsibilities.

## Latest Developments (2023–2026)
**Many-shot jailbreaking** exploited long context; **automated/transferable attacks** (GCG, PAIR, TAP) matured; **prompt injection** became the central agent-security concern as tool use proliferated. Defenses advanced via **constitutional classifiers**, **circuit breakers / representation engineering**, and **deliberative alignment**. Governments and labs formalized **dangerous-capability evals** (bio, cyber, autonomy). The consensus: jailbreak robustness is improving but **not solved**, and injection is an open architectural problem.

## Interview Angles
> 💡 **What labs actually ask:**
- **"Why does safety training fail under jailbreaks?"** Competing objectives + mismatched generalization (Wei et al.).
- **"Jailbreak vs prompt injection — distinguish them."** Bypassing refusals via crafted user input vs malicious instructions hidden in processed data (esp. for agents).
- **"How does GCG work?"** Gradient-based search for an adversarial suffix maximizing compliance probability; transfers to black-box models.
- **"How would you defend a tool-using agent against indirect injection?"** Least-privilege, untrusted-data handling, sandboxing, human approval, output/instruction separation — no single fix.

## Open Problems
Achieving **robust** safety (provable or near-provable resistance to adversarial inputs) is unsolved. **Prompt injection** lacks a fundamental solution because LLMs don't separate instructions from data. Reliable red-teaming coverage, defending open-weights models, and evaluating robustness against unknown future attacks remain open and adversarial.

## References
- Zou, A. et al. (2023). *Universal and Transferable Adversarial Attacks (GCG).* arXiv:2307.15043.
- Wei, A. et al. (2023). *Jailbroken: How Does LLM Safety Training Fail?* arXiv:2307.02483.
- Perez, E. et al. (2022). *Red Teaming LMs with LMs.* arXiv:2202.03286.
- Greshake, K. et al. (2023). *Indirect Prompt Injection.* arXiv:2302.12173.
- Anil, C. et al. (2024). *Many-shot Jailbreaking.* (Anthropic).
