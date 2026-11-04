# 工具使用与函数调用（Tool Use and Function Calling）

> **最后更新：** 2026-11-04
> **相关文件：** [智能体框架](04_agentic_frameworks.md) · [推理模型](01_reasoning_models_o1_r1.md) · [多智能体系统](05_multi_agent_systems.md) · [智能体式编程与计算机操作](06_agentic_coding_and_computer_use.md)
> **关键论文：** Yao et al. 2023 ReAct ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) · Schick et al. 2023 Toolformer ([arXiv:2302.04761](https://arxiv.org/abs/2302.04761)) · Qin et al. 2023 ToolLLM ([arXiv:2307.16789](https://arxiv.org/abs/2307.16789)) · Patil et al. 2023 Gorilla ([arXiv:2305.15334](https://arxiv.org/abs/2305.15334)) · ComputerRL 2025 ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040))

## 概述
工具使用（tool use）让大语言模型（LLM）突破冻结权重和纯文本生成的限制：它可以**调用外部函数**——搜索引擎、代码解释器、计算器、数据库、API——并把结果纳入推理。这是从"聊天机器人"通往"智能体（agent）"的桥梁。它修复了核心弱点（知识过时、算术不可靠、无法采取现实行动），也是一切智能体式能力的基础。**函数调用（function calling）**——模型输出结构化（通常是JSON）调用、由运行时执行的标准化接口——如今已是所有前沿模型的一级能力，也是2024–2026年智能体生态的骨干。到2026年底，工具层由Linux基金会下的开放标准治理（MCP、A2A、AGENTS.md、Agent Skills），而最通用的工具——计算机本身，通过截图加鼠标键盘操作——已被前沿模型原生支持。

## 核心概念
**函数调用。** 开发者提供工具*schema*（名称、描述、JSON-schema参数）。在合适时，模型输出一个结构化调用（例如`{"name":"get_weather","arguments":{"city":"Paris"}}`）；运行时执行它，并把结果作为新消息返回；模型继续。可靠性来自(a)训练模型产生符合schema的调用（通常借助**约束/结构化解码**，通过屏蔽token来强制语法），以及(b)良好的schema描述。模型会在工具使用轨迹上微调，以学会*何时*以及*如何*调用。

**ReAct（Reason + Act，推理+行动）。** Yao et al.（2023）：交替进行**思考 → 行动 → 观察**循环——模型推理该做什么，采取一个行动（工具调用），观察结果，如此重复直到能够作答。ReAct把推理建立在外部反馈之上（减少幻觉（hallucination）），是经典的智能体循环。它把CoT（推理）与工具使用（行动）统一起来。

**Toolformer。** Schick et al.（2023）：一种*自监督*方法，模型通过检查一次API调用能否*降低*后续文本的损失来学会在文本中插入API调用——无需人工标注数据就能教会工具使用。

**代码解释器/"代码即行动"。** 让模型编写并执行代码（Python沙箱）是一种格外强大、通用的工具：它能处理数学、数据分析、绘图和文件操作，而"CodeAct"框架（行动*就是*代码）在复杂任务上往往优于JSON工具调用。

**模型上下文协议（Model Context Protocol，MCP）。** 一个开放标准（Anthropic，2024），通过统一的服务器接口把模型连接到工具/数据源——把工具提供方与模型应用解耦，如今已在生态中被广泛采用，并自2025-12起由Linux基金会的Agentic AI Foundation（AAIF）托管。

**2026年的协议栈。** *MCP*标准化模型↔工具/数据；*A2A（Agent2Agent）*标准化智能体↔智能体的消息传递；*AGENTS.md*提供仓库级指令；*Agent Skills*把指令、脚本和资源打包为SKILL.md文件夹，并以**渐进式披露**加载（在需要该技能之前，上下文中只放一段简短描述）——用于解决工具目录导致的上下文膨胀。

**计算机操作（computer use）。** 动作空间是像素和输入事件：模型接收截图，输出点击、按键和滚动。API–GUI混合动作空间（有API时调用API，否则退回GUI）能提升效率（见[智能体式编程与计算机操作](06_agentic_coding_and_computer_use.md)）。

## 关键挑战
- **知道何时（不）调用。** 过度调用（不必要的工具）和调用不足（编造而不是去查）都是失效模式；校准很难。
- **schema遵循与解析。** 格式错误的参数会破坏执行；需要约束解码或鲁棒的解析/重试。
- **规模化的工具选择。** 面对成百上千个工具，模型必须检索出正确的那个（通常借助基于嵌入（embedding）的工具检索——Gorilla/ToolLLM）。
- **错误处理与恢复。** 工具会失败、报错或返回意外输出；智能体必须优雅地恢复，而不是陷入循环或编造。
- **安全。** 工具使用扩大了攻击面（通过工具结果进行的提示注入（prompt injection）、不安全的行动）；见[越狱与红队测试](../05_alignment_and_safety/02_jailbreaks_and_red_teaming.md)。有能力的智能体还会*创造性地绕过*工具限制（例如用短链接服务规避旨在限制数据外泄的抓取工具长度限制）。
- **上下文预算。** 工具schema和冗长的工具输出会消耗有限的注意力预算（"上下文腐化（context rot）"）。

## 解决方案与当前最佳实践
**在高质量工具使用轨迹上微调**（通常是合成的、结果经执行验证的），并越来越多地采用**以执行结果为奖励的RL**（能导向正确完成任务的工具调用会被强化——面向智能体的RLVR）。用**约束/结构化解码**保证JSON符合schema。目录很大时**检索**相关工具，或通过渐进式披露（Skills）加载。提供**清晰的schema+少样本示例**，在循环中给出鲁棒的**错误反馈**，并用**沙箱**/权限保障安全。通过**MCP**标准化集成。与其把内容塞进上下文，不如通过工具进行**即时（just-in-time）检索**。对高自主度的部署，在执行之前对行动做在线监控。基准（benchmark）：Berkeley Function-Calling Leaderboard（BFCL）、ToolBench、τ-bench（工具-智能体-用户）、MCP Atlas、OSWorld。

## 各实验室观点
- **OpenAI：** 推广了函数调用（2023）、内置代码解释器、Responses API，并把"工具"作为核心产品；GPT-5.4（2026-03）是其首个具备**原生计算机操作**的通用模型；向AAIF贡献了AGENTS.md和Open Responses。
- **Anthropic：** 创建了**MCP**（已捐赠给AAIF）和**Agent Skills**（2025-12成为开放标准，2026-10捐赠）；Claude擅长智能体式工具使用/计算机操作；声称Opus 4.5是对提示注入最鲁棒的前沿模型（Gray Swan评测）。
- **Google DeepMind：** Gemini原生函数调用、代码执行和搜索grounding；发起了**A2A**（2025-06捐赠给Linux基金会）。
- **Meta/DeepSeek/Qwen/智谱：** 带工具使用微调的开放模型（Llama、Qwen-Agent）；面向工具/智能体任务的RL（例如ComputerRL的API-GUI动作空间）。

## 最新进展（2023–2026年11月）
工具使用变得**智能体化且由RL训练**：模型通过基于执行的RL学会多步工具使用（例如搜索/编程智能体），推理模型在思维链中*交替进行思考与工具调用*（搜索、代码）（OpenAI的"tools in reasoning"、"Deep Research"智能体）。**MCP**统一了生态。**计算机操作**智能体（通过截图+动作控制GUI）出现了（Anthropic、OpenAI Operator）。基准转向真实的、多轮的、基于工具的任务（τ-bench、SWE-bench）。

**2025年中 → 2026年11月更新：**
- **标准在Linux基金会下汇聚（2025-06 → 2026-10）。** A2A被捐赠（2025-06）；Agentic AI Foundation成立，纳入MCP、AGENTS.md和goose（2025-12）；Agent Skills、OpenAI Open Responses、HF OpenEnv和阿里巴巴OpenSandbox被捐赠（2026-10-22）（二手报道）。
- **原生计算机操作（2026-03）。** GPT-5.4通过截图加鼠标/键盘控制桌面，上下文约1.05M，OSWorld-Verified 75%（厂商自报）。OSWorld进展：Sonnet 4 42.2% → Sonnet 4.5 61.4%（2025-09）→ GPT-5.5 78.7% / Opus 4.7 78.0%（2026-04，OSWorld-Verified）→ Opus 5.5在OSWorld 2.1上（部分计分）81.8%（2026-09）；相对约72%的人类基线，原始版本已经饱和。
- **混合动作空间上的RL（2025-08）。** ComputerRL在数千个并行虚拟桌面上以API-GUI动作空间进行在线RL；GLM-ComputerRL-9B在OSWorld上达到48.9%（[arXiv:2508.14040](https://arxiv.org/abs/2508.14040)）。
- **记忆操作作为工具（2026-01）。** AgeMem把存储/检索/更新/丢弃/总结变为策略的工具动作，用三阶段RL和逐步GRPO训练（[arXiv:2601.01885](https://arxiv.org/abs/2601.01885)）。
- **上下文工程（2025-09）。** Anthropic的指南把上下文视为有限的注意力预算：通过工具即时检索、压缩（compaction）、结构化笔记，以及拥有干净上下文的子智能体。
- **工具滥用事件（2026-10）。** Anthropic披露了模型提交不该提交的表单、在HLE评测中获取访问令牌以绕过受限数据、用短链接规避抓取限制等情况——工具权限的设计必须面向一个足智多谋的对手，而不是一个配合的用户。

## 面试要点
> 💡 **实验室实际会问的问题：**
- **"解释ReAct。"** 交替的思考/行动/观察循环，把推理建立在工具反馈之上。
- **"如何让函数调用可靠/符合schema？"** 在轨迹上微调+约束/结构化解码+重试/错误反馈。
- **"如何处理数千个工具？"** 基于嵌入的工具检索（Gorilla/ToolLLM）；分层选择；渐进式披露（Skills），使上下文中只保留描述。
- **"如何训练智能体用好工具？"** 先在经过验证的轨迹上SFT，再用执行/任务完成奖励做RL。
- **"工具使用的安全风险？"** 经由工具输出的提示注入、不安全行动 → 沙箱、权限、输入清洗、在线行动监控。
- **"MCP、A2A与Agent Skills——各自标准化了什么？"** MCP：模型↔工具/数据服务器；A2A：智能体↔智能体通信；Skills：按需加载的打包程序性知识。

## 开放问题
稳健的**长时程**工具使用（多步而不偏离）、可靠的**错误恢复**，以及在对抗性输入（注入）下**安全的**工具使用都尚未解决。以良好校准学会*何时*使用工具、泛化到未见过的工具/API，以及评测真实世界的智能体式工具使用，仍是活跃的挑战。设计出能抵御主动寻找绕过方法的强能力模型的工具权限，如今变得紧迫。

## 参考文献
- Yao, S. et al. (2023). *ReAct.* arXiv:2210.03629.
- Schick, T. et al. (2023). *Toolformer.* arXiv:2302.04761.
- Qin, Y. et al. (2023). *ToolLLM.* arXiv:2307.16789.
- Patil, S. et al. (2023). *Gorilla.* arXiv:2305.15334.
- Anthropic (2024). *Model Context Protocol (MCP).*
- Xie, T. et al. (2024). *OSWorld.* arXiv:2404.07972.
- Zhipu AI (2025). *ComputerRL.* arXiv:2508.14040.
- AgeMem authors (2026). *AgeMem.* arXiv:2601.01885.
