# 大语言模型训练知识库 · 简体中文参考译本

> ⚠️ **仅供参考。** 本目录是英文知识库 [`../llm-knowledge-db/`](../llm-knowledge-db/) 的简体中文译本，内容与英文版 2026-11-04 的版本同步。翻译力求忠实、准确，但难免存在疏漏；**如有任何歧义或出入，请以英文原版为准**。英文版为唯一权威版本，后续更新将首先在英文版中进行。

## 目录结构
- 文件夹与文件名与英文版完全一致（例如 `03_posttraining/06_GRPO_and_RLVR.md`），便于中英对照阅读。
- [`INDEX.md`](INDEX.md)：总目录与推荐阅读顺序。
- [`WHATS_NEW_2026.md`](WHATS_NEW_2026.md)：2025 年年中至 2026 年 11 月的最新动态速览。
- [`GLOSSARY.md`](GLOSSARY.md)：术语对照表（统一译法）。
- [`LLM训练知识库_简体中文版.pdf`](LLM训练知识库_简体中文版.pdf)：可打印的 A4 PDF（封面、带页码的目录、PDF 书签、每个主题另起一页）。
- `_build/`：PDF 构建脚本。

## 翻译约定
- 模型名称、论文标题、作者、arXiv 编号、基准名称、代码与数学公式保留原文。
- 关键术语在每个文件中首次出现时采用"中文（English）"形式，便于与英文文献对照。
- 为区分 *reasoning* 与 *inference*，本译本将前者译为"推理"，后者译为"推断"。

## 重新生成 PDF
在仓库根目录运行：

```bash
python3 llm-knowledge-db-zh-CN/_build/build_pdf.py
```

依赖：pandoc ≥ 3、Node.js + Playwright（Chromium）、Python 3 + pypdf，以及 Noto Serif/Sans CJK SC、Noto Color Emoji、STIX Math 字体。
