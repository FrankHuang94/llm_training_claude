#!/usr/bin/env python3
"""Build a print-ready PDF of the (Chinese) LLM knowledge database.

Pipeline: INDEX.md (chapter/file order) -> one combined Markdown -> pandoc (HTML5 + MathML)
-> Chromium via Playwright (render_pdf.js) -> PDF. The table of contents gets real page
numbers by a two-pass render: pass 1 reads the PDF outline (bookmarks) to find the page each
chapter/topic starts on, pass 2 writes those numbers into the TOC. The cover is rendered
separately without running header/footer and prepended.

Requirements: pandoc (>=3), node + playwright (chromium), python3 + pypdf,
fonts: Noto Serif/Sans CJK SC, Noto Color Emoji, STIX Math (or Latin Modern Math).

Usage (from repository root):
    python3 llm-knowledge-db-zh-CN/_build/build_pdf.py
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # llm-knowledge-db-zh-CN/

CHAPTER_RE = re.compile(r"^##\s+(\d\d_[A-Za-z0-9_]+)\s*[—-]+\s*(.+?)\s*$")
FILE_LINK_RE = re.compile(r"^\s*-\s+\[`?([^\]`]+?)`?\]\(([^)]+\.md)\)")
HEADING_RE = re.compile(r"^(#{1,6})(\s+.*)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
MD_LINK_RE = re.compile(r"\]\(((?:\.\./)?(?:[0-9]{2}_[A-Za-z0-9_]+/)?[0-9]{2}_[A-Za-z0-9_\-]+\.md)(#[^)]*)?\)")


def anchor_for(folder: str, stem: str) -> str:
    return f"f-{folder}-{stem}".replace("_", "-").lower()


def parse_index(index_md: str):
    """Return (front_text, chapters, back_text). chapters = [(folder, title, [relpath, ...])]."""
    lines = index_md.splitlines()
    chapters, front, back = [], [], []
    cur = None
    state = "front"
    for line in lines:
        m = CHAPTER_RE.match(line)
        if m:
            state = "chapters"
            cur = (m.group(1), m.group(2), [])
            chapters.append(cur)
            continue
        if state == "front":
            front.append(line)
            continue
        if state == "chapters":
            fm = FILE_LINK_RE.match(line)
            if fm:
                cur[2].append(fm.group(2))
                continue
            if line.startswith("## ") or line.strip() == "---" and cur[2]:
                # first non-chapter section after the chapter list -> back matter
                if line.startswith("## "):
                    state = "back"
                    back.append(line)
                continue
            continue
        back.append(line)
    return "\n".join(front), chapters, "\n".join(back)


def transform_file(md: str, folder: str, stem: str) -> tuple[str, str]:
    """Shift headings by +1, give the topic heading an id, rewrite cross-file links."""
    out, title, in_fence = [], None, False
    for line in md.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m = HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                if level == 1 and title is None:
                    title = text
                    out.append(f"## {text} {{#{anchor_for(folder, stem)} .topic}}")
                    continue
                out.append("#" * min(level + 1, 6) + " " + text)
                continue

            def repl(mm: re.Match) -> str:
                target = mm.group(1)
                parts = target.replace("../", "").split("/")
                f = parts[0] if len(parts) == 2 else folder
                s = Path(parts[-1]).stem
                return f"](#{anchor_for(f, s)})"

            line = MD_LINK_RE.sub(repl, line)
        out.append(line)
    return "\n".join(out), title or stem


def demote(md: str, by: int = 1) -> str:
    out, in_fence = [], False
    for line in md.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            m = HEADING_RE.match(line)
            if m:
                line = "#" * max(1, min(len(m.group(1)) + by, 6)) + m.group(2)
        out.append(line)
    return "\n".join(out)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"command failed: {cmd[0]}")
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    return r.stdout


def pandoc_html(markdown: str) -> str:
    return run(
        ["pandoc", "-f", "markdown+tex_math_dollars-implicit_figures", "-t", "html5",
         "--mathml", "--highlight-style=tango", "--wrap=none"],
        input=markdown,
    )


def page_shell(body: str, css: str, lang: str, title: str) -> str:
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
            f"<title>{html.escape(title)}</title><style>{css}</style></head>"
            f"<body>{body}</body></html>")


def outline_pages(pdf_path: Path, max_depth: int = 1) -> list[tuple[str, int]]:
    reader = PdfReader(str(pdf_path))
    found: list[tuple[str, int]] = []

    def walk(items, depth):
        for it in items:
            if isinstance(it, list):
                if depth < max_depth:
                    walk(it, depth + 1)
                continue
            found.append((it.title.strip(), reader.get_destination_page_number(it) + 1))

    walk(reader.outline, 0)
    return found


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", html.unescape(s))


def match_outline(toc_entries, found) -> dict[str, str]:
    """Walk the (flattened, document-ordered) outline and match each TOC entry in order."""
    pages, i = {}, 0
    for _, text, anchor in toc_entries:
        want = _norm(text)
        while i < len(found) and not _norm(found[i][0]).startswith(want):
            i += 1
        if i == len(found):
            raise SystemExit(f"heading not found in PDF outline: {text!r}")
        pages[anchor] = str(found[i][1])
        i += 1
    return pages


def build(args):
    src = Path(args.src).resolve()
    css = (HERE / "print.css").read_text(encoding="utf-8")
    index_md = (src / "INDEX.md").read_text(encoding="utf-8")
    front, chapters, back = parse_index(index_md)

    # --- front matter (preface) --------------------------------------------------
    front_body = re.sub(r"^#\s+.*$", "", front, count=1, flags=re.M).strip()
    preface_md = f"# {args.preface_title} {{#preface .front}}\n\n"
    readme = src / "README.md"
    if readme.exists():
        preface_md += demote(re.sub(r"^#\s+.*$", "", readme.read_text(encoding="utf-8"),
                                    count=1, flags=re.M).strip()) + "\n\n"
    preface_md += front_body.replace("\n---", "\n") + "\n\n" + demote(back) + "\n"

    toc_entries = [("front", args.preface_title, "preface")]
    parts = [preface_md]

    # --- optional "what's new" digest (kept at h1 level, links rewritten) -------------
    whats_new = src / "WHATS_NEW_2026.md"
    if whats_new.exists():
        body, wtitle = transform_file(whats_new.read_text(encoding="utf-8"), "", "whats-new")
        body = demote(body, -1).replace(f"{{#{anchor_for('', 'whats-new')} .topic}}",
                                        "{#whats-new .front}", 1)
        toc_entries.append(("front", wtitle, "whats-new"))
        parts.append(body)

    # --- chapters ------------------------------------------------------------------
    for ci, (folder, ctitle, files) in enumerate(chapters):
        num = folder.split("_")[0]
        cid = f"ch-{num}"
        label = args.chapter_label.format(n=int(num) + 1)
        toc_entries.append(("chapter", f"{label}　{ctitle}", cid))
        parts.append(f'# {label}　{ctitle} {{#{cid} .chapter}}\n\n'
                     f'<div class="chapter-folder">{html.escape(folder)}/</div>\n')
        for rel in files:
            path = (src / rel).resolve()
            fstem = path.stem
            ffolder = path.parent.name
            body, ftitle = transform_file(path.read_text(encoding="utf-8"), ffolder, fstem)
            toc_entries.append(("topic", ftitle, anchor_for(ffolder, fstem)))
            parts.append(body)
    # Links to files outside the book (README/GLOSSARY/PDF/English folder) become plain text.
    combined = re.sub(r"(?<!!)\[([^\]]+)\]\((?!https?:|#)[^)]*\)", r"\1", "\n\n".join(parts))
    content_html = pandoc_html(combined)

    def toc_html(pages: dict[str, str]) -> str:
        rows = []
        for kind, text, anchor in toc_entries:
            pg = pages.get(anchor, "000")
            rows.append(f'<a class="toc-{kind}" href="#{anchor}"><span class="t">'
                        f"{html.escape(text)}</span><span class=\"dots\"></span>"
                        f'<span class="p">{pg}</span></a>')
        return (f'<nav class="toc"><div class="toc-title">{html.escape(args.toc_title)}</div>'
                + "".join(rows) + "</nav>")

    cover_html = page_shell(
        f'<section class="cover"><div class="cover-top">{html.escape(args.cover_kicker)}</div>'
        f'<h1 class="cover-title">{html.escape(args.title)}</h1>'
        f'<div class="cover-sub">{html.escape(args.subtitle)}</div>'
        f'<div class="cover-rule"></div>'
        f'<div class="cover-note">{html.escape(args.cover_note)}</div>'
        f'<div class="cover-date">{html.escape(args.date_line)}</div></section>',
        css, args.lang, args.title)

    out_pdf = Path(args.out).resolve()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "cover.html").write_text(cover_html, encoding="utf-8")
        run(["node", str(HERE / "render_pdf.js"), str(td / "cover.html"), str(td / "cover.pdf"), "--plain"])

        pages: dict[str, str] = {}
        for attempt in range(3):
            body_html = page_shell(toc_html(pages) + content_html, css, args.lang, args.title)
            (td / "body.html").write_text(body_html, encoding="utf-8")
            run(["node", str(HERE / "render_pdf.js"), str(td / "body.html"), str(td / "body.pdf"), args.header])
            found = outline_pages(td / "body.pdf", max_depth=6)
            new_pages = match_outline(toc_entries, found)
            if new_pages == pages:
                break
            pages = new_pages
        if args.keep_html:
            Path(args.keep_html).write_text(body_html, encoding="utf-8")

        writer = PdfWriter()
        writer.append(str(td / "cover.pdf"))
        writer.append(str(td / "body.pdf"))  # keeps the bookmark outline
        writer.add_metadata({"/Title": args.title, "/Subject": args.subtitle,
                             "/Creator": "pandoc + Chromium (Playwright)"})
        writer.page_mode = "/UseOutlines"
        with open(out_pdf, "wb") as fh:
            writer.write(fh)
    n = len(PdfReader(str(out_pdf)).pages)
    print(f"wrote {out_pdf} ({n} pages)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT), help="knowledge-base folder containing INDEX.md")
    ap.add_argument("--out", default=str(ROOT / "LLM训练知识库_简体中文版.pdf"))
    ap.add_argument("--lang", default="zh-CN")
    ap.add_argument("--title", default="大语言模型训练知识库")
    ap.add_argument("--subtitle", default="LLM Training Knowledge Database · 简体中文参考译本")
    ap.add_argument("--header", default="大语言模型训练知识库 · 简体中文参考译本")
    ap.add_argument("--cover-kicker", default="研究参考 · 技术面试准备")
    ap.add_argument("--cover-note", default="本译本仅供参考；如有歧义，以英文原版（llm-knowledge-db/）为准。")
    ap.add_argument("--date-line", default="内容更新至 2026 年 11 月")
    ap.add_argument("--preface-title", default="前言与使用说明")
    ap.add_argument("--toc-title", default="目　录")
    ap.add_argument("--chapter-label", default="第 {n} 部分")
    ap.add_argument("--keep-html", default=None, help="also save the final body HTML here")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
