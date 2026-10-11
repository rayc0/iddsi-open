#!/usr/bin/env python3
"""Append trailing errata page(s) to paper/paper.pdf carrying the H-YARDSTICK
label-set caveat, so the PDF stops quoting kappas without saying which label set
they were measured against.

WHY TRAILING ERRATA AND NOT A REBUILD: paper/paper.pdf is a pandoc/LaTeX build
artifact. `paper/paper.md` and `paper/paper.html` are annotated in place, but no
TeX engine (pdflatex/xelatex/lualatex/tectonic/typst) exists in this container or
on this host, so `paper/build.sh` cannot be run here. Rather than re-flow body
text inside a justified LaTeX PDF, this appends clearly-labelled errata pages at
the END. Appending at the end shifts no existing page number, so the document's
contents and every existing page stay correct; no existing page is edited,
redacted or removed. The save is incremental, so the original file bytes stay in
place and the new pages are appended objects.

Run: python3 -P ops/annotate_paper_pdf.py
(pymupdf lives in the user site-packages; -P keeps sys.path safe without -I,
which would strip them.)
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"

PAGE_W, PAGE_H = 595.0, 842.0  # A4, matching the rest of the document
MARGIN, TOP = 56.0, 56.0
BOTTOM = PAGE_H - 56.0
LEADING = 11.4
SIZE = 8.6
TITLE_SIZE = 13.0

TITLE = "Erratum and annotation notice — label set for every quoted kappa (H-YARDSTICK)"

BODY = [
    "Added 2026-10-12 by lane rl03_yardstick_caveat. These pages are appended; every preceding page is",
    "unchanged, and no page number in this document's contents shifts because of them.",
    "",
    "Every kappa quoted in this note was measured against a label set, and the gate outcome is a",
    "property of that label set rather than of the model (H-YARDSTICK, hf/HAZARD_LOG.md). The",
    "annotation below is carried inline in the source of this document (paper/paper.md and",
    "paper/paper.html) and in hf/model_README.md, hf/HAZARD_LOG.md and README.md. It is reproduced here",
    "because this PDF is a build artifact that could not be regenerated where the annotation was made:",
    "no TeX engine was available to run paper/build.sh.",
    "",
    "The sentence this erratum adds to every kappa quoted above:",
    "",
    "    The v0.4/v0.5 numbers were measured against the v0.5 consensus label set and fail the",
    "    pre-registered wk1 >= 0.70 gate on it; the same models, measured against the v0.7",
    "    full-coverage consensus, pass it (v0.7: 0.838-0.860, 5 seeds, held-out n=108). Nothing about",
    "    the models changed between those two columns - only the yardstick did.",
    "",
    "Applied to the 17 kappa occurrences in this document, which sit on these pages:",
    "",
    "    page 1   (abstract)                        3 occurrences - the 0.42 model-eval figure and the",
    "                                              inter-judge 0.1062 / 0.2766 figures",
    "    page 3   (Section 1, contribution claim)   1 occurrence  - the 0.42 model-eval figure",
    "    page 6   (Section 3.3)                    3 occurrences - the 0.42 model-eval figure and the",
    "                                              inter-judge 0.1062 / 0.2766 figures",
    "    page 7   (Section 3.5, evaluation plan)    1 occurrence  - the metric the plan names; no value",
    "                                              is quoted there",
    "    page 8   (Section 3.5, results table)     3 occurrences - the 0.42 figure, the wk1 gate row",
    "                                              and the inter-judge row",
    "    page 10  (Sections 6 and 7)               6 occurrences - the 0.42 figure and the inter-judge",
    "                                              figures in the limitations and the conclusion",
    "",
    "Label set, stated per figure, because the two kinds of kappa are not comparable:",
    "",
    "    - The 0.42 model-eval kappa (CORAL decode fix, held-out n=109) was measured against the",
    "      Qwen tier-2 judge labels - the two-size, one-family weak label set of that run. It fails the",
    "      wk1 >= 0.70 gate on that yardstick.",
    "    - The v0.4/v0.5 model numbers were measured against the v0.5 consensus label set. They also",
    "      fail wk1 >= 0.70 on it.",
    "    - The 0.1062 / 0.2766 figures are agreement BETWEEN the two VLM judges who produced the label",
    "      set. They characterise the yardstick itself rather than a model scored against one, so they",
    "      have no model-kappa comparison of their own.",
    "    - The same checkpoints measured against the v0.7 full-coverage consensus pass wk1: v0.4",
    "      0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860 (5 seeds each).",
    "",
    "What this erratum does NOT do: it deletes no older number, moves neither the wk1 nor the wk2 gate,",
    "re-registers neither gate, does not change the release shape decided 2026-09-07, and does not alter",
    "the weight- and dataset-withholding decisions, which rest on the dangerous under-classification",
    "and calibration failures and hold on every yardstick. The wk2 33% failure is independent of the",
    "label set and stands on every reading.",
    "",
    "Reproduce: paper/paper.md and paper/paper.html carry the same sentence inline at each of the",
    "sections listed above. Rebuild this PDF from paper.md with paper/build.sh once a TeX engine is",
    "available, and these pages become redundant.",
]


def render() -> int:
    doc = pymupdf.open(PDF)
    before = doc.page_count
    if any("H-YARDSTICK" in page.get_text() for page in doc):
        print("already annotated")
        return 0

    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = TOP + TITLE_SIZE

    def newline(text: str, *, font: str, size: float, indent: float) -> None:
        nonlocal page, y
        if y > BOTTOM:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = TOP + size
        page.insert_text((indent, y), text, fontname=font, fontsize=size)
        y += LEADING

    page.insert_text((MARGIN, y), TITLE, fontname="hebo", fontsize=TITLE_SIZE)
    y += TITLE_SIZE + 12.0

    for line in BODY:
        if line == "":
            y += 6.0
            continue
        quoted = line.startswith("    ")
        newline(line.strip() if quoted else line, font="helv", size=SIZE,
                indent=MARGIN + (16.0 if quoted else 0.0))

    # Incremental save appends the new pages and leaves every original byte in
    # place, so the pre-existing pages are provably untouched.
    doc.save(PDF, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
    added = doc.page_count - before
    doc.close()
    print(f"appended {added} errata page(s); {PDF.relative_to(ROOT)} now has {before + added} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(render())
