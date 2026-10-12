#!/usr/bin/env python3
"""Audit: does every kappa quoted in this repo name the label set it was measured
against (H-YARDSTICK)?

Scope rule: a "site" is one occurrence of kappa/κ in a text file. A site counts as
ANNOTATED if a line within WINDOW lines of it names a label set -- detected by
one of the markers below. Files that only *name* the metric without quoting a
value (e.g. an evaluation plan listing what a report will contain) are reported
separately so the distinction stays visible rather than being silently merged.

SEALED inputs are never read: capture/sealed_eval_plan.json and
capture/sealed_eval_pilot.json are excluded by path, and paper/paper.pdf is
excluded because it is a binary build artifact audited by
ops/annotate_paper_pdf.py instead.

Run: python3 -P ops/audit_kappa_label_set.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 14
SEALED = ("capture/sealed_eval_plan.json", "capture/sealed_eval_pilot.json")
SKIP_SUFFIX = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".pyc"}
SKIP_DIRS = {".git", "__pycache__", "node_modules"}

MARKERS = (
    "label set for",
    "h-yardstick",
    "label sets",
    "against the v0.5 consensus",
    "against the v0.7",
    "qwen tier-2 judge labels",
    "judge labels",
    "consensus label set",
    "single-operator physical-test",
    "not stated by the caller",
    "not comparable to any model kappa",
)


def tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files = [ROOT / line for line in out if line]
    return [
        p
        for p in files
        if p.suffix.lower() not in SKIP_SUFFIX
        and not any(part in SKIP_DIRS for part in p.parts)
        and p.is_file()
    ]


def logical_lines(path: Path, lines: list[str]) -> list[str]:
    """Collapse HTML tables so one table is one logical line.

    pandoc's HTML output puts each cell on its own line, so a results table
    whose caption names the label set would otherwise read as a row of
    unannotated kappa sites. Joining the lines inside <table>...</table> keeps
    the audit honest about tables without loosening it anywhere else: prose,
    headings and list items are returned unchanged.
    """
    if path.suffix.lower() != ".html":
        return lines
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for line in lines:
        opens = line.count("<table")
        closes = line.count("</table>")
        if depth or opens:
            buf.append(line)
            depth += opens - closes
            if depth <= 0:
                out.append(" ".join(buf))
                buf = []
                depth = 0
            continue
        out.append(line)
    out.extend(buf)
    return out


def main() -> int:
    unannotated: list[tuple[Path, int, str]] = []
    annotated_files: set[Path] = set()
    total_sites = 0

    for path in tracked_text_files():
        rel = str(path.relative_to(ROOT))
        if rel in SEALED:
            print(f"SEALED, not read: {rel}")
            continue
        try:
            lines = logical_lines(
                path, path.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except OSError as exc:
            print(f"unreadable: {rel} ({exc})", file=sys.stderr)
            continue
        hits = [i for i, line in enumerate(lines) if re.search(r"kappa|κ", line, re.I)]
        if not hits:
            continue
        total_sites += len(hits)
        bad = []
        for i in hits:
            window = "\n".join(lines[max(0, i - WINDOW) : i + WINDOW + 1]).lower()
            if not any(marker in window for marker in MARKERS):
                bad.append((i + 1, lines[i].strip()[:90]))
        if bad:
            unannotated.extend((path, n, t) for n, t in bad)
        else:
            annotated_files.add(path)

    print(f"\nfiles scanned with a kappa mention: see below")
    print(f"total kappa/κ sites in tracked text files: {total_sites}")
    print(f"sites with a label-set annotation within {WINDOW} lines: {total_sites - len(unannotated)}")
    print(f"sites WITHOUT one: {len(unannotated)}")
    for path, lineno, text in unannotated:
        print(f"  {path.relative_to(ROOT)}:{lineno}: {text}")
    return 1 if unannotated else 0


if __name__ == "__main__":
    raise SystemExit(main())
