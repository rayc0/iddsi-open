#!/usr/bin/env python3
"""Annotate every results/**/report.md kappa with the label set it was measured
against (H-YARDSTICK). Annotation only: no metric value, denominator, CI or
provenance line is changed, added or removed.

The label set is a property of the run directory, stated from the release notes
in hf/model_README.md:
  v02_judgelabels   -> Qwen tier-2 judge labels (v0.2, row-wise stratified split)
  v03_groupdisjoint -> Qwen tier-2 judge labels (v0.3, group-disjoint split)
  v05_release/consensus_seed* -> v0.5 consensus labels
  fix1_siglip2_coral_seed17   -> Qwen tier-2 judge labels (v0.1 decode-fix run)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABEL_SETS = {
    "results/v02_judgelabels": "Qwen tier-2 judge labels (v0.2, row-wise stratified split)",
    "results/v03_groupdisjoint": "Qwen tier-2 judge labels (v0.3, group-disjoint split)",
    "results/v05_release": "v0.5 consensus labels",
    "results/fix1_siglip2_coral_seed17": "Qwen tier-2 judge labels (v0.1 CORAL decode-fix run)",
}

SCOPE_LINE = (
    "- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events."
)

ANCHOR = "- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label."


def annotation(label_set: str) -> str:
    return (
        f"- Label set for every kappa in this report (H-YARDSTICK): {label_set}. A weighted kappa is not\n"
        "  interpretable without the label set it was measured against: the same checkpoints fail the\n"
        "  wk1>=0.70 gate on the v0.5 consensus and pass it on the v0.7 full-coverage consensus\n"
        "  (v0.7: 0.838-0.860). Population and coverage are unchanged by this note.\n"
    )


def main() -> int:
    changed = 0
    for rel, label_set in LABEL_SETS.items():
        directory = ROOT / rel
        if not directory.is_dir():
            print(f"MISSING directory: {rel}", file=sys.stderr)
            return 1
        for path in sorted(directory.glob("*report.md")):
            text = path.read_text(encoding="utf-8")
            if "H-YARDSTICK" in text:
                print(f"already annotated: {path.relative_to(ROOT)}")
                continue
            if text.count(SCOPE_LINE) != 1:
                print(f"UNEXPECTED scope line count in {path.relative_to(ROOT)}", file=sys.stderr)
                return 1
            if text.count(ANCHOR) != 1:
                print(f"UNEXPECTED anchor count in {path.relative_to(ROOT)}", file=sys.stderr)
                return 1
            text = text.replace(ANCHOR, annotation(label_set) + ANCHOR, 1)
            path.write_text(text, encoding="utf-8")
            changed += 1
            print(f"annotated: {path.relative_to(ROOT)}")
    print(f"\n{changed} reports annotated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
