"""Render step2_decode_rule_comparison.json as a readable markdown table."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/work/lane_runs/fix01_decode/analysis")
SOURCE = OUT / "step2_decode_rule_comparison.json"


def pct(value, digits=1):
    return "n/a" if value is None else f"{100 * float(value):.{digits}f}%"


def num(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digigits}f}"


def row_for(scored: dict) -> list[str]:
    per = scored["per_level_recall"]
    cells = [
        f'{per["L5"]["recall_post_abstention_count"]}',
        f'{per["L6"]["recall_post_abstention_count"]}',
        pct(scored["coverage"]),
        num(scored["dangerous_underclassification_fnr_covered"]),
        num(scored["dangerous_fnr_detector"]),
        num(scored["weighted_kappa_covered"]),
        num(scored["macro_f1_with_abstentions"]),
    ]
    return cells


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# Decode-rule comparison on the v0.7 VALIDATION split (n=110; L3 14 / L4 8 / L5 6 / L6 9 / L7 73)")
    lines.append("")
    lines.append("Every decode parameter and every abstention threshold below is fitted on validation only.")
    lines.append("The test split is reported with parameters frozen on validation and is never used to choose a rule.")
    lines.append("")
    lines.append("Denominators: `L5 rec` and `L6 rec` are `correct/support` out of 6 and 9 validation events per seed.")
    lines.append("`FNR cov` = dangerous errors over the covered subset (the gate-facing number, H-ABSTBLIND).")
    lines.append("`FNR det` = dangerous errors that slip through over all dangerous errors.")
    lines.append("")

    for seed, entry in report["seeds"].items():
        lines.append(f"## seed {seed}")
        lines.append("")
        lines.append("| rule | L5 rec | L6 rec | coverage | FNR cov | FNR det | kappa cov | macro-F1 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, protocols in entry["rules"].items():
            for protocol in ("out_of_fold", "in_sample", "test"):
                scored = protocols.get(protocol)
                if scored is None:
                    continue
                label = {"out_of_fold": "OOF", "in_sample": "in-sample", "test": "TEST (frozen)"}[protocol]
                lines.append(f"| {name} [{label}] | " + " | ".join(row_for(scored)) + " |")
        lines.append("")

    lines.append("## pooled out-of-fold validation (5 seeds x 110 = 550 events)")
    lines.append("")
    lines.append("| rule | L5 rec | L6 rec | coverage | FNR cov | FNR det | macro rec |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, pooled in report["pooled_out_of_fold_validation"]["per_rule"].items():
        per = pooled["per_level_recall_post_abstention"]
        lines.append(
            f"| {name} | {per['L5']['correct']}/{per['L5']['support']} ({pct(per['L5']['recall'])}) "
            f"| {per['L6']['correct']}/{per['L6']['support']} ({pct(per['L6']['recall'])}) "
            f"| {pct(pooled['coverage'])} "
            f"| {num(pooled['dangerous_underclassification_fnr_covered'])} "
            f"| {num(pooled['dangerous_fnr_detector'])} "
            f"| {pct(pooled['macro_recall_post_abstention'])} |"
        )
    lines.append("")
    lines.append("## pre-abstention decode census, pooled out-of-fold validation")
    lines.append("")
    lines.append("| rule | L3 | L4 | L5 | L6 | L7 |")
    lines.append("|---|---|---|---|---|---|")
    for name, pooled in report["pooled_out_of_fold_validation"]["per_rule"].items():
        counts = pooled["decoded_level_counts_pre_abstention"]
        lines.append(f"| {name} | " + " | ".join(str(counts[f"L{l}"]) for l in range(3, 8)) + " |")
    lines.append("")
    lines.append("## post-abstention decided census, pooled out-of-fold validation")
    lines.append("")
    lines.append("| rule | L3 | L4 | L5 | L6 | L7 |")
    lines.append("|---|---|---|---|---|---|")
    for name, pooled in report["pooled_out_of_fold_validation"]["per_rule"].items():
        counts = pooled["decided_level_counts_post_abstention"]
        lines.append(f"| {name} | " + " | ".join(str(counts[f"L{l}"]) for l in range(3, 8)) + " |")
    lines.append("")

    (OUT / "step2_decode_rule_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
