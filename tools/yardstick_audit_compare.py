"""Compare the two independent recomputations of the H-YARDSTICK table.

``results/yardstick_audit/per_seed.json`` is produced by
``tools/yardstick_audit.py`` on the host: quadratic weighted kappa implemented
from its definition in numpy, CORAL rank decode re-implemented.

``/work/lane_runs/an01_yardstick/sklearn_check.json`` is produced by
``tools/yardstick_audit_sklearn_check.py`` inside the trainer container: the
project's own ``eval.metrics._point_metrics`` (i.e.
``sklearn.metrics.cohen_kappa_score(weights="quadratic")``) and the project's own
``train.model.coral_decode``.

This script asserts the two agree, and prints the gate verdict per model per
yardstick. Exits non-zero if they do not.

    python3 tools/yardstick_audit_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GATE = 0.70
TOLERANCE = 1e-12


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    mine = json.loads((here / "results/yardstick_audit/per_seed.json").read_text())
    theirs = json.loads(
        (Path("/home/tun/iddsi/lane_runs/an01_yardstick/sklearn_check.json")).read_text()
    )

    index = {
        (row["version"], row["seed"], row["yardstick"]): row["wk1_kappa_covered"]
        for row in mine
    }
    worst = 0.0
    for row in theirs:
        key = (row["version"], row["seed"], row["yardstick"])
        if key not in index:
            raise SystemExit(f"no host-side row for {key}")
        worst = max(worst, abs(index[key] - row["wk1_weighted_kappa_covered"]))
    print(f"cells compared: {len(theirs)}; max abs diff between the two implementations: {worst:.3e}")
    if worst > TOLERANCE:
        raise SystemExit("MISMATCH: the two independent recomputations disagree")

    print()
    print("Gate verdict, wk1 = quadratic weighted kappa on covered rows, gate >= 0.70")
    print()
    print("| yardstick | model | n events/seed | seeds | wk1 range | seeds passing | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    verdicts = {}
    for yardstick in sorted({row["yardstick"] for row in mine}):
        for version in ("v0.4", "v0.5", "v0.7"):
            rows = [
                r
                for r in mine
                if r["yardstick"] == yardstick and r["version"] == version
            ]
            if not rows:
                continue
            values = [r["wk1_kappa_covered"] for r in rows]
            passing = sum(v >= GATE for v in values)
            verdicts[(yardstick, version)] = passing == len(values)
            print(
                f"| {yardstick} | {version} | {rows[0]['n_events']} | {len(values)} "
                f"| {min(values):.4f} - {max(values):.4f} | {passing}/{len(values)} "
                f"| {'PASS' if passing == len(values) else 'FAIL'} |"
            )
    print()
    claim_a = all(
        verdicts[("v0.5 consensus", v)] is False for v in ("v0.4", "v0.5", "v0.7")
    )
    claim_b = all(
        verdicts[("v0.7 full-coverage consensus", v)] is True for v in ("v0.4", "v0.5", "v0.7")
    )
    print(f"published claim 'all three fail against v0.5 labels'   : {'CONFIRMED' if claim_a else 'REFUTED'}")
    print(f"published claim 'all three pass against v0.7 labels'   : {'CONFIRMED' if claim_b else 'REFUTED'}")


if __name__ == "__main__":
    main()
