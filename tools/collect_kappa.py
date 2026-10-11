#!/usr/bin/env python3
"""Collect quadratic weighted kappa for every exp05_loss run and compare it with
the v0.7 reference on the same two seeds.

Metric: metrics["weighted_kappa_covered"] -- Cohen's kappa with quadratic
weights, computed over the test-split events the model did not abstain on
(see trainer/eval/metrics.py:_point_metrics).  The v0.7 reference numbers are
read from results/v07_release/v07_seed*_metrics.json in this clone.

Usage: python3 collect_kappa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IDDSI_EXPERIMENT_REPO", "."))
RUNS = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("IDDSI_EXPERIMENT_RUNS", "runs"))
SEEDS = [17, 101]


def kappa_of(path: Path) -> dict:
    data = json.loads(path.read_text())
    m = data["metrics"]
    return {
        "kappa": m["weighted_kappa_covered"],
        "fnr": m["dangerous_underclassification_fnr_covered"],
        "coverage": m["coverage"],
        "macro_f1": m["macro_f1_with_abstentions"],
        "n": data["n"],
        "threshold": data["threshold"],
        "ci": data.get("confidence_intervals_95", {}).get("weighted_kappa_covered"),
        "calibration_nll": None,
    }


def main() -> int:
    ref = {}
    for seed in (17, 29, 43, 71, 101):
        p = REPO / "results" / "v07_release" / f"v07_seed{seed}_metrics.json"
        if p.exists():
            ref[seed] = kappa_of(p)["kappa"]

    print("=== v0.7 reference (test split, covered events, quadratic weighted kappa) ===")
    for seed, value in sorted(ref.items()):
        print(f"  seed {seed:>3}: {value:.4f}")
    if ref:
        print(f"  mean over {len(ref)} seeds: {sum(ref.values()) / len(ref):.4f}"
              f"  (min {min(ref.values()):.4f}, max {max(ref.values()):.4f})")
    ref_pair = [ref[s] for s in SEEDS if s in ref]
    if len(ref_pair) == len(SEEDS):
        print(f"  reference on MY seeds {SEEDS}: "
              f"{[round(v, 4) for v in ref_pair]} mean {sum(ref_pair) / len(ref_pair):.4f}")
    print()

    rows = []
    for arm_dir in sorted(RUNS.glob("*")):
        for seed in SEEDS:
            p = arm_dir / f"siglip2_v07_{arm_dir.name}_seed{seed}" / "metrics.json"
            if not p.exists():
                print(f"  MISSING {p}")
                continue
            entry = kappa_of(p)
            entry["arm"] = arm_dir.name
            entry["seed"] = seed
            entry["path"] = str(p)
            rows.append(entry)

    if not rows:
        print("no run metrics found under", RUNS)
        return 1

    print("=== exp05_loss loss-shaping runs ===")
    print(f"{'arm':<10} {'seed':>4} {'kappa':>8} {'d_vs_ref':>9} {'coverage':>9} "
          f"{'underFNR':>9} {'n':>5} {'95% CI':>18}")
    for r in rows:
        refv = ref.get(r["seed"])
        delta = f"{r['kappa'] - refv:+.4f}" if refv is not None else "n/a"
        ci = r["ci"]
        ci_s = f"[{ci[0]:.4f},{ci[1]:.4f}]" if ci else "n/a"
        print(f"{r['arm']:<10} {r['seed']:>4} {r['kappa']:>8.4f} {delta:>9} "
              f"{r['coverage']:>9.4f} {r['fnr']:>9.4f} {r['n']:>5} {ci_s:>18}")

    print()
    print("=== per-arm mean over seeds", SEEDS, "===")
    for arm in sorted({r["arm"] for r in rows}):
        vals = [r["kappa"] for r in rows if r["arm"] == arm and r["seed"] in SEEDS]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        base = sum(ref[s] for s in SEEDS if s in ref) / len(SEEDS) if ref else float("nan")
        verdict = "BEATS" if mean > base else "does not beat"
        print(f"  {arm:<10} n_seeds={len(vals)} mean kappa={mean:.4f}  "
              f"v0.7 mean on same seeds={base:.4f}  delta={mean - base:+.4f}  -> {verdict}")
        for r in sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"]):
            refv = ref.get(r["seed"])
            tag = f"seed {r['seed']}: {r['kappa']:.4f}"
            if refv is not None:
                tag += f" vs v0.7 {refv:.4f} ({r['kappa'] - refv:+.4f})"
            print(f"      {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
