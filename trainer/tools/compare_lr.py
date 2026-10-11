#!/usr/bin/env python3
"""Aggregate exp01_lr learning-rate sweep results against the v0.7 reference.

Usage:  python3 compare_lr.py [--root /work/lane_runs/exp01_lr]

Reads every ``<root>/lr_*/<name>_seed*/metrics.json`` produced by the sweep and
prints the quadratic weighted kappa (and the gate-relevant companions) per seed
per variant, next to the v0.7 reference read from
``results/v07_release/v07_seed*_metrics.json``.

Denominators are printed explicitly: weighted_kappa_covered is computed over
the ACCEPTED (confidence >= calibrated threshold) subset of the 108-event test
split, so ``n_kappa`` is the number of events the kappa actually saw.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def _find_repo_root() -> Path:
    """Locate the clone root by walking up to the directory holding results/v07_release."""
    for candidate in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (candidate / "results" / "v07_release").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


REPO = _find_repo_root()
REFERENCE_DIR = REPO / "results" / "v07_release"
REFERENCE_SEEDS = [17, 29, 43, 71, 101]

METRICS = [
    "weighted_kappa_covered",
    "macro_f1_with_abstentions",
    "coverage",
    "abstention_rate",
    "dangerous_underclassification_fnr_covered",
    "dangerous_underclassification_rate_all",
    "ece_covered",
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def summarise(label: str, path: Path) -> dict | None:
    data = load(path)
    if data is None:
        return None
    metrics = data.get("metrics", {})
    # n_kappa: the accepted count = coverage * n (kappa is computed on the covered subset)
    n = int(data.get("n", 0))
    coverage = metrics.get("coverage", float("nan"))
    accepted = int(round(coverage * n)) if n else 0
    row = {"label": label, "path": str(path), "n_test": n, "n_kappa": accepted}
    for name in METRICS:
        row[name] = metrics.get(name, float("nan"))
    calibration = load(path.with_name("calibration.json")) or {}
    row["calibration_target_met"] = calibration.get("target_met")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/lane_runs/exp01_lr")
    args = parser.parse_args()
    root = Path(args.root)

    print("=" * 100)
    print("REFERENCE — v0.7 full-coverage consensus (results/v07_release/)")
    print("=" * 100)
    reference: dict[int, dict] = {}
    for seed in REFERENCE_SEEDS:
        row = summarise(f"v0.7 seed {seed}", REFERENCE_DIR / f"v07_seed{seed}_metrics.json")
        if row is None:
            print(f"  seed {seed}: MISSING")
            continue
        reference[seed] = row
        print(
            f"  seed {seed:>3}: wk1={row['weighted_kappa_covered']:.4f}  "
            f"macroF1={row['macro_f1_with_abstentions']:.4f}  cov={row['coverage']:.4f} "
            f"({row['n_kappa']}/{row['n_test']})  undFNR={row['dangerous_underclassification_fnr_covered']:.4f}"
        )
    ref_values = [r["weighted_kappa_covered"] for r in reference.values()]
    if ref_values:
        print(
            f"  --> 5-seed mean wk1 = {sum(ref_values) / len(ref_values):.4f}   "
            f"min = {min(ref_values):.4f}   max = {max(ref_values):.4f}"
        )
        two = [reference[s]["weighted_kappa_covered"] for s in (17, 101) if s in reference]
        if len(two) == 2:
            print(f"  --> seeds [17,101] only: {two}  mean = {sum(two) / 2:.4f}")

    print()
    print("=" * 100)
    print("SWEEP — exp01_lr learning-rate variants (encoder lr, head lr = 10x)")
    print("=" * 100)
    variants = sorted(p for p in root.glob("lr_*") if p.is_dir())
    if not variants:
        print("  no lr_* variant directories found under", root)
    rows: list[dict] = []
    for variant in variants:
        for metrics_path in sorted(variant.glob("*_seed*/metrics.json")):
            row = summarise(metrics_path.parent.name, metrics_path)
            if row is None:
                continue
            row["variant"] = variant.name
            rows.append(row)
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant.name]
        if not subset:
            print(f"  {variant.name}: no completed seeds")
            continue
        print(f"  {variant.name}:")
        for row in subset:
            print(
                f"    {row['label']}: wk1={row['weighted_kappa_covered']:.4f}  "
                f"macroF1={row['macro_f1_with_abstentions']:.4f}  cov={row['coverage']:.4f} "
                f"({row['n_kappa']}/{row['n_test']})  undFNR={row['dangerous_underclassification_fnr_covered']:.4f}  "
                f"gate_met={row['calibration_target_met']}"
            )
        values = [r["weighted_kappa_covered"] for r in subset]
        print(
            f"    --> {len(values)}-seed mean wk1 = {sum(values) / len(values):.4f}   "
            f"min = {min(values):.4f}   max = {max(values):.4f}"
        )

    print()
    print("=" * 100)
    print("VERDICT vs the v0.7 reference (same seeds [17,101])")
    print("=" * 100)
    baseline = [reference[s]["weighted_kappa_covered"] for s in (17, 101) if s in reference]
    if baseline:
        base_mean = sum(baseline) / len(baseline)
        print(f"  v0.7 seeds [17,101] mean wk1 = {base_mean:.4f}")
        for variant in variants:
            subset = [r for r in rows if r["variant"] == variant.name]
            if not subset:
                continue
            mean = sum(r["weighted_kappa_covered"] for r in subset) / len(subset)
            delta = mean - base_mean
            better = "BEATS" if delta > 0 else ("ties" if abs(delta) < 1e-9 else "LOSES TO")
            print(f"  {variant.name:<28} mean wk1 = {mean:.4f}   delta = {delta:+.4f}   {better} v0.7")


if __name__ == "__main__":
    main()
