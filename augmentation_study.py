#!/usr/bin/env python3
"""Compare the exp04_aug augmentation runs against the v0.7 reference.

Reads only JSON files that already exist on disk and prints a table. Every
number is stated with the split and the event count it was computed over.

Usage:
    python3 augmentation_study.py            # human-readable table
    python3 augmentation_study.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

def _resolve(path: Path) -> Path:
    """``/work/...`` inside the container, ``~/iddsi/...`` on the host."""
    if path.exists():
        return path
    host = Path("/home/tun/iddsi") / path.relative_to("/work")
    return host if host.exists() else path


LANE = _resolve(Path("/work/lanes/exp04_aug"))
RUNS = _resolve(Path("/work/lane_runs/exp04_aug"))
V07_REFERENCE = LANE / "results" / "v07_release"

#: Seeds this lane ran, per the lane brief.
SEEDS = [17, 101]

VARIANTS = {
    "baseline": RUNS / "baseline",
    "strong": RUNS / "strong_photometric",
}


def load_metrics(path: Path) -> dict | None:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarise(run_dir: Path, prefix: str) -> dict:
    """Collect one metrics.json per seed under ``run_dir``."""
    per_seed = []
    for seed_dir in sorted(run_dir.glob(f"{prefix}_seed*")):
        payload = load_metrics(seed_dir)
        if payload is None:
            continue
        per_seed.append(
            {
                "seed": payload.get("seed", seed_dir.name),
                "run_dir": str(seed_dir),
                "n_test_events": payload["n"],
                "coverage": payload["metrics"]["coverage"],
                "n_covered_events": round(payload["n"] * payload["metrics"]["coverage"]),
                "kappa": payload["metrics"]["weighted_kappa_covered"],
                "kappa_ci95": payload["confidence_intervals_95"]["weighted_kappa_covered"],
                "macro_f1": payload["metrics"]["macro_f1_with_abstentions"],
                "dangerous_fnr_covered": payload["metrics"]["dangerous_underclassification_fnr_covered"],
                "ece_all": payload["metrics"]["ece_all"],
                "threshold": payload["threshold"],
            }
        )
    return {"per_seed": per_seed}


def mean_kappa(summary: dict) -> float | None:
    values = [row["kappa"] for row in summary["per_seed"]]
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {"seeds": SEEDS, "variants": {}, "v0.7_reference": {}}

    print("=" * 96)
    print("exp04_aug — augmentation study on the v0.7 recipe")
    print(f"manifest: /work/data/manifest_v07_weighted.jsonl (532 rows: 314 train / 110 val / 108 test)")
    print(f"seeds run by this lane: {SEEDS}")
    print("=" * 96)

    for label, run_dir in VARIANTS.items():
        prefix = "exp04_aug_baseline" if label == "baseline" else "exp04_aug_strong"
        summary = summarise(run_dir, prefix)
        report["variants"][label] = summary
        print(f"\n[{label}] {run_dir}")
        if not summary["per_seed"]:
            print("  (no runs found)")
            continue
        for row in summary["per_seed"]:
            print(
                f"  seed {row['seed']:>4}: kappa {row['kappa']:.4f} "
                f"(95% CI {row['kappa_ci95'][0]:.4f}-{row['kappa_ci95'][1]:.4f}) | "
                f"covered {row['n_covered_events']}/{row['n_test_events']} test events | "
                f"macro-F1 {row['macro_f1']:.4f} | dangerous FNR {row['dangerous_fnr_covered']:.4f} | "
                f"ECE {row['ece_all']:.4f}"
            )
        print(f"  mean kappa over {len(summary['per_seed'])} seeds: {mean_kappa(summary):.4f}")

    # v0.7 reference, read from the canonical release directory.
    print("\n[v0.7 reference] results/v07_release (5 seeds, published)")
    ref = []
    for seed in (17, 29, 43, 71, 101):
        path = V07_REFERENCE / f"v07_seed{seed}_metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        ref.append(
            {
                "seed": seed,
                "kappa": payload["metrics"]["weighted_kappa_covered"],
                "n_test_events": payload["n"],
                "n_covered_events": round(payload["n"] * payload["metrics"]["coverage"]),
                "coverage": payload["metrics"]["coverage"],
            }
        )
    report["v0.7_reference"] = {"per_seed": ref}
    for row in ref:
        print(
            f"  seed {row['seed']:>4}: kappa {row['kappa']:.4f} | "
            f"covered {row['n_covered_events']}/{row['n_test_events']} test events"
        )
    if ref:
        values = [row["kappa"] for row in ref]
        print(
            f"  mean over {len(ref)} seeds: {statistics.fmean(values):.4f} "
            f"(range {min(values):.4f}-{max(values):.4f})"
        )
        same_seeds = [row["kappa"] for row in ref if row["seed"] in SEEDS]
        if same_seeds:
            print(
                f"  mean over this lane's seeds {SEEDS}: {statistics.fmean(same_seeds):.4f}"
            )

    base = mean_kappa(report["variants"]["baseline"])
    strong = mean_kappa(report["variants"]["strong"])
    if base is not None and strong is not None:
        print("\n" + "-" * 96)
        print(f"baseline (as shipped) mean kappa : {base:.4f}")
        print(f"strong augmentation mean kappa   : {strong:.4f}")
        print(f"difference (strong - baseline)   : {strong - base:+.4f}")
        winner = "strong" if strong > base else ("baseline" if base > strong else "tie")
        print(f"better variant                  : {winner}")
        report["verdict"] = {
            "baseline_mean_kappa": base,
            "strong_mean_kappa": strong,
            "delta": strong - base,
            "better": winner,
        }
    print("-" * 96)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
