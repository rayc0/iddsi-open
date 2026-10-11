#!/usr/bin/env python3
"""Compare the lane's control and variant arms against the v0.7 reference.

Everything printed here is read from a file; nothing is hard-coded. The v0.7
reference numbers are re-measured from ``results/v07_release/*_metrics.json`` at
run time so the report cannot silently drift from the artefacts on disk.

Usage (host or container, CPU only):

    python3 tools/schedule_compare.py \
        --reference results/v07_release \
        --arms control=/work/lane_runs/exp02_schedule/control_shipped \
        --arms variant=/work/lane_runs/exp02_schedule/sched2x_cosine \
        --out /work/lane_runs/exp02_schedule/comparison.json

Each ``--arms`` value is ``label=run_dir`` where ``run_dir`` is a directory that
holds ``<experiment_name>_seed<seed>/metrics.json`` sub-directories.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

KAPPA = "weighted_kappa_covered"
UNDER_FNR = "dangerous_underclassification_fnr_covered"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_seeds(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Collect one metrics dict per seed found under ``run_dir``."""
    found: dict[int, dict[str, Any]] = {}
    for metrics_path in sorted(run_dir.glob("*_seed*/metrics.json")):
        seed = int(metrics_path.parent.name.rsplit("seed", 1)[1])
        found[seed] = _load(metrics_path)
    if not found:
        raise SystemExit(f"no *_seed*/metrics.json under {run_dir}")
    return found


def read_reference(reference_dir: Path) -> dict[int, dict[str, Any]]:
    """The v0.7 release metrics, keyed by seed, re-measured from disk."""
    found: dict[int, dict[str, Any]] = {}
    for metrics_path in sorted(reference_dir.glob("v07_seed*_metrics.json")):
        seed = int(metrics_path.stem.split("_")[1].replace("seed", ""))
        found[seed] = _load(metrics_path)
    if not found:
        raise SystemExit(f"no v07_seed*_metrics.json under {reference_dir}")
    return found


def summarise(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    kappas = {seed: data["metrics"][KAPPA] for seed, data in sorted(seeds.items())}
    values = list(kappas.values())
    coverages = {seed: data["metrics"]["coverage"] for seed, data in sorted(seeds.items())}
    return {
        "seeds": sorted(seeds),
        "kappa_by_seed": kappas,
        "kappa_mean": statistics.fmean(values),
        "kappa_min": min(values),
        "kappa_max": max(values),
        "kappa_spread_max_minus_min": max(values) - min(values),
        "n_events": sorted({data["n"] for data in seeds.values()}),
        "coverage_by_seed": coverages,
        "under_fnr_by_seed": {
            seed: data["metrics"][UNDER_FNR] for seed, data in sorted(seeds.items())
        },
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--arms", required=True, action="append", help="label=run_dir")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reference = summarise(read_reference(args.reference))
    arms: dict[str, dict[str, Any]] = {}
    for spec in args.arms:
        label, _, raw_dir = spec.partition("=")
        if not raw_dir:
            raise SystemExit(f"--arms expects label=dir, got {spec!r}")
        arms[label] = summarise(read_seeds(Path(raw_dir)))

    shared_seeds = sorted(set.intersection(*(set(arm["seeds"]) for arm in arms.values()))) if arms else []
    if not shared_seeds:
        raise SystemExit("the arms do not share any seed")

    print(f"# v0.7 reference — re-measured from {args.reference}")
    print(f"seeds {reference['seeds']} — kappa by seed: "
          + ", ".join(f"{seed}:{reference['kappa_by_seed'][seed]:.4f}" for seed in reference["seeds"]))
    print(f"reference mean {reference['kappa_mean']:.4f}, "
          f"range {reference['kappa_min']:.4f}-{reference['kappa_max']:.4f} "
          f"(spread {reference['kappa_spread_max_minus_min']:.4f}), n={reference['n_events']}")
    print()
    print("| arm | seed | wk1 kappa | coverage | under-FNR (covered) | n |")
    print("|---|---|---:|---:|---:|---:|")
    for label, arm in arms.items():
        for seed in arm["seeds"]:
            print(f"| {label} | {seed} | {arm['kappa_by_seed'][seed]:.4f} | "
                  f"{arm['coverage_by_seed'][seed]:.4f} | {arm['under_fnr_by_seed'][seed]:.4f} | "
                  f"{arm['n_events'][0]} |")
    print()

    # The variant's gain, judged against every spread available on disk.
    if "variant" in arms and "control" in arms:
        control, variant = arms["control"], arms["variant"]
        for seed in shared_seeds:
            delta = variant["kappa_by_seed"][seed] - control["kappa_by_seed"][seed]
            print(f"seed {seed}: variant - control = {delta:+.4f}")
        mean_delta = variant["kappa_mean"] - control["kappa_mean"]
        print(f"mean over shared seeds {shared_seeds}: variant - control = {mean_delta:+.4f}")
        print(f"control spread (max-min) = {control['kappa_spread_max_minus_min']:.4f}; "
              f"reference spread = {reference['kappa_spread_max_minus_min']:.4f}")
        inside_own = abs(mean_delta) <= control["kappa_spread_max_minus_min"]
        inside_reference = abs(mean_delta) <= reference["kappa_spread_max_minus_min"]
        print(f"|mean delta| <= control seed spread? {inside_own}")
        print(f"|mean delta| <= v0.7 reference seed spread? {inside_reference}")
        print(f"variant mean {variant['kappa_mean']:.4f} vs reference range "
              f"{reference['kappa_min']:.4f}-{reference['kappa_max']:.4f}")

    if args.out:
        payload = {"reference": reference, "arms": arms, "shared_seeds": shared_seeds}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
