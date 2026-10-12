#!/usr/bin/env python3
"""Why does the rank decode miss L5/L6?  Read the survival curve, not the recall.

The CORAL rank rule is ``level = #{k : P(y > k) > 0.5}``, so an L5 decode needs
``P(y > 3) <= 0.5 < P(y > 4)``.  That condition depends only on the *survival*
probabilities, not on the class mass, so it is the right place to look for the
cause of a missing interior level.  This script reads the stored validation
probabilities and reports, per true level:

* ``survival_P_gt_k`` mean and 10/50/90th percentiles -- whether the row's
  survival curve actually crosses 0.5 at the cut that would make it an L5/L6
  decode;
* ``interior_mass_ceiling`` from calibration.json -- the largest mass any
  interior class can carry, ``tanh(min_gap / 4)``.  A threshold above this
  abstains on every interior decode, which is the H-CORAL symptom arriving
  through the abstention gate instead of through argmax;
* how many rows of each true level satisfy the L5 and L6 decode conditions.

Everything is read from ``val_predictions.npz`` + ``calibration.json`` in the
run directory; nothing is recomputed from a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUNS = Path("/home/tun/iddsi/lane_runs/fix04_classbalance")
LEVELS = [3, 4, 5, 6, 7]
ARMS = [
    ("a_beta999", "fix04_a_beta999"),
    ("b_beta99999", "fix04_b_beta99999"),
    ("c_equalised", "fix04_c_equalised"),
    ("d_beta0_nobalance", "fix04_d_beta0_nobalance"),
]


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def survival_stats(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    survival = 1.0 - np.cumsum(probabilities, axis=1)[:, :-1]  # P(y > 0..3)
    out: dict[str, dict] = {}
    for level in range(5):
        mask = labels == level
        if not mask.any():
            continue
        row = {"n": int(mask.sum())}
        for k in range(4):
            column = survival[mask, k]
            row[f"P_gt_{k}_p10"] = _pct(column, 10)
            row[f"P_gt_{k}_p50"] = _pct(column, 50)
            row[f"P_gt_{k}_p90"] = _pct(column, 90)
            row[f"frac_P_gt_{k}_above_half"] = float((column > 0.5).mean())
        out[f"L{level + 3}"] = row
    return out


def decode_conditions(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """How many rows of each true level satisfy each level's decode condition."""
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    survival = 1.0 - np.cumsum(probabilities, axis=1)[:, :-1]
    out: dict[str, dict] = {}
    for level in range(5):
        mask = labels == level
        if not mask.any():
            continue
        # level j decodes iff P(y>j-1) <= 0.5 < P(y>j)
        below = survival[:, level - 1] if level >= 1 else np.zeros(len(labels))
        above = survival[:, level] if level <= 3 else np.ones(len(labels))
        condition = (below <= 0.5) & (above > 0.5)
        out[f"L{level + 3}"] = {
            "n": int(mask.sum()),
            "rows_meeting_own_decode_condition": int((condition & mask).sum()),
        }
    return out


def main() -> None:
    report: dict[str, dict] = {}
    for slug, name in ARMS:
        for run_dir in sorted((RUNS / slug).glob(f"{name}_seed*")):
            npz_path = run_dir / "val_predictions.npz"
            calib_path = run_dir / "calibration.json"
            if not npz_path.exists():
                print(f"WARNING missing {npz_path}", flush=True)
                continue
            with np.load(npz_path, allow_pickle=False) as data:
                probabilities = data["probabilities"]
                labels = data["labels"]
            calibration = json.loads(calib_path.read_text(encoding="utf-8"))
            key = f"{slug}/{run_dir.name}"
            report[key] = {
                "val_n": int(len(labels)),
                "val_level_support": {
                    f"L{level + 3}": int((labels == level).sum()) for level in range(5)
                },
                "temperature": calibration["temperature"],
                "threshold": calibration["threshold"],
                "interior_mass_ceiling": calibration.get("interior_mass_ceiling"),
                "threshold_above_interior_ceiling": (
                    calibration.get("interior_mass_ceiling") is not None
                    and calibration["threshold"] > calibration["interior_mass_ceiling"]
                ),
                "survival_stats": survival_stats(probabilities, labels),
                "decode_conditions": decode_conditions(probabilities, labels),
            }
    out_dir = RUNS / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "survival_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for key, entry in report.items():
        print(f"\n=== {key}  (val n={entry['val_n']}) ===")
        print(
            f"  temperature={entry['temperature']:.4f} threshold={entry['threshold']:.4f} "
            f"interior_mass_ceiling={entry['interior_mass_ceiling']} "
            f"threshold_above_ceiling={entry['threshold_above_interior_ceiling']}"
        )
        print("  true level | n | P(>3) p50 | frac P(>3)>0.5 | P(>4) p50 | frac P(>4)>0.5 | meets own decode")
        for level in ("L3", "L4", "L5", "L6", "L7"):
            stats = entry["survival_stats"].get(level)
            if stats is None:
                continue
            decode = entry["decode_conditions"][level]
            print(
                f"  {level} | {stats['n']:>3} | {stats['P_gt_3_p50']:.3f} | "
                f"{stats['frac_P_gt_3_above_half']:.3f} | {stats['P_gt_4_p50']:.3f} | "
                f"{stats['frac_P_gt_4_above_half']:.3f} | "
                f"{decode['rows_meeting_own_decode_condition']}/{decode['n']}"
            )
    print(f"\nwrote {out_dir / 'survival_analysis.json'}")


if __name__ == "__main__":
    main()
