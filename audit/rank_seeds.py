"""Rank the five v0.7 seeds on the SHIPPED artifacts only (no re-inference).

Reads each seed's ``calibration.json`` (the validation-selected threshold and
temperature that were actually shipped) and ``test_predictions.npz`` (the
calibrated held-out test probabilities), then recomputes the released metric
block with the shipped code path so the ranking is measured here rather than
copied from the previous summary.

Ranking rule (declared, applied identically to every seed):
  1. primary   : highest macro_f1_with_abstentions on the held-out test split
  2. tie-break : higher coverage, then lower dangerous under-classification FNR
  3. reported  : calibration target must have been met on validation (a seed
                 whose gate was not met is flagged, never silently promoted)

Usage:
    python audit/rank_seeds.py --runs-root /work/runs/v07 --out /work/lane_runs/an03_calibration/seed_ranking.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pin_trainer import pin as pin_trainer  # noqa: E402  (must precede train/eval imports)

CODE_PROVENANCE = pin_trainer()

import numpy as np  # noqa: E402

from eval.metrics import evaluate_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    seeds = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        calibration_path = run_dir / "calibration.json"
        predictions_path = run_dir / "test_predictions.npz"
        if not (calibration_path.exists() and predictions_path.exists()):
            continue
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        arrays = np.load(predictions_path, allow_pickle=False)
        probabilities = arrays["probabilities"].astype(np.float64)
        labels = arrays["labels"].astype(np.int64)
        result = evaluate_predictions(
            probabilities, labels, float(calibration["threshold"]),
            ece_bins=10, bootstrap_samples=0, bootstrap_seed=2026,
        )
        seeds.append(
            {
                "seed_dir": run_dir.name,
                "n_test_events": int(result["n"]),
                "shipped_threshold": float(calibration["threshold"]),
                "shipped_temperature": float(calibration["temperature"]),
                "calibration_target_met": bool(calibration["target_met"]),
                "selection_reason": calibration["selection_reason"],
                "metrics": result["metrics"],
            }
        )

    def rank_key(entry):
        metrics = entry["metrics"]
        return (
            entry["calibration_target_met"],
            metrics["macro_f1_with_abstentions"],
            metrics["coverage"],
            -metrics["dangerous_underclassification_fnr_covered"],
        )

    seeds.sort(key=rank_key, reverse=True)
    for position, entry in enumerate(seeds, start=1):
        entry["rank"] = position
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "runs_root": str(runs_root),
                "code_provenance": CODE_PROVENANCE,
                "n_seeds": len(seeds),
                "ranking_rule": (
                    "macro_f1_with_abstentions on held-out test (desc), then coverage (desc), "
                    "then dangerous_underclassification_fnr_covered (asc); calibration_target_met first"
                ),
                "seeds": seeds,
                "best_seed": seeds[0]["seed_dir"] if seeds else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "best_seed": seeds[0]["seed_dir"] if seeds else None}, indent=2))


if __name__ == "__main__":
    main()
