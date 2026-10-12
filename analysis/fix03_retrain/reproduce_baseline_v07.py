"""Reproduce the published v0.7 baseline from the wave-1 artifacts, with the
current trainer code, and compute the NEW reachability diagnostics on it.

Everything here is measured by this lane from the named input files. Nothing is
restated from a document.

Inputs (read-only):
  /work/lane_runs/an02_errors/artifacts/v07_seed*_test_predictions.npz
  /work/lane_runs/an02_errors/artifacts/v07_seed*_calibration.json
  /work/lane_runs/an02_errors/artifacts/v07_seed*_metrics.published.json

Output: /work/lane_runs/fix03_retrain/analysis/baseline_v07_reproduction.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.metrics import evaluate_predictions
from train.model import coral_decode

ART = Path("/work/lane_runs/an02_errors/artifacts")
OUT = Path("/work/lane_runs/fix03_retrain/analysis")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 29, 43, 71, 101]


def per_level_recall(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    """Recall over ALL events of a true level (abstentions count as misses).

    This is the definition the wave-1 analysis used for its published per-level
    recall (L3 0.90, L4 0.50, L5 0.00, L6 0.00, L7 1.00): the denominator is
    every event with that true label, and an abstention is not a hit.
    """
    out: dict[str, float | int] = {}
    for index in range(5):
        mask = labels == index
        support = int(mask.sum())
        hits = int(((predictions == index) & mask).sum())
        out[f"L{index + 3}"] = {
            "support": support,
            "hits": hits,
            "recall": (hits / support) if support else float("nan"),
        }
    return out


def main() -> None:
    summary: dict[str, object] = {
        "inputs": {
            "artifacts_dir": str(ART),
            "seeds": SEEDS,
            "note": (
                "test_predictions.npz holds the CALIBRATED (temperature-scaled) CORAL "
                "probabilities, the zero-based labels and the event_ids of the 108-event "
                "v0.7 held-out test split."
            ),
        },
        "seeds": {},
    }

    for seed in SEEDS:
        arrays = np.load(ART / f"v07_seed{seed}_test_predictions.npz", allow_pickle=False)
        probabilities = arrays["probabilities"]
        labels = arrays["labels"]
        calibration = json.loads((ART / f"v07_seed{seed}_calibration.json").read_text())
        published = json.loads((ART / f"v07_seed{seed}_metrics.published.json").read_text())
        threshold = float(calibration["threshold"])

        # Recompute the shipped metrics with the current code path.
        recomputed = evaluate_predictions(probabilities, labels, threshold, bootstrap_samples=0)
        raw_predictions, confidence = coral_decode(probabilities)
        accepted = confidence >= threshold
        predictions = np.where(accepted, raw_predictions, -1)

        # argmax decode on the SAME probabilities, same threshold.
        argmax_predictions_raw = probabilities.argmax(axis=1)
        argmax_confidence = probabilities.max(axis=1)
        argmax_accepted = argmax_confidence >= threshold
        argmax_predictions = np.where(argmax_accepted, argmax_predictions_raw, -1)

        # The interior mass ceiling that bounds any class-mass threshold.
        from train.model import coral_interior_mass_ceiling

        ceiling = coral_interior_mass_ceiling(probabilities)

        summary["seeds"][str(seed)] = {
            "threshold": threshold,
            "interior_mass_ceiling": ceiling,
            "threshold_exceeds_ceiling": (ceiling is not None and threshold > ceiling + 1e-12),
            "published": {
                "macro_f1_with_abstentions": published["metrics"]["macro_f1_with_abstentions"],
                "weighted_kappa_covered": published["metrics"]["weighted_kappa_covered"],
                "coverage": published["metrics"]["coverage"],
                "dangerous_fnr_covered": published["metrics"][
                    "dangerous_underclassification_fnr_covered"
                ],
            },
            "recomputed": {
                "confusion_matrix": recomputed["confusion_matrix"],
                "macro_f1_with_abstentions": recomputed["metrics"]["macro_f1_with_abstentions"],
                "weighted_kappa_covered": recomputed["metrics"]["weighted_kappa_covered"],
                "coverage": recomputed["metrics"]["coverage"],
                "dangerous_fnr_covered": recomputed["metrics"][
                    "dangerous_underclassification_fnr_covered"
                ],
            },
            "diagnostics": {
                "decoded_level_counts_pre_abstention": recomputed[
                    "decoded_level_counts_pre_abstention"
                ],
                "interior_decodes_total": recomputed["interior_decodes_total"],
                "interior_decodes_accepted": recomputed["interior_decodes_accepted"],
                "interior_levels_reachable": recomputed["interior_levels_reachable"],
            },
            "rank_emitted_level_counts": {
                f"L{index + 3}": int((predictions == index).sum()) for index in range(5)
            },
            "per_level_recall_all_events_rank_decode": per_level_recall(labels, predictions),
            "per_level_recall_all_events_argmax_decode": per_level_recall(labels, argmax_predictions),
            "argmax_level_counts_accepted": {
                f"L{index + 3}": int((argmax_predictions == index).sum()) for index in range(5)
            },
            "argmax_coverage": float(argmax_accepted.mean()),
        }

    (OUT / "baseline_v07_reproduction.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # A compact console table.
    print(f"{'seed':>5} {'thr':>8} {'cap':>8} {'thr>cap':>7} {'int_tot':>7} {'int_acc':>7} "
          f"{'macroF1':>8} {'kappa':>7} {'cov':>6} {'fnr_cov':>8}")
    for seed in SEEDS:
        row = summary["seeds"][str(seed)]
        print(
            f"{seed:>5} {row['threshold']:>8.4f} {row['interior_mass_ceiling']:>8.4f} "
            f"{str(row['threshold_exceeds_ceiling']):>7} "
            f"{row['diagnostics']['interior_decodes_total']:>7} "
            f"{row['diagnostics']['interior_decodes_accepted']:>7} "
            f"{row['recomputed']['macro_f1_with_abstentions']:>8.4f} "
            f"{row['recomputed']['weighted_kappa_covered']:>7.4f} "
            f"{row['recomputed']['coverage']:>6.3f} "
            f"{row['recomputed']['dangerous_fnr_covered']:>8.4f}"
        )
    print()
    for seed in SEEDS:
        row = summary["seeds"][str(seed)]
        rank = row["per_level_recall_all_events_rank_decode"]
        arg = row["per_level_recall_all_events_argmax_decode"]
        print(
            f"seed {seed}: rank recall "
            + " ".join(f"{k}={v['recall']:.2f}({v['hits']}/{v['support']})" for k, v in rank.items())
            + " | argmax recall "
            + " ".join(f"{k}={v['recall']:.2f}({v['hits']}/{v['support']})" for k, v in arg.items())
            + f" | argmax emitted {row['argmax_level_counts_accepted']}"
        )


if __name__ == "__main__":
    main()
