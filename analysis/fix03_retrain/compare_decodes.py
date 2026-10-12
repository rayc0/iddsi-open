"""Compare the CORAL decode rules on IDENTICAL weights, on the v0.7 splits.

For every run directory given on the command line this re-selects the
abstention threshold on the VALIDATION split only (never on test) under each
decode rule, then evaluates the TEST split once with that frozen threshold.

Why this exists: the wave-1 analysis established that on the v0.7 held-out
split no seed emits L5 or L6 as a decision, and that argmax never tops L4 or L6
either. That is consistent with two different causes -- "the decode never
produced L5" and "every L5 decode fell below the threshold" -- which the
published confusion matrix cannot tell apart. `evaluate_predictions` now emits
`decoded_level_counts_pre_abstention`, `interior_decodes_total`,
`interior_decodes_accepted` and `interior_levels_reachable`, and
`calibrate_abstention_threshold` clips its candidates to the CORAL interior
mass ceiling so a threshold can no longer delete the interior levels from the
output. This script runs both and reports the difference.

Nothing here reads a sealed file. Test predictions are only ever evaluated with
a threshold that was frozen on validation data first.

Output: /work/lane_runs/fix03_retrain/analysis/decode_comparison.json (+ .md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from eval.calibration import calibrate_abstention_threshold
from eval.metrics import evaluate_predictions
from train.model import coral_decode, coral_interior_mass_ceiling

RULES = ("rank", "argmax")
ANALYSIS = Path("/work/lane_runs/fix03_retrain/analysis")


def _recall(labels: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Per-level recall over ALL events of that true level (abstention = miss)."""
    out: dict[str, dict[str, float | int]] = {}
    for index in range(5):
        support = int((labels == index).sum())
        hits = int(((predictions == index) & (labels == index)).sum())
        out[f"L{index + 3}"] = {
            "support": support,
            "hits": hits,
            "recall_all_events": (hits / support) if support else float("nan"),
        }
    return out


def _recall_decided(labels: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Per-level recall over DECIDED events only -- the wave-1 published basis."""
    out: dict[str, dict[str, float | int]] = {}
    for index in range(5):
        support = int(((labels == index) & (predictions != -1)).sum())
        hits = int(((predictions == index) & (labels == index)).sum())
        out[f"L{index + 3}"] = {
            "decided_support": support,
            "hits": hits,
            "recall_decided": (hits / support) if support else float("nan"),
        }
    return out


def macro_f1_abstention_as_class(labels: np.ndarray, predictions: np.ndarray) -> float:
    """H-MACROF1-corrected macro-F1: -1 is a sixth class with its own P/R/F1.

    The shipped `macro_f1_with_abstentions` builds its label set from the TRUE
    labels only, so an abstention is charged against its true class's recall and
    its prediction-side false positive is counted nowhere. Both variants are
    reported here so the comparison with the v0.7 baseline is like-for-like on
    the shipped definition and honest on the corrected one.
    """
    classes = sorted({-1, 0, 1, 2, 3, 4})
    return float(
        f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
    )


def evaluate_run(run_dir: Path) -> dict[str, object]:
    val = np.load(run_dir / "val_predictions.npz", allow_pickle=False)
    test = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    shipped_calibration = json.loads((run_dir / "calibration.json").read_text())

    out: dict[str, object] = {
        "run_dir": str(run_dir),
        "n_val": int(len(val["labels"])),
        "n_test": int(len(test["labels"])),
        "val_level_support": {
            f"L{index + 3}": int((val["labels"] == index).sum()) for index in range(5)
        },
        "test_level_support": {
            f"L{index + 3}": int((test["labels"] == index).sum()) for index in range(5)
        },
        "shipped_calibration": shipped_calibration,
        "rules": {},
    }

    for rule in RULES:
        # --- threshold selection: VALIDATION ONLY -----------------------------
        selection = calibrate_abstention_threshold(
            val["probabilities"], val["labels"],
            float(shipped_calibration["target_under_fnr"]),
            1.0 - float(shipped_calibration["minimum_coverage"]),
            rule,
        )
        threshold = float(selection["threshold"])

        # --- test evaluation with that frozen threshold -----------------------
        result = evaluate_predictions(
            test["probabilities"], test["labels"], threshold,
            bootstrap_samples=0, decode=rule,
        )
        raw, confidence = coral_decode(test["probabilities"], rule)
        accepted = confidence >= threshold
        predictions = np.where(accepted, raw, -1)
        labels = test["labels"]

        under = int(((predictions < labels) & (predictions != -1)).sum())
        under_all = int(((raw < labels) & accepted).sum())
        over = int(((predictions > labels) & (predictions != -1)).sum())

        out["rules"][rule] = {
            "threshold_selected_on_val": threshold,
            "val_selection": selection,
            "test": {
                "n": int(len(labels)),
                "macro_f1_with_abstentions_shipped_definition": result["metrics"][
                    "macro_f1_with_abstentions"
                ],
                "macro_f1_abstention_as_class_HMACROF1": macro_f1_abstention_as_class(
                    labels, predictions
                ),
                "weighted_kappa_covered": result["metrics"]["weighted_kappa_covered"],
                "dangerous_fnr_covered": result["metrics"][
                    "dangerous_underclassification_fnr_covered"
                ],
                "dangerous_underclassification_rate_all": result["metrics"][
                    "dangerous_underclassification_rate_all"
                ],
                "coverage": result["metrics"]["coverage"],
                "abstention_rate": result["metrics"]["abstention_rate"],
                "decided_n": int(accepted.sum()),
                "abstained_n": int((~accepted).sum()),
                "wrong_level_decided": int(((predictions != labels) & (predictions != -1)).sum()),
                "over_predicted": over,
                "under_predicted": under,
                "under_predicted_all_rows": under_all,
                "per_level_recall_all_events": _recall(labels, predictions),
                "per_level_recall_decided_only": _recall_decided(labels, predictions),
                "emitted_level_counts": {
                    f"L{index + 3}": int((predictions == index).sum()) for index in range(5)
                },
                "decoded_level_counts_pre_abstention": result[
                    "decoded_level_counts_pre_abstention"
                ],
                "interior_decodes_total": result["interior_decodes_total"],
                "interior_decodes_accepted": result["interior_decodes_accepted"],
                "interior_levels_reachable": result["interior_levels_reachable"],
                "confusion_matrix": result["confusion_matrix"],
            },
            "test_interior_mass_ceiling": coral_interior_mass_ceiling(test["probabilities"]),
            "val_interior_mass_ceiling": coral_interior_mass_ceiling(val["probabilities"]),
        }

    return out


def main() -> None:
    run_dirs = [Path(argument) for argument in sys.argv[1:]]
    if not run_dirs:
        raise SystemExit("usage: compare_decodes.py <run_dir> [<run_dir> ...]")
    payload = {"decode_rules": list(RULES), "runs": [evaluate_run(path) for path in run_dirs]}
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "decode_comparison.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    for run in payload["runs"]:
        print(f"\n=== {run['run_dir']} ===")
        print(
            f"val n={run['n_val']} support {run['val_level_support']} | "
            f"test n={run['n_test']} support {run['test_level_support']}"
        )
        for rule in RULES:
            row = run["rules"][rule]
            test = row["test"]
            print(
                f"  [{rule:>6}] thr={row['threshold_selected_on_val']:.4f} "
                f"cap={row['test_interior_mass_ceiling']:.4f} "
                f"int_tot={test['interior_decodes_total']} "
                f"int_acc={test['interior_decodes_accepted']} "
                f"reach={test['interior_levels_reachable']}"
            )
            print(
                f"            macroF1={test['macro_f1_with_abstentions_shipped_definition']:.4f} "
                f"macroF1(+abst)={test['macro_f1_abstention_as_class_HMACROF1']:.4f} "
                f"kappa={test['weighted_kappa_covered']:.4f} "
                f"cov={test['coverage']:.3f} "
                f"fnr_cov={test['dangerous_fnr_covered']:.4f}"
            )
            print(
                "            recall(all events) "
                + " ".join(
                    f"{key}={value['recall_all_events']:.2f}({value['hits']}/{value['support']})"
                    for key, value in test["per_level_recall_all_events"].items()
                )
                + " | emitted "
                + str(test["emitted_level_counts"])
            )
    print(f"\nwrote {ANALYSIS / 'decode_comparison.json'}")


if __name__ == "__main__":
    main()
