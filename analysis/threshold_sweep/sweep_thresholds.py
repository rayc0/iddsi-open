"""Abstention-threshold sweep on the VALIDATION split only.

Question this answers
---------------------
Wave 1 measured that on the v0.7 held-out TEST split no seed ever emits L5 or
L6 as a decision (per-level recall L5 0.00, L6 0.00,
``analysis/errors_v07/outputs/structural_seed43.json``).  Two mechanisms can
produce that symptom and the shipped confusion matrix cannot tell them apart:

  (a) the CORAL rank decode never PRODUCES L5/L6 (a representation failure), or
  (b) the decode produces them but every such decode falls below the
      confidence threshold (an abstention-gate failure).

``trainer/eval/metrics.py`` now separates the two via
``decoded_level_counts_pre_abstention`` and ``interior_decodes_total/accepted``.
This script runs the shipped decode over EVERY threshold that is achievable on
the validation split and reports, per threshold:

  * coverage (accepted / n_val), with n_val stated
  * per-level recall on covered rows (and the covered support behind each rate)
  * dangerous-direction FNR under all three definitions in the repo, each with
    its own denominator
  * ECE (all rows, and covered rows)
  * the pre-abstention decode census and the per-level accepted census, so
    "the decode never produced L5" is distinguishable from "every L5 decode was
    abstained on"

Hard constraint honoured here: the threshold grid and every number are computed
on the VALIDATION split (n=110, split field ``val`` in
``/work/data/manifest_v07_weighted.jsonl``).  The test split is never loaded by
this script — there is no code path in it that opens a test manifest row.

The threshold grid is EXHAUSTIVE, not a sample: acceptance is
``confidence >= threshold``, so every threshold between two consecutive
distinct confidence values yields an identical acceptance set.  The grid is
therefore ``{0.0} u {distinct confidences} u {nextafter(max, inf)}`` — the same
candidate set ``eval.calibration.calibrate_abstention_threshold`` builds.  A
dense linear grid is also emitted for readers who want round numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LANE_ROOT = Path(__file__).resolve().parents[2]
TRAINER = LANE_ROOT / "trainer"
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from eval.calibration import calibrate_abstention_threshold  # noqa: E402
from eval.metrics import expected_calibration_error  # noqa: E402
from train.model import coral_decode, coral_interior_mass_ceiling  # noqa: E402
from train.utils import write_json  # noqa: E402

SEEDS = (17, 29, 43, 71, 101)
LEVELS = np.array([3, 4, 5, 6, 7], dtype=np.int64)
LINEAR_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_level_recall(labels: np.ndarray, predictions: np.ndarray, accepted: np.ndarray) -> dict:
    """Per-level recall over COVERED rows, with the denominator behind each rate.

    A level whose true rows are all abstained on has recall ``nan`` and
    ``covered_support == 0``; that is reported rather than shown as 0.0, so a
    level is never called "unrecalled" when the model simply declined to answer.
    """
    out: dict[str, dict] = {}
    for index in range(5):
        level = f"L{int(LEVELS[index])}"
        support = int((labels == index).sum())
        covered_support = int(((labels == index) & accepted).sum())
        hits = int(((labels == index) & accepted & (predictions == index)).sum())
        out[level] = {
            "support": support,
            "covered_support": covered_support,
            "hits": hits,
            "recall_covered": (hits / covered_support) if covered_support else float("nan"),
            "recall_all_rows": hits / support if support else float("nan"),
        }
    return out


def _row(
    threshold: float,
    confidence: np.ndarray,
    raw_predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    ceiling: float | None,
) -> dict:
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    covered_true = labels[accepted]
    covered_pred = raw_predictions[accepted]
    if accepted_count:
        dangerous_covered = int((covered_pred < covered_true).sum())
        fnr_covered = dangerous_covered / accepted_count
        rate_all = dangerous_covered / len(labels)
        accuracy = float((covered_pred == covered_true).mean())
    else:
        dangerous_covered, fnr_covered, rate_all, accuracy = 0, float("nan"), float("nan"), float("nan")
    accepted_census = {f"L{int(LEVELS[i])}": int(((raw_predictions == i) & accepted).sum()) for i in range(5)}
    interior = (raw_predictions >= 1) & (raw_predictions <= 3)
    return {
        "threshold": float(threshold),
        "coverage": accepted_count / len(labels),
        "accepted": accepted_count,
        "abstained": len(labels) - accepted_count,
        "n": int(len(labels)),
        "accuracy_covered": accuracy,
        "per_level_recall_covered": per_level_recall(labels, raw_predictions, accepted),
        "emitted_level_counts": {k: v for k, v in accepted_census.items() if v},
        "levels_emitted_as_decisions": sorted(k for k, v in accepted_census.items() if v),
        "L5_or_L6_reachable": bool(accepted_census["L5"] > 0 or accepted_census["L6"] > 0),
        "interior_decodes_total": int(interior.sum()),
        "interior_decodes_accepted": int((interior & accepted).sum()),
        "interior_levels_reachable": bool((interior & accepted).any()),
        "dangerous_errors_total": int((raw_predictions < labels).sum()),
        "dangerous_errors_accepted": dangerous_covered,
        "dangerous_errors_caught_by_abstention": int((raw_predictions < labels).sum()) - dangerous_covered,
        # (1) abstention-as-detector FNR: accepted dangerous / all dangerous
        "dangerous_fnr_abstention_detector": (
            dangerous_covered / int((raw_predictions < labels).sum())
            if int((raw_predictions < labels).sum()) else float("nan")
        ),
        # (2) the shipped eval.metrics name: dangerous over covered rows
        "dangerous_underclassification_fnr_covered": fnr_covered,
        # (3) all-rows rate: dangerous accepted / all N
        "dangerous_underclassification_rate_all": rate_all,
        "ece_all": expected_calibration_error(probabilities, labels, 10),
        "ece_covered": (
            expected_calibration_error(probabilities, labels, 10, accepted) if accepted_count else float("nan")
        ),
        "decoded_level_counts_pre_abstention": {
            f"L{int(LEVELS[i])}": int((raw_predictions == i).sum()) for i in range(5)
        },
        "exceeds_interior_mass_ceiling": bool(ceiling is not None and float(threshold) > float(ceiling) + 1e-12),
    }


def _emitted_composition(
    threshold: float,
    confidence: np.ndarray,
    raw_predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """True-level composition of the accepted L5 and L6 decisions at a threshold.

    Split into the dangerous direction (true level ABOVE the prediction, i.e.
    ``y_pred < y_true``: predicting a SAFER texture than truth) and the
    over-prediction direction, so "the level is reachable" can be read against
    "the level is reachable in which direction".
    """
    accepted = confidence >= threshold
    out: dict[str, dict] = {}
    for level in (2, 3):
        mask = (raw_predictions == level) & accepted
        true_levels = labels[mask] + 3
        counts = {f"true_L{3 + index}": int((true_levels == 3 + index).sum()) for index in range(5)}
        under = int((true_levels > level + 3).sum())
        over = int((true_levels < level + 3).sum())
        out[f"L{level + 3}"] = {
            "n_emitted": int(mask.sum()),
            "true_level_counts": counts,
            "dangerous_direction_under": under,
            "over_direction": over,
            "correct": int((true_levels == level + 3).sum()),
        }
    return out


def sweep_one(probabilities: np.ndarray, labels: np.ndarray, seed: int, shipped_threshold: float) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    labels = np.asarray(labels, dtype=np.int64)
    n = int(len(labels))
    raw_predictions, confidence = coral_decode(probabilities)
    dangerous_total = int((raw_predictions < labels).sum())
    ceiling = coral_interior_mass_ceiling(probabilities)

    # Exhaustive candidate grid (see module docstring) plus a round-number grid,
    # plus the shipped threshold itself so the shipped operating point is
    # reported exactly rather than at the nearest grid point.
    distinct = np.unique(confidence)
    exhaustive = np.unique(
        np.concatenate(
            ([0.0], distinct, [np.nextafter(float(distinct.max()), np.inf)], [float(shipped_threshold)])
        )
    )
    exhaustive_set = {float(t) for t in exhaustive}
    linear_set = {float(t) for t in LINEAR_GRID}
    union = np.unique(np.concatenate((exhaustive, LINEAR_GRID)))

    all_rows = [
        _row(float(t), confidence, raw_predictions, labels, probabilities, ceiling) for t in union
    ]
    rows = [row for row in all_rows if row["threshold"] in exhaustive_set]
    linear_rows = [row for row in all_rows if row["threshold"] in linear_set]

    shipped_row = min(rows, key=lambda row: abs(row["threshold"] - float(shipped_threshold)))
    exact = [row for row in rows if row["threshold"] == float(shipped_threshold)]
    reachable_rows = [row for row in rows if row["L5_or_L6_reachable"]]
    l5_rows = [row for row in rows if row["emitted_level_counts"].get("L5", 0) > 0]
    l6_rows = [row for row in rows if row["emitted_level_counts"].get("L6", 0) > 0]
    # Among thresholds that emit L5 or L6, the cheapest in covered FNR, and the
    # most conservative one that still satisfies the 0.05 under-classification gate.
    emitting = [row for row in reachable_rows]
    finite_emitting = [row for row in emitting if row["dangerous_underclassification_fnr_covered"] == row["dangerous_underclassification_fnr_covered"]]
    cheapest = (
        min(finite_emitting, key=lambda row: row["dangerous_underclassification_fnr_covered"])
        if finite_emitting else None
    )
    gate_meeting = [row for row in emitting if row["dangerous_underclassification_fnr_covered"] <= 0.05]
    gate_meeting = sorted(gate_meeting, key=lambda row: (-row["coverage"], row["threshold"]))

    interior_confidences = confidence[(raw_predictions >= 1) & (raw_predictions <= 3)]
    both_rows = [
        row for row in rows
        if row["emitted_level_counts"].get("L5", 0) > 0 and row["emitted_level_counts"].get("L6", 0) > 0
    ]
    boundary_threshold = max(both_rows, key=lambda item: item["threshold"])["threshold"] if both_rows else None
    return {
        "seed": int(seed),
        "split": "val",
        "n_val": n,
        "val_level_support": {f"L{int(LEVELS[i])}": int((labels == i).sum()) for i in range(5)},
        "val_L5_support": int((labels == 2).sum()),
        "val_L6_support": int((labels == 3).sum()),
        "decoded_level_counts_pre_abstention": {
            f"L{int(LEVELS[i])}": int((raw_predictions == i).sum()) for i in range(5)
        },
        "interior_decodes_total": int(interior_confidences.size),
        "interior_mass_ceiling": (None if ceiling is None else float(ceiling)),
        "interior_confidence_min": (float(interior_confidences.min()) if interior_confidences.size else None),
        "interior_confidence_max": (float(interior_confidences.max()) if interior_confidences.size else None),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "n_distinct_confidences": int(distinct.size),
        "n_candidate_thresholds": int(len(rows)),
        "shipped_threshold": float(shipped_threshold),
        "shipped_threshold_row": shipped_row,
        "reachability_boundary_threshold": (None if boundary_threshold is None else float(boundary_threshold)),
        "emitted_L5_L6_composition_at_boundary": (
            None if boundary_threshold is None
            else _emitted_composition(float(boundary_threshold), confidence, raw_predictions, labels)
        ),
        "emitted_L5_L6_composition_at_threshold_zero": _emitted_composition(
            0.0, confidence, raw_predictions, labels
        ),
        "rows": rows,
        "linear_grid_rows": linear_rows,
        "answer": {
            "any_threshold_makes_L5_reachable": bool(l5_rows),
            "any_threshold_makes_L6_reachable": bool(l6_rows),
            "any_threshold_makes_L5_or_L6_reachable": bool(reachable_rows),
            "n_thresholds_with_L5_or_L6": len(reachable_rows),
            "max_coverage_with_L5_or_L6": (
                max(row["coverage"] for row in reachable_rows) if reachable_rows else None
            ),
            "max_coverage_with_L5": (max(row["coverage"] for row in l5_rows) if l5_rows else None),
            "max_coverage_with_L6": (max(row["coverage"] for row in l6_rows) if l6_rows else None),
            "coverage_at_shipped_threshold": shipped_row["coverage"],
            "shipped_threshold_is_a_grid_point": bool(exact),
            "cheapest_fnr_covered_among_L5_or_L6_emitting": cheapest,
            "n_L5_or_L6_emitting_thresholds_meeting_the_0_05_gate": len(gate_meeting),
            "most_conservative_L5_or_L6_emitting_threshold_meeting_the_0_05_gate": (
                gate_meeting[0] if gate_meeting else None
            ),
            "threshold_at_max_coverage_with_L5_or_L6": (
                max(reachable_rows, key=lambda row: row["coverage"])["threshold"] if reachable_rows else None
            ),
        },
    }


def flatten_row(seed: int, row: dict) -> dict:
    recall = row["per_level_recall_covered"]
    flat = {
        "seed": seed,
        "threshold": row["threshold"],
        "coverage": row["coverage"],
        "accepted": row["accepted"],
        "abstained": row["abstained"],
        "n": row["n"],
        "accuracy_covered": row["accuracy_covered"],
        "ece_all": row["ece_all"],
        "ece_covered": row["ece_covered"],
        "dangerous_errors_total": row["dangerous_errors_total"],
        "dangerous_errors_accepted": row["dangerous_errors_accepted"],
        "dangerous_fnr_abstention_detector": row["dangerous_fnr_abstention_detector"],
        "dangerous_underclassification_fnr_covered": row["dangerous_underclassification_fnr_covered"],
        "dangerous_underclassification_rate_all": row["dangerous_underclassification_rate_all"],
        "interior_decodes_total": row["interior_decodes_total"],
        "interior_decodes_accepted": row["interior_decodes_accepted"],
        "interior_levels_reachable": row["interior_levels_reachable"],
        "L5_or_L6_reachable": row["L5_or_L6_reachable"],
        "levels_emitted_as_decisions": "|".join(row["levels_emitted_as_decisions"]),
        "exceeds_interior_mass_ceiling": row["exceeds_interior_mass_ceiling"],
    }
    for level in ("L3", "L4", "L5", "L6", "L7"):
        stats = recall[level]
        flat[f"{level}_support"] = stats["support"]
        flat[f"{level}_covered_support"] = stats["covered_support"]
        flat[f"{level}_hits"] = stats["hits"]
        flat[f"{level}_recall_covered"] = stats["recall_covered"]
        flat[f"{level}_recall_all_rows"] = stats["recall_all_rows"]
        flat[f"{level}_emitted"] = row["emitted_level_counts"].get(level, 0)
        flat[f"{level}_decoded_pre_abstention"] = row["decoded_level_counts_pre_abstention"][level]
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="/work/lane_runs/fix02_threshold")
    parser.add_argument("--out-dir", default="/work/lane_runs/fix02_threshold")
    parser.add_argument("--suffix", default="", help="suffix of the input npz files, e.g. '_cpu'")
    parser.add_argument("--tag", default="val", help="tag for the output file names")
    args = parser.parse_args()

    import eval.metrics as _em
    import eval.calibration as _ec
    import train.model as _tm
    provenance = {
        "lane": "fix02_threshold",
        "trainer_dir": str(TRAINER),
        "eval.metrics": f"{_em.__file__} sha256={sha256(Path(_em.__file__))}",
        "eval.calibration": f"{_ec.__file__} sha256={sha256(Path(_ec.__file__))}",
        "train.model": f"{_tm.__file__} sha256={sha256(Path(_tm.__file__))}",
        "manifest": "/work/data/manifest_v07_weighted.jsonl (split field == 'val' only)",
        "note": "sys.path pinned to this lane's trainer clone; the container's editable install "
                "of another lane's clone is shadowed. TEST rows are never loaded.",
    }

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, csv_rows = [], []
    for seed in SEEDS:
        npz_path = in_dir / f"val_predictions_seed{seed}{args.suffix}.npz"
        archive = np.load(npz_path, allow_pickle=True)
        result = sweep_one(
            archive["probabilities"], archive["labels"], int(archive["seed"]), float(archive["shipped_threshold"])
        )
        result["npz"] = str(npz_path)
        result["npz_sha256"] = sha256(npz_path)
        # Re-select the shipped threshold from these probabilities as a check that
        # the sweep grid contains the shipped operating point.
        result["reselect_check"] = calibrate_abstention_threshold(
            np.asarray(archive["probabilities"], dtype=np.float64),
            np.asarray(archive["labels"], dtype=np.int64),
            0.05, 0.30,
        )
        results.append(result)
        csv_rows.extend(flatten_row(seed, row) for row in result["linear_grid_rows"])
        print(
            f"seed {seed}: n_val={result['n_val']} L5 support={result['val_L5_support']} "
            f"L6 support={result['val_L6_support']} decoded={result['decoded_level_counts_pre_abstention']} "
            f"any_L5_or_L6={result['answer']['any_threshold_makes_L5_or_L6_reachable']}",
            flush=True,
        )

    write_json(out_dir / f"threshold_sweep_{args.tag}.json", {"provenance": provenance, "seeds": results})
    fieldnames = list(csv_rows[0].keys())
    with (out_dir / f"threshold_sweep_{args.tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(
        out_dir / f"threshold_sweep_{args.tag}_answer.json",
        {
            "provenance": provenance,
            "seeds": [
                {
                    "seed": r["seed"],
                    "n_val": r["n_val"],
                    "val_level_support": r["val_level_support"],
                    "val_L5_support": r["val_L5_support"],
                    "val_L6_support": r["val_L6_support"],
                    "decoded_level_counts_pre_abstention": r["decoded_level_counts_pre_abstention"],
                    "interior_decodes_total": r["interior_decodes_total"],
                    "interior_mass_ceiling": r["interior_mass_ceiling"],
                    "interior_confidence_min": r["interior_confidence_min"],
                    "interior_confidence_max": r["interior_confidence_max"],
                    "shipped_threshold": r["shipped_threshold"],
                    "shipped_threshold_row": r["shipped_threshold_row"],
                    "reachability_boundary_threshold": r["reachability_boundary_threshold"],
                    "emitted_L5_L6_composition_at_boundary": r["emitted_L5_L6_composition_at_boundary"],
                    "emitted_L5_L6_composition_at_threshold_zero": r[
                        "emitted_L5_L6_composition_at_threshold_zero"
                    ],
                    "answer": r["answer"],
                }
                for r in results
            ],
        },
    )
    print(json.dumps({r["seed"]: r["answer"] for r in results}, indent=2, default=str))


if __name__ == "__main__":
    main()
