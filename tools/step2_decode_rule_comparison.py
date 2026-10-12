"""Step 2: compare candidate decode rules on the v0.7 VALIDATION split only.

Protocol
--------
For each of the five released v0.7 seeds and each candidate rule:

1. **Out-of-fold (the headline number).**  The 110 validation events are split
   into 5 stratified folds.  For each fold the rule's decode parameters (the
   per-level cut-points, where it has any) AND its abstention threshold are
   fitted on the other four folds, then the held-out fold is decoded and gated.
   The five held-out folds are pooled, so every validation event is scored by a
   rule that never saw it.  This is what keeps the comparison honest: fitting
   four cut-points on 110 events with 6 L5 and 9 L6 would otherwise be fitting
   ~6 numbers with four free parameters.

2. **In-sample (reference only, optimistic).**  Fit on all 110 validation
   events, score on the same 110.  Reported so the size of the in-sample
   optimism is visible; never used to choose a rule.

3. **Test (frozen parameters, reported not tuned).**  Fit on all 110 validation
   events, score the 108 held-out test events.  The test split is never used to
   fit or to choose anything; it is reported because that is the split the
   published numbers are quoted on.

The abstention gate is identical for every rule: highest coverage whose
dangerous under-classification rate over the covered subset is <= 0.05, among
candidates covering >= 70%; if none qualifies, the least under-classification
among the eligible and ``target_met: false``.  For rules whose confidence is a
CORAL decoded-level mass the candidates are additionally clipped to the CORAL
interior mass ceiling ``tanh(gap / 4)``, which is the H-CORAL reachability
constraint already implemented in :func:`eval.calibration.calibrate_abstention_threshold`.

Outputs: ``analysis/step2_decode_rule_comparison.json`` and a markdown table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lane_common import DISCLAIMER, GATE, PRED_DIR, SEEDS, load_seed, shipped_calibration
from eval.decode_rules import (
    DEFAULT_MAX_ABSTENTION,
    DEFAULT_TARGET_UNDER_FNR,
    DecodeOutput,
    default_rules,
    level_counts,
    score_decode,
    select_abstention_threshold,
)
from eval.calibration import calibrate_abstention_threshold
from train.model import coral_decode, coral_interior_mass_ceiling

OUT = Path("/work/lane_runs/fix01_decode/analysis")
N_FOLDS = 5
CV_SEED = 20261012


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Deterministic stratified fold assignment (round-robin within each level)."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for level in np.unique(labels):
        indices = np.flatnonzero(labels == level)
        indices = indices[rng.permutation(len(indices))]
        for position, index in enumerate(indices):
            folds[position % n_folds].append(int(index))
    return [np.sort(np.asarray(fold, dtype=np.int64)) for fold in folds]


def fresh_rules() -> list:
    """A fresh rule instance per (rule, seed, protocol) — rules are stateful."""
    return default_rules()


def run_protocol(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rule,
    *,
    split: str,
    folds: list[np.ndarray] | None,
) -> dict:
    """Fit + gate either out-of-fold (``folds`` given) or on all of ``split``."""
    n = len(labels)
    if folds is None:
        fitted = rule.fit(probabilities, labels, split=split)
        output = fitted.decode(probabilities)
        ceiling = (
            coral_interior_mass_ceiling(probabilities)
            if fitted.confidence_is_coral_mass and (output.levels >= 1).any() and (output.levels <= 3).any()
            else None
        )
        selection = select_abstention_threshold(
            labels,
            output.levels,
            output.confidence,
            split=split,
            target_under_fnr=DEFAULT_TARGET_UNDER_FNR,
            max_abstention=DEFAULT_MAX_ABSTENTION,
            interior_mass_ceiling=ceiling,
        )
        scored = score_decode(labels, output, selection["threshold"], probabilities=probabilities, rule_name=rule.name)
        scored["_selection"] = selection
        scored["_fit_report"] = getattr(fitted, "fit_report", {})
        scored["_parameters"] = fitted.parameters()
        return scored

    levels = np.empty(n, dtype=np.int64)
    confidence = np.empty(n, dtype=np.float64)
    band_lo = np.empty(n, dtype=np.int64)
    band_hi = np.empty(n, dtype=np.int64)
    thresholds = np.empty(n, dtype=np.float64)
    per_fold = []
    for fold_index, held in enumerate(folds):
        train = np.setdiff1d(np.arange(n), held)
        fitted = rule.fit(probabilities[train], labels[train], split=split)
        train_output = fitted.decode(probabilities[train])
        ceiling = (
            coral_interior_mass_ceiling(probabilities[train])
            if fitted.confidence_is_coral_mass
            and ((train_output.levels >= 1) & (train_output.levels <= 3)).any()
            else None
        )
        selection = select_abstention_threshold(
            labels[train],
            train_output.levels,
            train_output.confidence,
            split=split,
            target_under_fnr=DEFAULT_TARGET_UNDER_FNR,
            max_abstention=DEFAULT_MAX_ABSTENTION,
            interior_mass_ceiling=ceiling,
        )
        held_output = fitted.decode(probabilities[held])
        levels[held] = held_output.levels
        confidence[held] = held_output.confidence
        band_lo[held] = held_output.band_lo
        band_hi[held] = held_output.band_hi
        thresholds[held] = selection["threshold"]
        per_fold.append(
            {
                "fold": fold_index,
                "n_held": int(len(held)),
                "n_train": int(len(train)),
                "held_support": level_counts(labels[held]),
                "parameters": fitted.parameters(),
                "threshold": float(selection["threshold"]),
                "target_met": bool(selection["target_met"]),
                "train_coverage_at_threshold": float(selection["coverage"]),
                "train_under_fnr_at_threshold": float(selection["under_fnr"]),
                "fit_report": getattr(fitted, "fit_report", {}),
            }
        )
    pooled = DecodeOutput(levels=levels, confidence=confidence, band_lo=band_lo, band_hi=band_hi, detail={})
    scored = score_decode(labels, pooled, thresholds, probabilities=probabilities, rule_name=rule.name)
    scored["_selection"] = {
        "protocol": "out_of_fold",
        "per_fold_thresholds": [float(v) for v in thresholds],
        "distinct_thresholds": sorted({float(v) for v in thresholds}),
    }
    scored["_fit_report"] = {"per_fold": per_fold}
    scored["_parameters"] = {"per_fold": [f["parameters"] for f in per_fold]}
    return scored


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "lane": "fix01_decode",
        "step": "2_decode_rule_comparison",
        "disclaimer": DISCLAIMER,
        "gate": {
            "target_under_fnr": DEFAULT_TARGET_UNDER_FNR,
            "max_abstention": DEFAULT_MAX_ABSTENTION,
            "shipped_gate": GATE,
        },
        "protocol": {
            "selection_split": "val",
            "n_folds": N_FOLDS,
            "cv_seed": CV_SEED,
            "out_of_fold": "decode parameters and abstention threshold fitted on the other folds",
            "in_sample": "fitted and scored on all 110 validation events (optimistic, reference only)",
            "test": "fitted on all 110 validation events, scored on the 108 held-out test events with frozen parameters",
        },
        "seeds": {},
    }

    for seed in SEEDS:
        val = load_seed(seed, "val")
        test = load_seed(seed, "test")
        probabilities, labels = val["probabilities"], val["labels"]
        folds = stratified_folds(labels, N_FOLDS, CV_SEED)
        support = {f"L{int(l) + 3}": int((labels == l).sum()) for l in range(5)}
        shipped = shipped_calibration(seed)

        entry = {
            "val_n": int(len(labels)),
            "val_support": support,
            "test_n": int(len(test["labels"])),
            "test_support": {f"L{int(l) + 3}": int((test["labels"] == l).sum()) for l in range(5)},
            "shipped_temperature": float(shipped["temperature"]),
            "shipped_threshold": float(shipped["threshold"]),
            "rules": {},
        }

        # Shipped reference: the repo's own selector on this lane's val dump.
        shipped_selection = calibrate_abstention_threshold(probabilities, labels, GATE["target_under_fnr"], GATE["max_abstention"])
        shipped_output = DecodeOutput(
            levels=coral_decode(probabilities)[0].astype(np.int64),
            confidence=coral_decode(probabilities)[1],
            band_lo=coral_decode(probabilities)[0].astype(np.int64),
            band_hi=coral_decode(probabilities)[0].astype(np.int64),
        )
        entry["rules"]["shipped_pipeline_reference"] = {
            "out_of_fold": None,
            "in_sample": score_decode(
                labels, shipped_output, shipped_selection["threshold"],
                probabilities=probabilities, rule_name="shipped_pipeline_reference",
            ),
            "test": None,
            "_selection": shipped_selection,
        }

        for rule in fresh_rules():
            entry["rules"][rule.name] = {
                "description": rule.description,
                "out_of_fold": run_protocol(probabilities, labels, rule, split="val", folds=folds),
                "in_sample": run_protocol(probabilities, labels, rule, split="val", folds=None),
                "test": run_protocol(test["probabilities"], test["labels"], rule, split="val", folds=None),
            }
        report["seeds"][str(seed)] = entry
        print(f"seed {seed}: {len(entry['rules'])} rules x 3 protocols done")

    # Pooled across seeds: the out-of-fold protocol repeated for each seed gives
    # 5 x 110 = 550 out-of-fold validation events, which is the only place the
    # L5 (6/seed) and L6 (9/seed) supports are large enough to read a rate off.
    pooled = {"description": "pool of the five per-seed out-of-fold validation protocols", "per_rule": {}}
    rule_names = [r.name for r in fresh_rules()]
    for name in rule_names:
        per_level = {f"L{l}": {"support": 0, "hit": 0, "predicted": 0} for l in range(3, 8)}
        counts = {"coverage_num": 0, "n": 0, "accepted": 0, "dangerous_covered": 0, "dangerous_total": 0}
        decoded_counts = {f"L{l}": 0 for l in range(3, 8)}
        decided_counts = {f"L{l}": 0 for l in range(3, 8)}
        for seed in SEEDS:
            scored = report["seeds"][str(seed)]["rules"][name]["out_of_fold"]
            counts["n"] += scored["n"]
            counts["accepted"] += scored["accepted"]
            counts["dangerous_covered"] += scored["dangerous_errors_covered"]
            counts["dangerous_total"] += scored["dangerous_errors_total"]
            for key, value in scored["per_level_recall"].items():
                per_level[key]["support"] += value["support"]
                per_level[key]["hit"] += int(value["recall_post_abstention_count"].split("/")[0])
                per_level[key]["predicted"] += value["predicted_post_abstention"]
            for key, value in scored["decoded_level_counts_pre_abstention"].items():
                decoded_counts[key] += value
            for key, value in scored["decided_level_counts_post_abstention"].items():
                decided_counts[key] += value
        pooled["per_rule"][name] = {
            "n": counts["n"],
            "coverage": counts["accepted"] / counts["n"],
            "accepted": counts["accepted"],
            "decoded_level_counts_pre_abstention": decoded_counts,
            "decided_level_counts_post_abstention": decided_counts,
            "per_level_recall_post_abstention": {
                key: {
                    "support": value["support"],
                    "correct": value["hit"],
                    "recall": (value["hit"] / value["support"]) if value["support"] else None,
                    "predicted": value["predicted"],
                }
                for key, value in per_level.items()
            },
            "macro_recall_post_abstention": float(
                np.mean([v["hit"] / v["support"] for v in per_level.values() if v["support"]])
            ),
            "dangerous_underclassification_fnr_covered": (
                counts["dangerous_covered"] / counts["accepted"] if counts["accepted"] else None
            ),
            "dangerous_errors_total": counts["dangerous_total"],
            "dangerous_errors_covered": counts["dangerous_covered"],
            "dangerous_fnr_detector": (
                counts["dangerous_covered"] / counts["dangerous_total"] if counts["dangerous_total"] else None
            ),
        }
    report["pooled_out_of_fold_validation"] = pooled

    (OUT / "step2_decode_rule_comparison.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / 'step2_decode_rule_comparison.json'}")


if __name__ == "__main__":
    main()
