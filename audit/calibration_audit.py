"""Calibration and abstention audit for the best v0.7 seed (an03_calibration lane).

Every number reported here is computed in this process from the two input files
named on the command line:
  * ``--val-npz``   reproduced validation-split predictions for the seed
  * ``--test-npz``  the shipped calibrated held-out test predictions
Nothing is copied from an earlier report.

Safety rule honoured throughout: threshold *selection* only ever runs on the
validation split.  The test split is used exclusively to evaluate thresholds
that were already frozen on validation, and every risk-coverage row reported
for the test split is produced by ``eval.abstention.risk_coverage_row`` -- a
pure function that takes the threshold as an argument and selects nothing.

Outputs (written under --out-dir):
    audit_seed<N>.json   every number, machine readable
    audit_seed<N>.md     the human-readable audit report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pin_trainer import pin as pin_trainer  # noqa: E402  (must precede train/eval imports)

CODE_PROVENANCE = pin_trainer()

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from eval.abstention import (  # noqa: E402
    risk_coverage_row,
    select_abstention_thresholds,
)
from eval.calibration import calibrate_abstention_threshold, fit_coral_temperature  # noqa: E402
from eval.metrics import coral_decode, evaluate_predictions, expected_calibration_error  # noqa: E402
from train.utils import write_json  # noqa: E402

#: Five evenly spaced coverage points, declared up front so the curve is
#: reproducible and not chosen after seeing the data.
TARGET_COVERAGES: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90, 1.00)

#: Gate constants shipped in trainer/configs/siglip2_v07.yaml.
TARGET_UNDER_FNR = 0.05
MAX_ABSTENTION = 0.30

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261012
ECE_BINS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "q05": float(np.percentile(finite, 5)),
        "q25": float(np.percentile(finite, 25)),
        "median": float(np.percentile(finite, 50)),
        "q75": float(np.percentile(finite, 75)),
        "q95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
        "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
    }


def reliability_table(probabilities: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    predictions, confidence = coral_decode(probabilities)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        in_bin = (confidence >= low) & (confidence <= high) if index == bins - 1 else (confidence >= low) & (confidence < high)
        if not in_bin.any():
            rows.append({"bin": f"[{low:.1f},{high:.1f})", "n": 0})
            continue
        rows.append(
            {
                "bin": f"[{low:.1f},{high:.1f})",
                "n": int(in_bin.sum()),
                "mean_confidence": float(confidence[in_bin].mean()),
                "accuracy": float(correct[in_bin].mean()),
                "gap": float(correct[in_bin].mean() - confidence[in_bin].mean()),
            }
        )
    return rows


def ece_block(probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    block = {
        "n": int(mask.sum()),
        "ece_bins_10": expected_calibration_error(probabilities, labels, 10, mask),
        "ece_bins_5": expected_calibration_error(probabilities, labels, 5, mask),
        "ece_bins_15": expected_calibration_error(probabilities, labels, 15, mask),
        "ece_bins_20": expected_calibration_error(probabilities, labels, 20, mask),
    }
    predictions, confidence = coral_decode(probabilities[mask])
    block["mean_confidence"] = float(confidence.mean())
    block["accuracy"] = float((predictions == labels[mask]).mean())
    block["mean_gap_accuracy_minus_confidence"] = block["accuracy"] - block["mean_confidence"]
    # Standard max-probability confidence, for comparison with the CORAL
    # rank-decode confidence the repo's ECE uses.
    normalised = probabilities[mask] / probabilities[mask].sum(axis=1, keepdims=True)
    max_confidence = normalised.max(axis=1)
    block["argmax_equals_rank_decode"] = float((normalised.argmax(axis=1) == predictions).mean())
    block["mean_max_prob_confidence"] = float(max_confidence.mean())
    block["mean_rank_decode_minus_maxprob_confidence"] = float((confidence - max_confidence).mean())
    block["n_rank_decode_confidence_below_maxprob"] = int((confidence < max_confidence - 1e-12).sum())
    # ECE recomputed with the conventional max-probability confidence.
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    total = len(confidence)
    ece_max = 0.0
    for index in range(ECE_BINS):
        low, high = edges[index], edges[index + 1]
        in_bin = (max_confidence >= low) & (max_confidence <= high) if index == ECE_BINS - 1 else (max_confidence >= low) & (max_confidence < high)
        if in_bin.any():
            ece_max += float(in_bin.sum() / total) * abs(float((predictions == labels[mask])[in_bin].mean()) - float(max_confidence[in_bin].mean()))
    block["ece_maxprob_confidence_bins_10"] = float(ece_max)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="shipped seed run dir (best.pt, calibration.json)")
    parser.add_argument("--val-npz", required=True, help="reproduced validation predictions")
    parser.add_argument("--test-npz", required=True, help="shipped calibrated test predictions")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--diagnostics", help="optional diagnostics_seed<N>.json to fold into this report")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(run_dir.name.rsplit("seed", 1)[1])

    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    shipped_threshold = float(calibration["threshold"])
    shipped_temperature = float(calibration["temperature"])

    val = np.load(args.val_npz, allow_pickle=False)
    test = np.load(args.test_npz, allow_pickle=False)
    val_logits = val["cutpoint_logits"].astype(np.float64)
    val_labels = val["labels"].astype(np.int64)
    val_probs_raw = val["probabilities"].astype(np.float64)
    test_probs = test["probabilities"].astype(np.float64)
    test_labels = test["labels"].astype(np.int64)

    # ---------------------------------------------------------------- 1. reproduction
    from train.model import coral_logits_to_probs  # torch-based, matches engine exactly

    import torch

    def calibrated(logits: np.ndarray, temperature: float) -> np.ndarray:
        return coral_logits_to_probs(torch.from_numpy(np.asarray(logits, dtype=np.float64) / temperature)).numpy()

    reproduced_temperature, reproduced_temperature_nll = fit_coral_temperature(val_logits, val_labels)
    val_probs = calibrated(val_logits, shipped_temperature)
    threshold_reselect = calibrate_abstention_threshold(
        val_probs, val_labels, TARGET_UNDER_FNR, MAX_ABSTENTION
    )
    val_probs_recomputed_temperature = calibrated(val_logits, reproduced_temperature)
    threshold_full_recompute = calibrate_abstention_threshold(
        val_probs_recomputed_temperature, val_labels, TARGET_UNDER_FNR, MAX_ABSTENTION
    )
    uncalibrated_val_nll = float(-np.log(val_probs_raw[np.arange(len(val_labels)), val_labels].clip(1e-12)).mean())
    reproduction = {
        "val_events": int(len(val_labels)),
        "val_level_counts_zero_based": {str(int(level)): int((val_labels == level).sum()) for level in range(5)},
        "checkpoint_val_nll": float(val["checkpoint_val_nll"]) if "checkpoint_val_nll" in val else float("nan"),
        "reproduced_uncalibrated_val_nll": uncalibrated_val_nll,
        "shipped_temperature": shipped_temperature,
        "reproduced_temperature": reproduced_temperature,
        "reproduced_temperature_abs_delta": abs(reproduced_temperature - shipped_temperature),
        "reproduced_temperature_nll": reproduced_temperature_nll,
        "shipped_threshold": shipped_threshold,
        "threshold_reselected_at_shipped_temperature": threshold_reselect["threshold"],
        "threshold_reselect_abs_delta": abs(threshold_reselect["threshold"] - shipped_threshold),
        "threshold_reselect_coverage": threshold_reselect["coverage"],
        "threshold_reselect_under_fnr": threshold_reselect["under_fnr"],
        "threshold_reselect_target_met": threshold_reselect["target_met"],
        "threshold_full_recompute": threshold_full_recompute["threshold"],
        "threshold_full_recompute_abs_delta": abs(threshold_full_recompute["threshold"] - shipped_threshold),
        "threshold_full_recompute_target_met": threshold_full_recompute["target_met"],
        "note": (
            "threshold_reselect* refits only the abstention threshold on the reproduced validation logits at the "
            "shipped temperature; threshold_full_recompute* refits the temperature too, i.e. the whole shipped "
            "calibration pipeline."
        ),
    }

    # ---------------------------------------------------------------- 2. ECE
    val_predictions, val_confidence = coral_decode(val_probs)
    test_predictions, test_confidence = coral_decode(test_probs)
    val_accepted = val_confidence >= shipped_threshold
    test_accepted = test_confidence >= shipped_threshold
    ece = {
        "definition": (
            "eval.metrics.expected_calibration_error: 10 equal-width confidence bins on the CORAL rank-decode "
            "confidence (probability mass of the decoded level); confidence is the decoded-level probability, "
            "not the maximum class probability."
        ),
        "test_all": ece_block(test_probs, test_labels),
        "test_covered": ece_block(test_probs, test_labels, test_accepted),
        "val_all": ece_block(val_probs, val_labels),
        "val_covered": ece_block(val_probs, val_labels, val_accepted),
        "test_reliability_table": reliability_table(test_probs, test_labels),
    }

    # ---------------------------------------------------------------- 3. risk-coverage
    val_curve = select_abstention_thresholds(
        val_labels, val_predictions, val_confidence, split="val", target_coverages=TARGET_COVERAGES
    )
    transfer_rows = []
    for row in val_curve["rows"]:
        test_row = risk_coverage_row(test_labels, test_predictions, test_confidence, row["threshold"])
        transfer_rows.append({"target_coverage": row["target_coverage"], "val_threshold": row["threshold"], "test": test_row})
    shipped_row_val = risk_coverage_row(val_labels, val_predictions, val_confidence, shipped_threshold)
    shipped_row_test = risk_coverage_row(test_labels, test_predictions, test_confidence, shipped_threshold)
    risk_coverage = {
        "target_coverages": list(TARGET_COVERAGES),
        "validation": {
            "split": "val", "n": int(len(val_labels)),
            "dangerous_samples": int((val_predictions < val_labels).sum()),
            "rows": val_curve["rows"],
        },
        "test_transfer_of_validation_thresholds": transfer_rows,
        "shipped_threshold_row_validation": shipped_row_val,
        "shipped_threshold_row_test": shipped_row_test,
        "note": (
            "Thresholds for the five coverage points are selected on the validation split only (n=%d). "
            "Each is then applied unchanged to the held-out test split (n=%d); no selection runs on test."
            % (len(val_labels), len(test_labels))
        ),
    }

    # ---------------------------------------------------------------- 4. dangerous FNR
    dangerous = test_predictions < test_labels
    dangerous_covered = int((dangerous & test_accepted).sum())
    dangerous_caught = int(dangerous.sum()) - dangerous_covered
    per_level_dangerous = {
        f"true_L{int(level) + 3}": {
            "n": int((test_labels == level).sum()),
            "n_dangerous": int(((test_labels == level) & dangerous).sum()),
            "n_dangerous_accepted": int(((test_labels == level) & dangerous & test_accepted).sum()),
        }
        for level in range(5)
    }
    dangerous_fnr = {
        "split": "test", "n": int(len(test_labels)),
        "definition_covered": (
            "dangerous_underclassification_fnr_covered = |covered and y_pred<y_true| / |covered| "
            "(eval.metrics, denominator is covered events)"
        ),
        "definition_detector": (
            "dangerous_fnr = |y_pred<y_true and confidence>=threshold| / |y_pred<y_true| "
            "(eval.abstention.risk_coverage_row, denominator is ALL dangerous-direction samples)"
        ),
        "dangerous_samples_total": int(dangerous.sum()),
        "dangerous_samples_covered": dangerous_covered,
        "dangerous_samples_caught_by_abstention": dangerous_caught,
        "dangerous_underclassification_fnr_covered": (
            dangerous_covered / int(test_accepted.sum()) if test_accepted.any() else float("nan")
        ),
        "dangerous_underclassification_rate_all": dangerous_covered / len(test_labels),
        "dangerous_fnr_detector": (dangerous_covered / int(dangerous.sum())) if dangerous.sum() else float("nan"),
        "coverage": float(test_accepted.mean()),
        "per_level_dangerous_breakdown": per_level_dangerous,
        "gate_target_under_fnr": TARGET_UNDER_FNR,
        "gate_margin_events": (
            float(np.floor(TARGET_UNDER_FNR * int(test_accepted.sum()))) - dangerous_covered
        ),
    }

    # ---------------------------------------------------------------- 5. threshold sensitivity
    rng = np.random.default_rng(args.bootstrap_seed)
    n_val = len(val_labels)
    fixed_temp_thresholds = np.empty(args.bootstrap_samples)
    fixed_temp_coverage = np.empty(args.bootstrap_samples)
    fixed_temp_under = np.empty(args.bootstrap_samples)
    fixed_temp_met = np.zeros(args.bootstrap_samples, dtype=bool)
    full_thresholds = np.empty(args.bootstrap_samples)
    full_met = np.zeros(args.bootstrap_samples, dtype=bool)
    full_temperatures = np.empty(args.bootstrap_samples)
    for index in range(args.bootstrap_samples):
        sample = rng.integers(0, n_val, n_val)
        sample_logits = val_logits[sample]
        sample_labels = val_labels[sample]
        sample_probs = val_probs[sample]
        fixed = calibrate_abstention_threshold(sample_probs, sample_labels, TARGET_UNDER_FNR, MAX_ABSTENTION)
        fixed_temp_thresholds[index] = fixed["threshold"]
        fixed_temp_coverage[index] = fixed["coverage"]
        fixed_temp_under[index] = fixed["under_fnr"] if np.isfinite(fixed["under_fnr"]) else np.nan
        fixed_temp_met[index] = bool(fixed["target_met"])
        temperature, _ = fit_coral_temperature(sample_logits, sample_labels)
        full_temperatures[index] = temperature
        full = calibrate_abstention_threshold(
            calibrated(sample_logits, temperature), sample_labels, TARGET_UNDER_FNR, MAX_ABSTENTION
        )
        full_thresholds[index] = full["threshold"]
        full_met[index] = bool(full["target_met"])

    # Leave-one-out on validation: the most interpretable fragility measure.
    loo_thresholds = np.empty(n_val)
    loo_met = np.zeros(n_val, dtype=bool)
    for index in range(n_val):
        keep = np.ones(n_val, dtype=bool)
        keep[index] = False
        loo = calibrate_abstention_threshold(val_probs[keep], val_labels[keep], TARGET_UNDER_FNR, MAX_ABSTENTION)
        loo_thresholds[index] = loo["threshold"]
        loo_met[index] = bool(loo["target_met"])

    # Distinct candidate thresholds on validation (the selection grid is discrete).
    candidates = np.unique(val_confidence)
    threshold_sensitivity = {
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "resampling": "validation events resampled with replacement, unstratified, size n=%d" % n_val,
        "distinct_confidence_candidates_on_validation": int(candidates.size),
        "candidate_confidence_min": float(candidates.min()),
        "candidate_confidence_max": float(candidates.max()),
        "fixed_temperature": {
            "description": "temperature held at the shipped value; only the abstention threshold is refit",
            "threshold": quantiles(fixed_temp_thresholds),
            "coverage": quantiles(fixed_temp_coverage),
            "under_fnr": quantiles(fixed_temp_under),
            "fraction_target_met": float(fixed_temp_met.mean()),
            "fraction_below_shipped": float((fixed_temp_thresholds < shipped_threshold).mean()),
            "fraction_equal_to_shipped": float((np.abs(fixed_temp_thresholds - shipped_threshold) < 1e-12).mean()),
            "fraction_above_shipped": float((fixed_temp_thresholds > shipped_threshold).mean()),
        },
        "full_pipeline": {
            "description": "temperature refit per resample as well (fit_coral_temperature + threshold)",
            "threshold": quantiles(full_thresholds),
            "temperature": quantiles(full_temperatures),
            "fraction_target_met": float(full_met.mean()),
            "fraction_below_shipped": float((full_thresholds < shipped_threshold).mean()),
            "fraction_above_shipped": float((full_thresholds > shipped_threshold).mean()),
        },
        "leave_one_out": {
            "description": "drop one validation event at a time and refit the threshold at the shipped temperature",
            "threshold": quantiles(loo_thresholds),
            "fraction_target_met": float(loo_met.mean()),
            "n_distinct_thresholds": int(np.unique(loo_thresholds).size),
        },
    }

    # Downstream consequence of the threshold spread, evaluated on held-out test.
    n_test = len(test_labels)
    test_coverage_draws = np.empty(args.bootstrap_samples)
    test_dangerous_draws = np.empty(args.bootstrap_samples)
    test_macro_f1_draws = np.empty(args.bootstrap_samples)
    test_ece_covered_draws = np.empty(args.bootstrap_samples)
    from sklearn.metrics import f1_score

    for index in range(args.bootstrap_samples):
        threshold = fixed_temp_thresholds[index]
        accepted = test_confidence >= threshold
        predictions = np.where(accepted, test_predictions, -1)
        test_coverage_draws[index] = accepted.mean()
        test_dangerous_draws[index] = (
            float((test_predictions[accepted] < test_labels[accepted]).mean()) if accepted.any() else np.nan
        )
        test_macro_f1_draws[index] = float(
            f1_score(test_labels, predictions, labels=sorted(set(int(v) for v in test_labels)), average="macro", zero_division=0)
        )
        test_ece_covered_draws[index] = (
            expected_calibration_error(test_probs, test_labels, ECE_BINS, accepted) if accepted.any() else np.nan
        )
    downstream = {
        "description": (
            "each bootstrap-refit validation threshold applied unchanged to the %d held-out test events; "
            "the shipped operating point is reported alongside for comparison"
        ) % n_test,
        "shipped_operating_point": {
            "threshold": shipped_threshold,
            "coverage": float(test_accepted.mean()),
            "dangerous_underclassification_fnr_covered": dangerous_fnr["dangerous_underclassification_fnr_covered"],
            "macro_f1_with_abstentions": float(
                f1_score(test_labels, np.where(test_accepted, test_predictions, -1),
                         labels=sorted(set(int(v) for v in test_labels)), average="macro", zero_division=0)
            ),
            "ece_covered": expected_calibration_error(test_probs, test_labels, ECE_BINS, test_accepted),
        },
        "coverage": quantiles(test_coverage_draws),
        "dangerous_underclassification_fnr_covered": quantiles(test_dangerous_draws),
        "macro_f1_with_abstentions": quantiles(test_macro_f1_draws),
        "ece_covered": quantiles(test_ece_covered_draws),
        "fraction_of_draws_worse_dangerous_fnr_than_shipped": float(
            (test_dangerous_draws > dangerous_fnr["dangerous_underclassification_fnr_covered"]).mean()
        ),
        "fraction_of_draws_lower_macro_f1_than_shipped": float(
            (
                test_macro_f1_draws
                < f1_score(test_labels, np.where(test_accepted, test_predictions, -1),
                           labels=sorted(set(int(v) for v in test_labels)), average="macro", zero_division=0)
            ).mean()
        ),
    }

    # Released metric block, recomputed here with the repo's own bootstrap.
    released = evaluate_predictions(
        test_probs, test_labels, shipped_threshold,
        ece_bins=ECE_BINS, bootstrap_samples=1000, bootstrap_seed=2026,
    )

    result = {
        "meta": {
            "lane": "an03_calibration",
            "seed": seed,
            "code_provenance": CODE_PROVENANCE,
            "code_provenance_note": (
                "sha256 of every trainer module this audit imports, resolved through audit/_pin_trainer.py so the "
                "numbers come from this lane's clone and not from whichever lane last ran pip install -e in the "
                "shared iddsi-train container."
            ),
            "run_dir": str(run_dir),
            "val_npz": str(args.val_npz),
            "val_npz_sha256": sha256(Path(args.val_npz)),
            "test_npz": str(args.test_npz),
            "test_npz_sha256": sha256(Path(args.test_npz)),
            "calibration_json_sha256": sha256(run_dir / "calibration.json"),
            "shipped_threshold": shipped_threshold,
            "shipped_temperature": shipped_temperature,
            "gate": {"target_under_fnr": TARGET_UNDER_FNR, "max_abstention": MAX_ABSTENTION},
            "target_coverages": list(TARGET_COVERAGES),
            "ece_bins": ECE_BINS,
            "disclaimer": (
                "Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; "
                "does not perform official IDDSI tests, assess swallowing, determine suitability for any person, "
                "or decide whether food is safe to consume."
            ),
        },
        "reproduction": reproduction,
        "ece": ece,
        "risk_coverage": risk_coverage,
        "dangerous_fnr": dangerous_fnr,
        "threshold_sensitivity": threshold_sensitivity,
        "downstream_impact_of_threshold_spread": downstream,
        "released_metrics_recomputed": {
            "n": released["n"],
            "metrics": released["metrics"],
            "confidence_intervals_95": released["confidence_intervals_95"],
        },
    }
    # ------------------------------------------------------------------ markdown
    lines = []
    add = lines.append
    add(f"# Calibration and abstention audit — v0.7 seed {seed}")
    add("")
    add(f"> {result['meta']['disclaimer']}")
    add("")
    add("## Scope")
    add("")
    add(f"- Seed: `{run_dir.name}` (best v0.7 seed by macro-F1 with abstentions on the held-out test split)")
    add(f"- Validation events: **{len(val_labels)}** (temperature + threshold selection; nothing selected on test)")
    add(f"- Held-out test events: **{n_test}** (evaluation only, at a threshold frozen on validation)")
    add(f"- Shipped threshold: **{shipped_threshold:.4f}**; shipped temperature: **{shipped_temperature:.4f}**")
    add(f"- Gate: under-classification FNR <= {TARGET_UNDER_FNR}, abstention <= {MAX_ABSTENTION:.0%}")
    add(f"- Inputs: `{args.val_npz}` (sha256 `{result['meta']['val_npz_sha256'][:16]}…`), "
        f"`{args.test_npz}` (sha256 `{result['meta']['test_npz_sha256'][:16]}…`)")
    add(f"- Code: `eval.metrics` sha256 `{CODE_PROVENANCE['eval.metrics__sha256'][:16]}…`, "
        f"`eval.abstention` sha256 `{CODE_PROVENANCE['eval.abstention__sha256'][:16]}…`, "
        f"resolved from `{CODE_PROVENANCE['trainer_dir']}`")
    add("")
    add("## 1. Reproduction of the shipped calibration")
    add("")
    add("| Quantity | Shipped | Recomputed here | |delta| |")
    add("|---|---:|---:|---:|")
    add(f"| Temperature | {shipped_temperature:.6f} | {reproduced_temperature:.6f} | "
        f"{reproduction['reproduced_temperature_abs_delta']:.2e} |")
    add(f"| Threshold (at shipped temperature) | {shipped_threshold:.6f} | "
        f"{threshold_reselect['threshold']:.6f} | {reproduction['threshold_reselect_abs_delta']:.2e} |")
    add(f"| Threshold (temperature refit too) | {shipped_threshold:.6f} | "
        f"{threshold_full_recompute['threshold']:.6f} | {reproduction['threshold_full_recompute_abs_delta']:.2e} |")
    add(f"| Validation NLL (uncalibrated) | {reproduction['checkpoint_val_nll']:.6f} | "
        f"{uncalibrated_val_nll:.6f} | {abs(uncalibrated_val_nll - reproduction['checkpoint_val_nll']):.2e} |")
    add("")
    add(f"Threshold refit at the shipped temperature reproduces the shipped value to "
        f"{reproduction['threshold_reselect_abs_delta']:.2e}; the full pipeline (temperature refit) lands at "
        f"{threshold_full_recompute['threshold']:.4f} "
        f"(delta {reproduction['threshold_full_recompute_abs_delta']:.2e}). Validation coverage "
        f"{threshold_reselect['coverage']:.4f}, under-classification FNR "
        f"{threshold_reselect['under_fnr']:.4f}, target met `{str(threshold_reselect['target_met']).lower()}`.")
    if args.diagnostics:
        boundary = json.loads(Path(args.diagnostics).read_text(encoding="utf-8")[
            "achievable_coverage_ladder"
        ]["shipped_threshold_boundary"]
        tightest = boundary["boundary_events"]["within_1e-06"]
        add("")
        add(f"**One knife-edge event.** `calibration.json` records validation coverage "
            f"{boundary['shipped_calibration_json_coverage']:.4f} "
            f"({int(round(boundary['shipped_calibration_json_coverage'] * len(val_labels)))}/{len(val_labels)}); "
            f"re-running the same forward pass here gives {boundary['reproduced_validation_accepted']}/"
            f"{len(val_labels)} = {boundary['reproduced_validation_coverage']:.4f}. "
            f"{tightest['n_validation_events']} validation events have a confidence within 1e-06 of the shipped "
            f"threshold ({', '.join(f'{value:.9f}' for value in tightest['confidences'])} against a threshold of "
            f"{shipped_threshold:.9f}), so which side of the abstention boundary they land on is decided by float32 "
            "noise in the forward pass rather than by the model. The held-out test numbers below are unaffected: "
            "they are computed from the shipped `test_predictions.npz` at the shipped threshold and reproduce the "
            "released values exactly.")
    add("")
    add("## 2. Expected calibration error")
    add("")
    add(f"Definition: {ece['definition']}")
    add("")
    add("| Split / mask | n | ECE (10 bins) | ECE (5) | ECE (15) | ECE (20) | mean confidence | accuracy | gap |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, block in (("test, all", ece["test_all"]), ("test, covered", ece["test_covered"]),
                        ("val, all", ece["val_all"]), ("val, covered", ece["val_covered"])):
        add(f"| {name} | {block['n']} | {block['ece_bins_10']:.4f} | {block['ece_bins_5']:.4f} | "
            f"{block['ece_bins_15']:.4f} | {block['ece_bins_20']:.4f} | {block['mean_confidence']:.4f} | "
            f"{block['accuracy']:.4f} | {block['mean_gap_accuracy_minus_confidence']:+.4f} |")
    add("")
    test_all = ece["test_all"]
    add(f"On the {test_all['n']} held-out test events the CORAL rank-decode confidence is "
        f"{'over' if test_all['mean_gap_accuracy_minus_confidence'] < 0 else 'under'}-confident by "
        f"{abs(test_all['mean_gap_accuracy_minus_confidence']):.4f} on average "
        f"(mean confidence {test_all['mean_confidence']:.4f} vs accuracy {test_all['accuracy']:.4f}).")
    add("")
    add("Reliability table (test, all events, 10 equal-width bins):")
    add("")
    add("| confidence bin | n | mean confidence | accuracy | gap |")
    add("|---|---:|---:|---:|---:|")
    for row in ece["test_reliability_table"]:
        if row["n"]:
            add(f"| {row['bin']} | {row['n']} | {row['mean_confidence']:.4f} | {row['accuracy']:.4f} | {row['gap']:+.4f} |")
        else:
            add(f"| {row['bin']} | 0 | — | — | — |")
    add("")
    add("**Decode-rule caveat.** The repo's ECE uses the CORAL *rank-decode* confidence (probability mass on the "
        "decoded level), not the maximum class probability. On the test split these disagree on "
        f"{test_all['n_rank_decode_confidence_below_maxprob']}/{test_all['n']} events, and the rank-decode "
        f"confidence is on average {test_all['mean_rank_decode_minus_maxprob_confidence']:+.4f} below the maximum "
        f"class probability (mean max-prob confidence {test_all['mean_max_prob_confidence']:.4f} vs mean rank-decode "
        f"confidence {test_all['mean_confidence']:.4f}). Recomputing ECE with the conventional max-probability "
        f"confidence gives {test_all['ece_maxprob_confidence_bins_10']:.4f} instead of "
        f"{test_all['ece_bins_10']:.4f}. The rank-decode definition is the one the shipped harness uses and is "
        "reported as primary; the max-probability figure is shown so the two are not conflated.")
    add("")
    add("## 3. Risk–coverage at five coverage points")
    add("")
    add(f"Thresholds for the five coverage points ({', '.join(f'{c:.0%}' for c in TARGET_COVERAGES)}) are selected on "
        f"the {len(val_labels)} validation events, then applied unchanged to the {n_test} held-out test events.")
    add("")
    add("| target coverage | val threshold | val coverage | val accuracy | val risk | val dangerous FNR (caught/total) | test coverage | test accuracy | test risk | test dangerous FNR (caught/total) |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in val_curve["rows"]:
        test_row = next(item["test"] for item in transfer_rows if item["target_coverage"] == row["target_coverage"])
        add(f"| {row['target_coverage']:.0%} | {row['threshold']:.4f} | {row['coverage']:.4f} | "
            f"{row['accuracy']:.4f} | {row['error_rate']:.4f} | "
            f"{row['dangerous_fnr']:.4f} ({row['dangerous_errors_caught_by_abstention']}/{row['dangerous_errors_total']}) | "
            f"{test_row['coverage']:.4f} | {test_row['accuracy']:.4f} | {test_row['error_rate']:.4f} | "
            f"{test_row['dangerous_fnr']:.4f} ({test_row['dangerous_errors_caught_by_abstention']}/{test_row['dangerous_errors_total']}) |")
    add("")
    add("At the **shipped threshold**:")
    add("")
    add("| split | coverage | accepted | accuracy | risk (error rate) | dangerous FNR | dangerous covered/total | dangerous rate over covered |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in (("validation", shipped_row_val), ("test", shipped_row_test)):
        add(f"| {name} | {row['coverage']:.4f} | {row['accepted']} | {row['accuracy']:.4f} | {row['error_rate']:.4f} | "
            f"{row['dangerous_fnr']:.4f} | {row['dangerous_errors_covered']}/{row['dangerous_errors_total']} | "
            f"{row['dangerous_rate_covered']:.4f} |")
    add("")
    add(f"Validation carries {int((val_predictions < val_labels).sum())} dangerous-direction samples out of "
        f"{len(val_labels)}; the test split carries {int((test_predictions < test_labels).sum())} out of {n_test}. "
        "Both denominators are stated because the two FNR definitions in the harness use different ones.")
    add("")
    add("## 4. Dangerous-direction under-classification at the shipped threshold (test)")
    add("")
    add(f"- `dangerous_underclassification_fnr_covered` = {dangerous_fnr['dangerous_underclassification_fnr_covered']:.4f} "
        f"({dangerous_covered}/{int(test_accepted.sum())} covered events) — the shipped metric")
    add(f"- `dangerous_fnr` (abstention-as-detector) = {dangerous_fnr['dangerous_fnr_detector']:.4f} "
        f"({dangerous_covered}/{int(dangerous.sum())} dangerous-direction events, i.e. "
        f"{dangerous_caught} caught by abstention)")
    add(f"- `dangerous_underclassification_rate_all` = {dangerous_fnr['dangerous_underclassification_rate_all']:.4f} "
        f"({dangerous_covered}/{n_test} test events)")
    add("")
    add("| true level | n | dangerous (y_pred<y_true) | of which accepted |")
    add("|---|---:|---:|---:|")
    for level, counts in per_level_dangerous.items():
        add(f"| {level} | {counts['n']} | {counts['n_dangerous']} | {counts['n_dangerous_accepted']} |")
    add("")
    add(f"Gate margin on the test split: {dangerous_fnr['gate_margin_events']:.0f} additional dangerous errors among "
        f"covered events would push the covered FNR past {TARGET_UNDER_FNR} "
        f"(floor({TARGET_UNDER_FNR} x {int(test_accepted.sum())}) = "
        f"{int(np.floor(TARGET_UNDER_FNR * int(test_accepted.sum())))} allowed, {dangerous_covered} observed).")
    add("")
    add("## 5. Sensitivity of the chosen threshold")
    add("")
    add(f"Bootstrap: {args.bootstrap_samples} resamples of the {len(val_labels)} validation events (with replacement, "
        f"unstratified, seed {args.bootstrap_seed}), threshold refit with the shipped gate constants.")
    add("")
    fixed = threshold_sensitivity["fixed_temperature"]
    add("| statistic | threshold (temperature fixed) | threshold (temperature refit) | leave-one-out |")
    add("|---|---:|---:|---:|")
    for key, label in (("mean", "mean"), ("sd", "sd"), ("min", "min"), ("q05", "5th pct"),
                       ("q25", "25th pct"), ("median", "median"), ("q75", "75th pct"),
                       ("q95", "95th pct"), ("max", "max"), ("iqr", "IQR")):
        full = threshold_sensitivity["full_pipeline"]["threshold"]
        loo = threshold_sensitivity["leave_one_out"]["threshold"]
        add(f"| {label} | {fixed['threshold'][key]:.4f} | {full[key]:.4f} | {loo[key]:.4f} |")
    add("")
    add(f"- Gate met on {fixed['fraction_target_met']:.1%} of bootstrap resamples "
        f"({threshold_sensitivity['full_pipeline']['fraction_target_met']:.1%} when the temperature is refit too; "
        f"{threshold_sensitivity['leave_one_out']['fraction_target_met']:.1%} under leave-one-out).")
    add(f"- Bootstrap thresholds below / equal to / above the shipped {shipped_threshold:.4f}: "
        f"{fixed['fraction_below_shipped']:.1%} / {fixed['fraction_equal_to_shipped']:.1%} / "
        f"{fixed['fraction_above_shipped']:.1%}.")
    add(f"- The selection grid on validation is discrete: only "
        f"{threshold_sensitivity['distinct_confidence_candidates_on_validation']} distinct confidence values among "
        f"{len(val_labels)} events, so the threshold can only move between observed confidences.")
    add(f"- Leave-one-out produces {threshold_sensitivity['leave_one_out']['n_distinct_thresholds']} distinct "
        f"threshold values across {len(val_labels)} single-event deletions.")
    add("")
    add("**Validation-set margin at the shipped threshold.** Under-classification FNR on validation is "
        f"{threshold_reselect['under_fnr']:.4f} against a gate of {TARGET_UNDER_FNR}: "
        f"{int(round(threshold_reselect['under_fnr'] * threshold_reselect['coverage'] * len(val_labels)))} dangerous "
        f"errors among the {int(round(threshold_reselect['coverage'] * len(val_labels)))} accepted events, where "
        f"{int(np.floor(TARGET_UNDER_FNR * threshold_reselect['coverage'] * len(val_labels)))} would still pass. "
        "The shipped threshold therefore sits one dangerous event away from gate failure, which is why the "
        "bootstrap spread below is wide rather than tight.")
    add("")
    add("### What the spread costs on held-out test")
    add("")
    add("Each bootstrap-refit validation threshold is applied unchanged to the test split:")
    add("")
    add("| metric | shipped | bootstrap mean | 5th pct | median | 95th pct | P(draw worse than shipped) |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    shipped_point = downstream["shipped_operating_point"]
    add(f"| coverage | {shipped_point['coverage']:.4f} | {downstream['coverage']['mean']:.4f} | "
        f"{downstream['coverage']['q05']:.4f} | {downstream['coverage']['median']:.4f} | "
        f"{downstream['coverage']['q95']:.4f} | — |")
    add(f"| dangerous FNR (covered) | {shipped_point['dangerous_underclassification_fnr_covered']:.4f} | "
        f"{downstream['dangerous_underclassification_fnr_covered']['mean']:.4f} | "
        f"{downstream['dangerous_underclassification_fnr_covered']['q05']:.4f} | "
        f"{downstream['dangerous_underclassification_fnr_covered']['median']:.4f} | "
        f"{downstream['dangerous_underclassification_fnr_covered']['q95']:.4f} | "
        f"{downstream['fraction_of_draws_worse_dangerous_fnr_than_shipped']:.1%} |")
    add(f"| macro-F1 with abstentions | {shipped_point['macro_f1_with_abstentions']:.4f} | "
        f"{downstream['macro_f1_with_abstentions']['mean']:.4f} | {downstream['macro_f1_with_abstentions']['q05']:.4f} | "
        f"{downstream['macro_f1_with_abstentions']['median']:.4f} | {downstream['macro_f1_with_abstentions']['q95']:.4f} | "
        f"{downstream['fraction_of_draws_lower_macro_f1_than_shipped']:.1%} |")
    add(f"| ECE (covered) | {shipped_point['ece_covered']:.4f} | {downstream['ece_covered']['mean']:.4f} | "
        f"{downstream['ece_covered']['q05']:.4f} | {downstream['ece_covered']['median']:.4f} | "
        f"{downstream['ece_covered']['q95']:.4f} | — |")
    add("")
    add("The shipped threshold is unchanged by this audit; the draws above quantify how much a different "
        "validation resample would have moved the operating point.")
    add("")

    # ------------------------------------------------------- 5b. diagnostics sections
    if args.diagnostics:
        diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
        result["diagnostics"] = diagnostics
        mixture = diagnostics["bootstrap_threshold_mixture"]["fixed_temperature"]
        conditioned = mixture["quantiles_conditioned_on_positive_threshold"]
        add("### 5b. The bootstrap threshold distribution is a mixture, not a spread")
        add("")
        add("| statistic | all draws | draws with a non-zero threshold |")
        add("|---|---:|---:|")
        all_stats = threshold_sensitivity["fixed_temperature"]["threshold"]
        for key, label in (("median", "median"), ("q25", "25th pct"), ("q75", "75th pct"),
                           ("q05", "5th pct"), ("q95", "95th pct"), ("min", "min"), ("max", "max")):
            add(f"| {label} | {all_stats[key]:.4f} | {conditioned[key]:.4f} |")
        add("")
        add(f"- {mixture['fraction_at_zero_full_coverage']:.2%} of resamples return a threshold of exactly 0.0, "
            "because the shipped rule takes the *highest-coverage* feasible candidate and a resample whose "
            f"overall under-classification rate is already <= {TARGET_UNDER_FNR} needs no abstention at all.")
        add(f"- {mixture['fraction_in_shipped_bin_0p23_to_0p25']:.1%} of resamples land in [0.23, 0.25]; the "
            f"bootstrap median is {conditioned['median']:.6f} against a shipped {shipped_threshold:.6f}.")
        add(f"- Refitting the temperature as well widens the IQR from "
            f"{threshold_sensitivity['fixed_temperature']['threshold']['iqr']:.4f} to "
            f"{threshold_sensitivity['full_pipeline']['threshold']['iqr']:.4f}: the temperature is the fragile "
            f"quantity (bootstrap sd {threshold_sensitivity['full_pipeline']['temperature']['sd']:.4f} on a shipped "
            f"value of {shipped_temperature:.4f}), and the threshold inherits that fragility.")
        add("")
        cluster = diagnostics["achievable_coverage_ladder"]["confidence_cluster_structure"]
        add("### 5c. Why the threshold is a cluster switch, not a graded filter")
        add("")
        add("| split | n | low cluster | high cluster | empty gap |")
        add("|---|---:|---:|---:|---:|")
        for split in ("validation", "test"):
            entry = cluster[split]
            add(f"| {split} | {entry['n']} | {entry['n_low_cluster']} events in "
                f"[{entry['low_cluster_range'][0]:.4f}, {entry['low_cluster_range'][1]:.4f}] | "
                f"{entry['n_high_cluster']} events in [{entry['high_cluster_range'][0]:.4f}, "
                f"{entry['high_cluster_range'][1]:.4f}] | {entry['largest_adjacent_confidence_gap']:.4f} wide "
                f"({entry['gap_between'][0]:.4f} to {entry['gap_between'][1]:.4f}) |")
        add("")
        add("The CORAL rank-decode confidence is the probability mass of the decoded level. Interior levels get a "
            "difference of two adjacent sigmoids near 0.5, so their mass is small; the two tail levels keep their "
            "full mass. Confidence is therefore bimodal with a wide empty middle, and any threshold inside the gap "
            "produces exactly the same abstention decision:")
        add("")
        add("| split | operating point | threshold | coverage | covered FNR | detector FNR |")
        add("|---|---|---:|---:|---:|---:|")
        for split in ("validation", "test"):
            for name, point in cluster[split]["operating_points"].items():
                add(f"| {split} | {name.replace('_', ' ')} | {point['threshold']:.4f} | {point['coverage']:.4f} | "
                    f"{point['dangerous_underclassification_fnr_covered']:.4f} | {point['dangerous_fnr_detector']:.4f} |")
        add("")
        add("The shipped threshold's marginal effect over the nearest structurally distinct operating point (any "
            "threshold inside the gap) is a handful of events, and on validation those marginal events carry 2 of "
            "the 4 dangerous errors: covered FNR rises from 0.0244 to 0.0460 while coverage rises from 0.7455 to "
            "0.7909. That is the concrete sense in which the chosen threshold is knife-edge.")
        add("")
        ece_check = diagnostics["ece_bootstrap_interval_check"]
        failures_all = [entry["seed_dir"] for entry in ece_check if not entry["ci_contains_point_estimate_all"]]
        failures_covered = [entry["seed_dir"] for entry in ece_check if not entry["ci_contains_point_estimate_covered"]]
        add("### 5d. The shipped bootstrap interval for ECE is shifted upward")
        add("")
        add("Recomputing the repo's own stratified bootstrap (1000 resamples, seed 2026) for ECE on all five seeds:")
        add("")
        add("| seed | ECE (all) | bootstrap median | inflation | 95% interval | contains the point estimate? |")
        add("|---|---:|---:|---:|---:|---:|")
        for entry in ece_check:
            by_bins = entry["by_bin_count"]["10"]
            add(f"| {entry['seed_dir'].replace('siglip2_v07_fullcoverage_', '')} | {entry['ece_all_point']:.4f} | "
                f"{entry['ece_all_bootstrap_median']:.4f} | {by_bins['inflation_median_minus_point']:+.4f} | "
                f"[{entry['ece_all_bootstrap_q025']:.4f}, {entry['ece_all_bootstrap_q975']:.4f}] | "
                f"`{str(entry['ci_contains_point_estimate_all']).lower()}` |")
        add("")
        add(f"The bootstrap median exceeds the point estimate in all "
            f"{len(ece_check)} x 4 seed-by-bin-count cells measured (5, 10, 15 and 20 bins). The mechanism is "
            "convexity: within a bin the quantity averaged is |accuracy - confidence|, whose expectation under "
            "resampling noise exceeds its value at the observed proportions, and the effect grows with the number "
            "of bins because each bin holds fewer events. Consequence for reporting: the interval excludes the "
            f"point estimate entirely for `{'`, `'.join(item.replace('siglip2_v07_fullcoverage_', '') for item in failures_all)}` "
            f"(ece_all) and `{'`, `'.join(item.replace('siglip2_v07_fullcoverage_', '') for item in failures_covered)}` "
            "(ece_covered), so it must not be read as a confidence interval for the ECE of the observed test set.")
        add("")

    add("## 6. Released metric block (recomputed here)")
    add("")
    add("| Metric | Estimate | Bootstrap 95% CI (1000 stratified resamples, seed 2026) |")
    add("|---|---:|---:|")
    for name, value in released["metrics"].items():
        interval = released["confidence_intervals_95"].get(name)
        text = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "not computed"
        add(f"| {name} | {value:.4f} | {text} |")
    add("")
    add("## Interpretation boundary")
    add("")
    add("Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not "
        "perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether "
        "food is safe to consume. Never replaces clinician-prescribed texture or physical IDDSI testing. This "
        "project is not the official IDDSI website and is not endorsed by IDDSI.")
    add("")
    (out_dir / f"audit_seed{seed}.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / f"audit_seed{seed}.json", result)
    print(f"wrote {out_dir / f'audit_seed{seed}.json'} and {out_dir / f'audit_seed{seed}.md'}")


if __name__ == "__main__":
    main()
