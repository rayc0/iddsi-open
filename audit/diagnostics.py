"""Follow-up diagnostics for the an03_calibration audit.

Three questions the main audit script raised but did not answer:

1. The bootstrap threshold distribution has IQR 0.0028 yet min 0.0 and max
   0.61 -- it is a mixture, not a spread.  This script decomposes it into its
   point masses so the "spread" is not reported as if it were one mode.
2. The risk-coverage curve jumps from a threshold of 0.6275 at 60% coverage to
   0.2394 at 80%, which suggests the confidence distribution has a hole.  This
   script measures the achievable-coverage ladder directly.
3. The shipped bootstrap 95% CI for ECE does not contain the ECE point
   estimate.  This script measures whether that is systematic across seeds.

Nothing here changes a shipped threshold; every function that selects a
threshold runs on validation data only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pin_trainer import pin as pin_trainer  # noqa: E402

CODE_PROVENANCE = pin_trainer()

import numpy as np  # noqa: E402

from eval.calibration import calibrate_abstention_threshold, fit_coral_temperature  # noqa: E402
from eval.metrics import coral_decode, expected_calibration_error  # noqa: E402
from train.model import coral_logits_to_probs  # noqa: E402
from train.utils import write_json  # noqa: E402

import torch  # noqa: E402

TARGET_UNDER_FNR = 0.05
MAX_ABSTENTION = 0.30


def calibrated(logits: np.ndarray, temperature: float) -> np.ndarray:
    return coral_logits_to_probs(torch.from_numpy(np.asarray(logits, dtype=np.float64) / temperature)).numpy()


def value_counts_summary(values: np.ndarray, decimals: int = 6) -> list[dict]:
    rounded = np.round(values, decimals)
    unique, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)
    return [
        {"value": float(unique[index]), "count": int(counts[index]), "fraction": float(counts[index] / counts.sum())}
        for index in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--all-seed-dirs", nargs="*", default=[], help="extra seed dirs for the ECE-CI check")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    parser.add_argument("--ece-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--ece-bootstrap-seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    shipped_threshold = float(calibration["threshold"])
    shipped_temperature = float(calibration["temperature"])

    val = np.load(args.val_npz, allow_pickle=False)
    test = np.load(args.test_npz, allow_pickle=False)
    val_logits = val["cutpoint_logits"].astype(np.float64)
    val_labels = val["labels"].astype(np.int64)
    val_probs = calibrated(val_logits, shipped_temperature)
    test_probs = test["probabilities"].astype(np.float64)
    test_labels = test["labels"].astype(np.int64)
    _, val_confidence = coral_decode(val_probs)
    _, test_confidence = coral_decode(test_probs)

    # ------------------------------------------------ 1. bootstrap mixture decomposition
    rng = np.random.default_rng(args.bootstrap_seed)
    n_val = len(val_labels)
    draws = np.empty(args.bootstrap_samples)
    draws_full = np.empty(args.bootstrap_samples)
    for index in range(args.bootstrap_samples):
        sample = rng.integers(0, n_val, n_val)
        draws[index] = calibrate_abstention_threshold(
            val_probs[sample], val_labels[sample], TARGET_UNDER_FNR, MAX_ABSTENTION
        )["threshold"]
        temperature, _ = fit_coral_temperature(val_logits[sample], val_labels[sample])
        draws_full[index] = calibrate_abstention_threshold(
            calibrated(val_logits[sample], temperature), val_labels[sample], TARGET_UNDER_FNR, MAX_ABSTENTION
        )["threshold"]
    positive = draws[draws > 0.0]
    mixture = {
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "fixed_temperature": {
            "distinct_values": value_counts_summary(draws),
            "fraction_at_zero_full_coverage": float((draws == 0.0).mean()),
            "fraction_at_shipped_threshold": float((np.abs(draws - shipped_threshold) < 1e-9).mean()),
            "fraction_in_shipped_bin_0p23_to_0p25": float(((draws >= 0.23) & (draws <= 0.25)).mean()),
            "quantiles_conditioned_on_positive_threshold": {
                "n": int(positive.size),
                "min": float(positive.min()),
                "q05": float(np.percentile(positive, 5)),
                "q25": float(np.percentile(positive, 25)),
                "median": float(np.percentile(positive, 50)),
                "q75": float(np.percentile(positive, 75)),
                "q95": float(np.percentile(positive, 95)),
                "max": float(positive.max()),
                "iqr": float(np.percentile(positive, 75) - np.percentile(positive, 25)),
            },
        },
        "full_pipeline": {
            "distinct_values": value_counts_summary(draws_full),
            "fraction_at_zero_full_coverage": float((draws_full == 0.0).mean()),
            "fraction_at_shipped_threshold": float((np.abs(draws_full - shipped_threshold) < 1e-9).mean()),
        },
        "interpretation": (
            "A threshold of exactly 0.0 is not noise: it is what the shipped selection rule returns when the "
            "validation resample already satisfies under-classification FNR <= 0.05 at 100% coverage, because the "
            "rule takes the HIGHEST-coverage feasible candidate. The bootstrap distribution is therefore a mixture "
            "of 'no abstention needed' and 'abstain below ~0.24', and its overall sd overstates the instability of "
            "the operating point the shipped run actually chose."
        ),
    }

    # ------------------------------------------------ 2. achievable coverage ladder
    val_sorted = np.sort(val_confidence)[::-1]
    test_sorted = np.sort(test_confidence)[::-1]
    gaps = []
    for index in range(min(len(val_sorted) - 1, 40)):
        low, high = val_sorted[index + 1], val_sorted[index]
        gaps.append({"rank": index + 1, "coverage": (index + 1) / len(val_sorted), "lower": float(low), "upper": float(high),
                     "gap": float(high - low)})
    widest = sorted(gaps, key=lambda item: -item["gap"])[:6]
    ladder = {
        "validation": {
            "n": int(len(val_sorted)),
            "top_confidences": [float(value) for value in val_sorted[:12]],
            "widest_adjacent_confidence_gaps": widest,
            "n_validation_events_in_0p28_to_0p53": int(((val_confidence > 0.28) & (val_confidence < 0.53)).sum()),
            "confidence_histogram_0p0_to_1p0_step_0p05": {
                f"[{low:.2f},{low + 0.05:.2f})": int(((val_confidence >= low) & (val_confidence < low + 0.05)).sum())
                for low in np.arange(0.0, 1.0, 0.05)
            },
        },
        "test": {
            "n": int(len(test_sorted)),
            "n_test_events_in_0p28_to_0p53": int(((test_confidence > 0.28) & (test_confidence < 0.53)).sum()),
            "confidence_histogram_0p0_to_1p0_step_0p05": {
                f"[{low:.2f},{low + 0.05:.2f})": int(((test_confidence >= low) & (test_confidence < low + 0.05)).sum())
                for low in np.arange(0.0, 1.0, 0.05)
            },
        },
        "achievable_coverages_validation": [
            {"accepted": int(k), "coverage": float(k / len(val_sorted)), "threshold": float(val_sorted[k - 1])}
            for k in range(1, len(val_sorted) + 1)
            if k == 1 or k == len(val_sorted) or val_sorted[k - 1] != val_sorted[k - 2]
        ],
    }
    # Boundary events: how tightly the shipped threshold pins coverage.
    boundary = {}
    for tolerance in (1e-9, 1e-6, 1e-4, 1e-3, 1e-2):
        near = np.abs(val_confidence - shipped_threshold) <= tolerance
        boundary[f"within_{tolerance:g}"] = {
            "n_validation_events": int(near.sum()),
            "event_indices": np.flatnonzero(near).tolist(),
            "confidences": [float(value) for value in val_confidence[near]],
        }
    # Coverage implied by the shipped threshold on the reproduced validation set.
    accepted_val = int((val_confidence >= shipped_threshold).sum())
    order = np.argsort(-val_confidence)
    ladder["confidence_rank_window_10_to_45"] = [
        {"rank": int(rank + 1), "confidence": float(val_confidence[order[rank]]), "event_id": str(val["event_ids"][order[rank]])}
        for rank in range(9, 45)
    ]
    ladder["confidence_rank_window_70_to_100"] = [
        {"rank": int(rank + 1), "confidence": float(val_confidence[order[rank]]), "event_id": str(val["event_ids"][order[rank]])}
        for rank in range(69, min(100, len(order)))
    ]
    # Fine-grained band counts, to locate the hole in the confidence distribution
    # that makes the abstention threshold behave like a cluster switch.
    band_edges = [0.0, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.0001]
    for split_name, confidences in (("validation", val_confidence), ("test", test_confidence)):
        bands = {}
        for low, high in zip(band_edges[:-1], band_edges[1:]):
            count = int(((confidences >= low) & (confidences < high)).sum())
            if count:
                bands[f"[{low:.2f},{high:.2f})"] = count
        ladder[f"{split_name}_confidence_bands"] = bands

    # The abstention threshold is a cluster switch: measure the hole explicitly.
    cluster = {}
    for split_name, confidences, labels in (
        ("validation", val_confidence, val_labels),
        ("test", test_confidence, test_labels),
    ):
        from eval.metrics import coral_decode as _decode

        predictions, _ = _decode(val_probs if split_name == "validation" else test_probs)
        ordered = np.sort(confidences)
        adjacent = ordered[1:] - ordered[:-1]
        split_at = int(np.argmax(adjacent))
        low_cluster = confidences <= ordered[split_at]
        high_cluster = ~low_cluster
        hole_threshold = float((ordered[split_at] + ordered[split_at + 1]) / 2.0)
        operating_points = {}
        for name, threshold in (
            ("shipped", shipped_threshold),
            ("inside_the_hole", hole_threshold),
            ("low_cluster_maximum", float(ordered[split_at])),
        ):
            accepted = confidences >= threshold
            operating_points[name] = {
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "accepted": int(accepted.sum()),
                "dangerous_underclassification_fnr_covered": (
                    float((predictions[accepted] < labels[accepted]).mean()) if accepted.any() else float("nan")
                ),
                "dangerous_fnr_detector": (
                    float(((predictions < labels) & accepted).sum() / max(int((predictions < labels).sum()), 1))
                    if int((predictions < labels).sum())
                    else float("nan")
                ),
            }
        cluster[split_name] = {
            "n": int(confidences.size),
            "largest_adjacent_confidence_gap": float(adjacent[split_at]),
            "gap_between": [float(ordered[split_at]), float(ordered[split_at + 1])],
            "n_low_cluster": int(low_cluster.sum()),
            "low_cluster_range": [float(confidences[low_cluster].min()), float(confidences[low_cluster].max())],
            "n_high_cluster": int(high_cluster.sum()),
            "high_cluster_range": [float(confidences[high_cluster].min()), float(confidences[high_cluster].max())],
            "n_plateaus_gaps_above_0p01": int((adjacent > 0.01).sum()),
            "operating_points": operating_points,
        }
    ladder["confidence_cluster_structure"] = cluster
    ladder["cluster_note"] = (
        "The CORAL rank-decode confidence is the probability mass of the decoded level. For interior levels that "
        "mass is a difference of two adjacent sigmoids near 0.5 and is therefore small, while the two tail levels "
        "(L3, L7) keep their full mass. The result is a bimodal confidence distribution with a wide empty middle, "
        "so a confidence threshold acts as a cluster switch rather than a graded filter."
    )
    ladder["shipped_threshold_boundary"] = {
        "shipped_threshold": shipped_threshold,
        "shipped_calibration_json_coverage": float(calibration["coverage"]),
        "reproduced_validation_accepted": accepted_val,
        "reproduced_validation_coverage": accepted_val / len(val_labels),
        "boundary_events": boundary,
        "note": (
            "calibration.json records the coverage the ORIGINAL run measured on its own validation forward pass. "
            "The reproduction re-runs that forward pass on CPU, so a validation event whose confidence lands within "
            "float32 noise of the threshold can flip sides. Any nonzero count in the tightest tolerance bucket is "
            "such a knife-edge event."
        ),
    }

    # ------------------------------------------------ 3. ECE bootstrap CI anomaly
    ece_check = []
    for seed_dir in [run_dir, *[Path(item) for item in args.all_seed_dirs]]:
        arrays = np.load(seed_dir / "test_predictions.npz", allow_pickle=False)
        probabilities = arrays["probabilities"].astype(np.float64)
        labels = arrays["labels"].astype(np.int64)
        seed_calibration = json.loads((seed_dir / "calibration.json").read_text(encoding="utf-8"))
        threshold = float(seed_calibration["threshold"])
        point = expected_calibration_error(probabilities, labels, 10)
        point_covered = expected_calibration_error(probabilities, labels, 10, coral_decode(probabilities)[1] >= threshold)
        rng_ece = np.random.default_rng(args.ece_bootstrap_seed)
        by_class = [np.flatnonzero(labels == label) for label in range(5)]
        all_draws, covered_draws = [], []
        for _ in range(args.ece_bootstrap_samples):
            sampled = np.concatenate([rng_ece.choice(indices, len(indices), replace=True) for indices in by_class if len(indices)])
            rng_ece.shuffle(sampled)
            all_draws.append(expected_calibration_error(probabilities[sampled], labels[sampled], 10))
            mask = coral_decode(probabilities[sampled])[1] >= threshold
            covered_draws.append(expected_calibration_error(probabilities[sampled], labels[sampled], 10, mask))
        all_draws = np.asarray(all_draws)
        covered_draws = np.asarray(covered_draws)
        # Does the upward shift grow with the number of bins?  If it does, the
        # mechanism is the convexity of |accuracy - confidence| inside a bin
        # averaged over resampling noise, not a coding error.
        by_bins = {}
        for bins in (5, 10, 15, 20):
            point_bins = expected_calibration_error(probabilities, labels, bins)
            rng_bins = np.random.default_rng(args.ece_bootstrap_seed)
            draws_bins = []
            for _ in range(args.ece_bootstrap_samples):
                sampled = np.concatenate(
                    [rng_bins.choice(indices, len(indices), replace=True) for indices in by_class if len(indices)]
                )
                rng_bins.shuffle(sampled)
                draws_bins.append(expected_calibration_error(probabilities[sampled], labels[sampled], bins))
            draws_bins = np.asarray(draws_bins)
            by_bins[str(bins)] = {
                "point": float(point_bins),
                "bootstrap_median": float(np.median(draws_bins)),
                "inflation_median_minus_point": float(np.median(draws_bins) - point_bins),
                "bootstrap_q025": float(np.percentile(draws_bins, 2.5)),
                "bootstrap_q975": float(np.percentile(draws_bins, 97.5)),
                "ci_contains_point": bool(np.percentile(draws_bins, 2.5) <= point_bins <= np.percentile(draws_bins, 97.5)),
            }
        ece_check.append(
            {
                "seed_dir": seed_dir.name,
                "n_test_events": int(len(labels)),
                "threshold": threshold,
                "ece_all_point": float(point),
                "ece_all_bootstrap_median": float(np.median(all_draws)),
                "ece_all_bootstrap_q025": float(np.percentile(all_draws, 2.5)),
                "ece_all_bootstrap_q975": float(np.percentile(all_draws, 97.5)),
                "ece_all_fraction_of_draws_below_point": float((all_draws < point).mean()),
                "ece_covered_point": float(point_covered),
                "ece_covered_bootstrap_median": float(np.median(covered_draws)),
                "ece_covered_bootstrap_q025": float(np.percentile(covered_draws, 2.5)),
                "ece_covered_bootstrap_q975": float(np.percentile(covered_draws, 97.5)),
                "ece_covered_fraction_of_draws_below_point": float((covered_draws < point_covered).mean()),
                "ci_contains_point_estimate_all": bool(
                    np.percentile(all_draws, 2.5) <= point <= np.percentile(all_draws, 97.5)
                ),
                "ci_contains_point_estimate_covered": bool(
                    np.percentile(covered_draws, 2.5) <= point_covered <= np.percentile(covered_draws, 97.5)
                ),
                "by_bin_count": by_bins,
            }
        )

    result = {
        "meta": {
            "lane": "an03_calibration",
            "run_dir": str(run_dir),
            "code_provenance": CODE_PROVENANCE,
            "shipped_threshold": shipped_threshold,
            "shipped_temperature": shipped_temperature,
            "gate": {"target_under_fnr": TARGET_UNDER_FNR, "max_abstention": MAX_ABSTENTION},
        },
        "bootstrap_threshold_mixture": mixture,
        "achievable_coverage_ladder": ladder,
        "ece_bootstrap_interval_check": ece_check,
    }
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
