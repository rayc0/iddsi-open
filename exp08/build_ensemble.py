"""Seed-ensembling experiment for the exp08_ensemble lane.

Builds probability-averaged ensembles over the five v0.7 seed checkpoints,
fixes the abstention threshold on the VALIDATION split with the same rule the
single seeds use (``train.eval.calibration.calibrate_abstention_threshold``,
target under-classification FNR <= 0.05, max abstention 0.30), and evaluates the
held-out TEST split with the trainer's own evaluator (``eval.metrics`` /
``eval.cli``).

What is compared (all on the same 108 group-disjoint held-out test events):
  * each single seed, at its own validation-fitted threshold  (v0.7 baseline)
  * the mean of those five single-seed kappas                   (v0.7 baseline)
  * the 5-seed probability average, threshold fixed on validation
  * every 2/3/4-seed subset ensemble (diagnostic: shows the trend and the
    spread, NOT a model-selection result -- see the caveat in the report)
  * a coverage-matched diagnostic that removes the coverage confound

Outputs (under exp08/ensemble/):
  mean_prob_test.npz          ensemble test probabilities fed to eval.cli
  mean_prob_metrics.json      eval.cli metrics for the ensemble
  mean_prob_calibration.json  the validation-fitted threshold + gate outcome
  ensemble_analysis.json      every number quoted in the report
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "trainer")):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval.calibration import calibrate_abstention_threshold  # noqa: E402
from eval.metrics import _point_metrics, evaluate_predictions  # noqa: E402
from train.model import coral_decode, coral_logits_to_probs  # noqa: E402

SEEDS = [17, 29, 43, 71, 101]
LEVELS = [3, 4, 5, 6, 7]


def load_seed(raw_dir: Path, seed: int, release_dir: Path) -> dict:
    tag = f"v07_seed{seed}"
    calibration = json.loads(
        (release_dir / f"siglip2_v07_fullcoverage_seed{seed}" / "calibration.json").read_text(encoding="utf-8")
    )
    out = {"seed": seed, "temperature": float(calibration["temperature"]), "threshold": float(calibration["threshold"])}
    for split in ("val", "test"):
        arrays = np.load(raw_dir / f"{tag}_{split}_raw.npz")
        probs = coral_logits_to_probs(torch.from_numpy(arrays["logits"] / calibration["temperature"])).numpy()
        out[f"{split}_probs"] = probs
        out[f"{split}_logits_T"] = arrays["logits"] / calibration["temperature"]
        out[f"{split}_labels"] = arrays["labels"]
        out[f"{split}_event_ids"] = arrays["event_ids"]
    return out


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-12)).mean())


def decode_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    metrics = _point_metrics(probs, labels, threshold, 10)
    metrics["threshold"] = float(threshold)
    metrics["n_events"] = int(len(labels))
    return metrics


def threshold_for_coverage(probs: np.ndarray, labels: np.ndarray, target_coverage: float) -> dict:
    """Diagnostic only: pick the TEST threshold that reproduces a target coverage.

    Fitted on the held-out test split, so it is not a legitimate model choice;
    it exists solely to show whether a kappa gap is a coverage artifact.
    """
    predictions, confidence = coral_decode(probs)
    candidates = np.unique(np.concatenate(([0.0], confidence, [np.nextafter(confidence.max(), np.inf)])))
    best = None
    for threshold in candidates:
        coverage = float((confidence >= threshold).mean())
        if abs(coverage - target_coverage) < 1e-12:
            best = float(threshold)
            break
    if best is None:
        best = float(candidates[np.argmin(np.abs([(confidence >= t).mean() - target_coverage for t in candidates]))])
    return decode_metrics(probs, labels, best)


def paired_bootstrap_delta(
    probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray, thr_a: float, thr_b: float,
    samples: int, seed: int,
) -> dict:
    """Stratified paired bootstrap of QWK(a) - QWK(b) over the test events."""
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == label) for label in range(5)]
    deltas, kappa_a, kappa_b = [], [], []
    for _ in range(samples):
        parts = [rng.choice(idx, len(idx), replace=True) for idx in by_class if len(idx)]
        sampled = np.concatenate(parts)
        rng.shuffle(sampled)
        try:
            ka = _point_metrics(probs_a[sampled], labels[sampled], thr_a, 10)["weighted_kappa_covered"]
            kb = _point_metrics(probs_b[sampled], labels[sampled], thr_b, 10)["weighted_kappa_covered"]
        except ValueError:
            continue
        if np.isfinite(ka) and np.isfinite(kb):
            deltas.append(ka - kb)
            kappa_a.append(ka)
            kappa_b.append(kb)
    deltas = np.asarray(deltas)
    return {
        "bootstrap_samples": int(len(deltas)),
        "mean_delta_qwk": float(deltas.mean()),
        "ci95_delta_qwk": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_delta_gt_0": float((deltas > 0).mean()),
        "ci95_qwk_a": [float(np.percentile(kappa_a, 2.5)), float(np.percentile(kappa_a, 97.5))],
        "ci95_qwk_b": [float(np.percentile(kappa_b, 2.5)), float(np.percentile(kappa_b, 97.5))],
    }


def mcnemar(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray, thr_a: float, thr_b: float) -> dict:
    """Exact McNemar on the covered events of both methods (abstentions counted as wrong)."""
    from math import comb

    pred_a, conf_a = coral_decode(probs_a)
    pred_b, conf_b = coral_decode(probs_b)
    correct_a = (pred_a == labels) & (conf_a >= thr_a)
    correct_b = (pred_b == labels) & (conf_b >= thr_b)
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((correct_b & ~correct_a).sum())
    n = a_only + b_only
    p = float(comb(n, a_only) / 2 ** n * (1.0 if a_only * 2 > n else 1.0)) if n else float("nan")
    # two-sided exact binomial test at p=0.5
    if n:
        tail = sum(comb(n, k) for k in range(0, min(a_only, b_only) + 1)) / 2 ** n
        p = float(min(1.0, 2 * tail))
    return {
        "correct_a": int(correct_a.sum()),
        "correct_b": int(correct_b.sum()),
        "discordant_a_only": a_only,
        "discordant_b_only": b_only,
        "exact_two_sided_p": p,
        "n_test": int(len(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="exp08/raw")
    parser.add_argument("--release-dir", default="/work/runs/v07")
    parser.add_argument("--out-dir", default="exp08/ensemble")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = {seed: load_seed(raw_dir, seed, Path(args.release_dir)) for seed in SEEDS}

    labels = seeds[SEEDS[0]]["test_labels"]
    val_labels = seeds[SEEDS[0]]["val_labels"]
    event_ids = seeds[SEEDS[0]]["test_event_ids"]
    for seed in SEEDS:
        assert (seeds[seed]["test_labels"] == labels).all(), "test label order differs between seeds"
        assert (seeds[seed]["test_event_ids"] == event_ids).all(), "test event order differs between seeds"
        assert (seeds[seed]["val_labels"] == val_labels).all(), "val label order differs between seeds"

    # ---- singles (v0.7 baseline), recomputed here from the same code path ----
    singles = {}
    for seed in SEEDS:
        data = seeds[seed]
        singles[seed] = decode_metrics(data["test_probs"], labels, data["threshold"])
        singles[seed]["val_nll"] = nll(data["val_probs"], val_labels)
        singles[seed]["temperature"] = data["temperature"]
    singles_kappas = np.array([singles[s]["weighted_kappa_covered"] for s in SEEDS])
    mean_single_kappa = float(singles_kappas.mean())

    # ---- ensemble: mean of temperature-calibrated probabilities ----
    ensemble_val = np.mean([seeds[s]["val_probs"] for s in SEEDS], axis=0)
    ensemble_test = np.mean([seeds[s]["test_probs"] for s in SEEDS], axis=0)
    ensemble_logit_val = np.mean([seeds[s]["val_logits_T"] for s in SEEDS], axis=0)
    ensemble_logit_test = np.mean([seeds[s]["test_logits_T"] for s in SEEDS], axis=0)

    calibration = calibrate_abstention_threshold(ensemble_val, val_labels, 0.05, 0.30)
    threshold = float(calibration["threshold"])
    ensemble = decode_metrics(ensemble_test, labels, threshold)
    ensemble["val_nll"] = nll(ensemble_val, val_labels)
    ensemble["calibration"] = calibration

    # ---- secondary ensemble: mean of temperature-scaled cut-point logits ----
    logit_calibration = calibrate_abstention_threshold(
        coral_logits_to_probs(torch.from_numpy(ensemble_logit_val)).numpy(), val_labels, 0.05, 0.30
    )
    logit_test = coral_logits_to_probs(torch.from_numpy(ensemble_logit_test)).numpy()
    logit_ensemble = decode_metrics(logit_test, labels, float(logit_calibration["threshold"]))
    logit_ensemble["calibration"] = logit_calibration

    # ---- subset ensembles (diagnostic) ----
    subsets = {}
    for size in (2, 3, 4, 5):
        for combo in itertools.combinations(SEEDS, size):
            val_probs = np.mean([seeds[s]["val_probs"] for s in combo], axis=0)
            test_probs = np.mean([seeds[s]["test_probs"] for s in combo], axis=0)
            cal = calibrate_abstention_threshold(val_probs, val_labels, 0.05, 0.30)
            key = "+".join(str(s) for s in combo)
            subsets[key] = decode_metrics(test_probs, labels, float(cal["threshold"]))
            subsets[key]["val_target_met"] = bool(cal["target_met"])

    # ---- coverage-matched diagnostic (test-fitted; not a model claim) ----
    mean_single_coverage = float(np.mean([singles[s]["coverage"] for s in SEEDS]))
    coverage_matched = {
        "target_coverage_mean_single": mean_single_coverage,
        "ensemble_at_mean_single_coverage": threshold_for_coverage(ensemble_test, labels, mean_single_coverage),
        "singles_at_ensemble_coverage": {
            str(s): threshold_for_coverage(seeds[s]["test_probs"], labels, ensemble["coverage"]) for s in SEEDS
        },
    }

    # ---- paired comparisons: ensemble vs each single ----
    paired = {}
    for seed in SEEDS:
        paired[str(seed)] = {
            "bootstrap": paired_bootstrap_delta(
                ensemble_test, seeds[seed]["test_probs"], labels, threshold, seeds[seed]["threshold"],
                args.bootstrap_samples, args.bootstrap_seed,
            ),
            "mcnemar": mcnemar(ensemble_test, seeds[seed]["test_probs"], labels, threshold, seeds[seed]["threshold"]),
        }

    # ---- write the ensemble prediction file and run the trainer's own eval CLI ----
    npz_path = out_dir / "mean_prob_test.npz"
    np.savez_compressed(npz_path, probabilities=ensemble_test, labels=labels, event_ids=event_ids)
    (out_dir / "mean_prob_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    cli = subprocess.run(
        [sys.executable, "-m", "eval.cli", "--predictions", str(npz_path.resolve()),
         "--threshold", str(threshold), "--output-dir", str((out_dir / "eval_cli").resolve())],
        cwd=str(REPO / "trainer"), capture_output=True, text=True,
    )
    (out_dir / "eval_cli_stdout.txt").write_text(cli.stdout + "\n" + cli.stderr, encoding="utf-8")
    if cli.returncode != 0:
        print(cli.stderr[-2000:])

    analysis = {
        "experiment": "exp08_ensemble / seedens (probability averaging over v0.7 seed checkpoints)",
        "denominators": {
            "manifest_rows": 532,
            "train_rows": 314,
            "val_rows": int(len(val_labels)),
            "test_rows": int(len(labels)),
            "test_label_counts_L3_L7": np.bincount(labels, minlength=5).tolist(),
            "val_label_counts_L3_L7": np.bincount(val_labels, minlength=5).tolist(),
            "split": "group-disjoint val/test inherited from v0.3; 108 held-out test events",
            "threshold_rule": "calibrate_abstention_threshold(target_under_fnr=0.05, max_abstention=0.30) on val",
        },
        "singles_v07": {str(s): singles[s] for s in SEEDS},
        "mean_of_singles_qwk": mean_single_kappa,
        "single_qwk_min_max": [float(singles_kappas.min()), float(singles_kappas.max())],
        "ensemble_mean_prob": ensemble,
        "ensemble_mean_cutpoint_logit": logit_ensemble,
        "subset_ensembles": subsets,
        "coverage_matched_diagnostic": coverage_matched,
        "paired_vs_singles": paired,
        "eval_cli_exit_code": cli.returncode,
    }
    (out_dir / "ensemble_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "mean_of_singles_qwk": mean_single_kappa,
        "single_qwk_min_max": [float(singles_kappas.min()), float(singles_kappas.max())],
        "ensemble_qwk": ensemble["weighted_kappa_covered"],
        "ensemble_coverage": ensemble["coverage"],
        "ensemble_threshold": threshold,
        "ensemble_val_target_met": calibration["target_met"],
        "ensemble_val_under_fnr": calibration["under_fnr"],
        "logit_ensemble_qwk": logit_ensemble["weighted_kappa_covered"],
        "eval_cli_exit_code": cli.returncode,
    }, indent=2))


if __name__ == "__main__":
    main()
