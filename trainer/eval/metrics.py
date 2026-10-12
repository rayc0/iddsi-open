from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from train.model import coral_decode, _check_decode_rule

LEVELS = np.array([3, 4, 5, 6, 7], dtype=np.int64)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
    mask: np.ndarray | None = None,
    decode: str = "rank",
) -> float:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    if not mask.any():
        return float("nan")
    masked_predictions, confidence = coral_decode(probabilities[mask], decode)
    correct = masked_predictions == labels[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            in_bin = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            in_bin = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if in_bin.any():
            ece += float(in_bin.sum() / total) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    return float(ece)


def risk_coverage_curve(
    probabilities: np.ndarray, labels: np.ndarray, points: int = 51, decode: str = "rank"
) -> list[dict[str, float | int]]:
    predictions, confidence = coral_decode(probabilities, decode)
    thresholds = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, min(points, len(labels) + 1))))
    curve: list[dict[str, float | int]] = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "accepted": count,
                "error_risk": float((predictions[accepted] != labels[accepted]).mean()) if count else float("nan"),
                "underclassification_risk": (
                    float((predictions[accepted] < labels[accepted]).mean()) if count else float("nan")
                ),
            }
        )
    return curve


def per_level_recall(
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    accepted: np.ndarray,
    levels: np.ndarray = LEVELS,
) -> dict[str, dict[str, float]]:
    """Per-level recall with every denominator stated.

    Two denominators are reported per level and they are NOT the same quantity:

    ``support`` / ``recall_all``
        All events whose true label is this level.  An abstention is a miss,
        so this is the recall a reader expects from the word "recall" and the
        one that falls when coverage falls.
    ``covered_support`` / ``recall_covered``
        Only the accepted (non-abstained) events whose true label is this
        level.  This is the recall inside the covered subset the kappa is
        computed on (H-KAPPACOV) and it can rise while ``recall_all`` falls.

    ``emitted_as_decision`` counts how many times the level was actually
    output as a decision anywhere in the split, so a level with recall 0.00
    can be told apart from a level that is never emitted at all.
    """
    labels = np.asarray(labels)
    raw_predictions = np.asarray(raw_predictions)
    accepted = np.asarray(accepted, dtype=bool)
    output: dict[str, dict[str, float]] = {}
    for index, level in enumerate(levels):
        true_mask = labels == index
        support = int(true_mask.sum())
        correct_all = int((true_mask & (raw_predictions == index)).sum())
        covered_true = true_mask & accepted
        covered_support = int(covered_true.sum())
        correct_covered = int((covered_true & (raw_predictions == index)).sum())
        output[f"L{int(level)}"] = {
            "support": support,
            "recall_all": (correct_all / support) if support else float("nan"),
            "covered_support": covered_support,
            "recall_covered": (correct_covered / covered_support) if covered_support else float("nan"),
            "emitted_as_decision": int(((raw_predictions == index) & accepted).sum()),
        }
    return output


def _point_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float, bins: int) -> dict[str, float]:
    raw_predictions, confidence = coral_decode(probabilities)


def _point_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float, bins: int, decode: str = "rank"
) -> dict[str, float]:
    raw_predictions, confidence = coral_decode(probabilities, decode)
    accepted = confidence >= threshold
    predictions = np.where(accepted, raw_predictions, -1)
    covered_true = labels[accepted]
    covered_pred = raw_predictions[accepted]
    kappa = (
        float(cohen_kappa_score(covered_true, covered_pred, weights="quadratic"))
        if len(covered_true) > 1 and len(np.unique(covered_true)) > 1
        else float("nan")
    )
    return {
        "macro_f1_with_abstentions": float(
            f1_score(labels, predictions, labels=sorted(set(int(v) for v in labels)), average="macro", zero_division=0)
        ),
        "weighted_kappa_covered": kappa,
        "dangerous_underclassification_fnr_covered": (
            float((covered_pred < covered_true).mean()) if accepted.any() else float("nan")
        ),
        "dangerous_underclassification_rate_all": float(((raw_predictions < labels) & accepted).mean()),
        "coverage": float(accepted.mean()),
        "abstention_rate": float(1.0 - accepted.mean()),
        "ece_all": expected_calibration_error(probabilities, labels, bins, decode=decode),
        "ece_covered": expected_calibration_error(probabilities, labels, bins, accepted, decode),
    }


def _bootstrap_ci(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    bins: int,
    samples: int,
    seed: int,
    decode: str = "rank",
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    # Stratified resampling preserves rare levels in small held-out sets.
    indices_by_class = [np.flatnonzero(labels == label) for label in range(5)]
    for _ in range(samples):
        parts = [rng.choice(indices, len(indices), replace=True) for indices in indices_by_class if len(indices)]
        sampled = np.concatenate(parts)
        rng.shuffle(sampled)
        metrics = _point_metrics(probabilities[sampled], labels[sampled], threshold, bins, decode)
        for name, value in metrics.items():
            values.setdefault(name, []).append(value)
    output: dict[str, list[float]] = {}
    for name, metric_values in values.items():
        array = np.asarray(metric_values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        output[name] = (
            [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
            if len(finite)
            else [float("nan"), float("nan")]
        )
    return output


def evaluate_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    ece_bins: int = 10,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2026,
    decode: str = "rank",
) -> dict[str, Any]:
    decode = _check_decode_rule(decode)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), 5):
        raise ValueError("probabilities must have shape [N, 5]")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    # A label outside 0..4 has no row in the confusion matrix below, so it would
    # vanish from every count while `n` still claims it: a silent denominator
    # mismatch. Reject it here instead of reporting a matrix that does not sum
    # to the number of evaluated events.
    if labels.size and (labels.min() < 0 or labels.max() > 4):
        bad = sorted({int(v) for v in labels if not 0 <= int(v) <= 4})
        raise ValueError(
            f"labels must be zero-based level indices 0..4 (L3..L7); "
            f"found out-of-range values {bad} in {labels.size} events"
        )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
        # a row of zeros would silently decode to L3 once normalised
        raise ValueError("every probability row must sum to a positive value")
    probabilities = probabilities / row_sums
    raw_predictions, confidence = coral_decode(probabilities, decode)
    predictions = np.where(confidence >= threshold, raw_predictions, -1)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3, 4, -1])
    point = _point_metrics(probabilities, labels, threshold, ece_bins, decode)
    # The decode BEFORE abstention, under whichever rule was selected.  Without
    # it the published confusion matrix cannot distinguish "the decode never
    # produced L5" from "every L5 decode fell below the threshold" — the
    # H-CORAL symptom and its abstention-gate twin look identical in the
    # artifact.
    interior = (raw_predictions >= 1) & (raw_predictions <= 3)
    return {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "decode_rule": decode,
        "metrics": point,
        "decoded_level_counts_pre_abstention": {
            f"L{int(LEVELS[index])}": int((raw_predictions == index).sum()) for index in range(5)
        },
        # argmax over the same reconstructed categorical.  The rank rule and
        # argmax agree only on the two tail classes, so a level that argmax
        # never returns is a *representation* failure (the interior mass is a
        # difference of adjacent sigmoids and is capped at tanh(gap/4)) rather
        # than a decode-rule choice.  Reported beside the rank counts so the
        # two can be told apart without re-deriving either.
        "argmax_level_counts": {
            f"L{int(LEVELS[index])}": int((probabilities.argmax(axis=1) == index).sum()) for index in range(5)
        },
        "per_level_recall": per_level_recall(labels, raw_predictions, confidence >= threshold),
        "interior_decodes_total": int(interior.sum()),
        "interior_decodes_accepted": int((interior & (confidence >= threshold)).sum()),
        "interior_levels_reachable": bool((interior & (confidence >= threshold)).any()),
        "confidence_intervals_95": _bootstrap_ci(
            probabilities, labels, threshold, ece_bins, bootstrap_samples, bootstrap_seed, decode
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_rows": ["true_L3", "true_L4", "true_L5", "true_L6", "true_L7", "true_unclear"],
        "confusion_columns": ["pred_L3", "pred_L4", "pred_L5", "pred_L6", "pred_L7", "pred_unclear"],
        "risk_coverage_curve": risk_coverage_curve(probabilities, labels, decode=decode),
    }
