from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from train.model import coral_decode

LEVELS = np.array([3, 4, 5, 6, 7], dtype=np.int64)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10, mask: np.ndarray | None = None
) -> float:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    if not mask.any():
        return float("nan")
    masked_predictions, confidence = coral_decode(probabilities[mask])
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


def risk_coverage_curve(probabilities: np.ndarray, labels: np.ndarray, points: int = 51) -> list[dict[str, float | int]]:
    predictions, confidence = coral_decode(probabilities)
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


def _point_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float, bins: int) -> dict[str, float]:
    raw_predictions, confidence = coral_decode(probabilities)
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
        "ece_all": expected_calibration_error(probabilities, labels, bins),
        "ece_covered": expected_calibration_error(probabilities, labels, bins, accepted),
    }


def _bootstrap_ci(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    bins: int,
    samples: int,
    seed: int,
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
        metrics = _point_metrics(probabilities[sampled], labels[sampled], threshold, bins)
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
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), 5):
        raise ValueError("probabilities must have shape [N, 5]")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    raw_predictions, confidence = coral_decode(probabilities)
    predictions = np.where(confidence >= threshold, raw_predictions, -1)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3, 4, -1])
    point = _point_metrics(probabilities, labels, threshold, ece_bins)
    return {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "metrics": point,
        "confidence_intervals_95": _bootstrap_ci(
            probabilities, labels, threshold, ece_bins, bootstrap_samples, bootstrap_seed
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_rows": ["true_L3", "true_L4", "true_L5", "true_L6", "true_L7", "true_unclear"],
        "confusion_columns": ["pred_L3", "pred_L4", "pred_L5", "pred_L6", "pred_L7", "pred_unclear"],
        "risk_coverage_curve": risk_coverage_curve(probabilities, labels),
    }
