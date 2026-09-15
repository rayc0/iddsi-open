from __future__ import annotations

from typing import Any

import numpy as np

from train.model import coral_decode, coral_logits_to_probs


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature_min: float = 0.25,
    temperature_max: float = 4.0,
    steps: int = 81,
) -> tuple[float, float]:
    """Fit one scalar temperature by deterministic log-grid NLL search."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits and labels have incompatible shapes")
    temperatures = np.geomspace(temperature_min, temperature_max, steps)
    losses = []
    for temperature in temperatures:
        probabilities = softmax(logits / temperature)
        losses.append(float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12)).mean()))
    best = int(np.argmin(losses))
    return float(temperatures[best]), float(losses[best])


def calibrate_abstention_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_under_fnr: float,
    max_abstention: float,
) -> dict[str, Any]:
    """Choose the highest-coverage validation threshold satisfying the declared risk gate.

    If the validation data cannot satisfy the gate at the minimum required coverage,
    this returns the least-risk eligible candidate and explicitly marks target_met false.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or not len(labels):
        raise ValueError("probabilities must be a non-empty [N, C] matrix matching labels")
    predictions, confidences = coral_decode(probabilities)
    min_coverage = 1.0 - max_abstention
    candidates = np.unique(np.concatenate(([0.0], confidences, [np.nextafter(confidences.max(), np.inf)])))
    records: list[dict[str, float]] = []
    for threshold in candidates:
        accepted = confidences >= threshold
        coverage = float(accepted.mean())
        if not accepted.any():
            under_fnr = float("nan")
            error_rate = float("nan")
        else:
            under_fnr = float((predictions[accepted] < labels[accepted]).mean())
            error_rate = float((predictions[accepted] != labels[accepted]).mean())
        records.append(
            {"threshold": float(threshold), "coverage": coverage, "under_fnr": under_fnr, "error_rate": error_rate}
        )
    eligible = [record for record in records if record["coverage"] + 1e-12 >= min_coverage]
    feasible = [record for record in eligible if record["under_fnr"] <= target_under_fnr]
    if feasible:
        chosen = max(feasible, key=lambda record: (record["coverage"], -record["threshold"]))
        reason = "highest coverage satisfying the configured validation under-classification gate"
        target_met = True
    else:
        chosen = min(eligible, key=lambda record: (record["under_fnr"], record["error_rate"], -record["coverage"]))
        reason = "gate not met; least under-classification risk among candidates meeting minimum coverage"
        target_met = False
    return {
        **chosen,
        "target_under_fnr": float(target_under_fnr),
        "minimum_coverage": float(min_coverage),
        "target_met": target_met,
        "selection_reason": reason,
    }



def fit_coral_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature_min: float = 0.25,
    temperature_max: float = 4.0,
    steps: int = 81,
) -> tuple[float, float]:
    """Fit one scalar temperature on the CORAL cut-point logits.

    ``fit_temperature`` re-softmaxes ``log(p)``, which is monotone for argmax
    but NOT for the ordinal rank decode: rescaling the reconstructed
    categorical changes the cumulative sums and therefore the decoded level.
    Scaling the cut-point logits keeps the ordinal structure intact.
    """
    import torch

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits and labels have incompatible shapes")
    temperatures = np.geomspace(temperature_min, temperature_max, steps)
    losses = []
    for temperature in temperatures:
        probs = coral_logits_to_probs(torch.from_numpy(logits / temperature)).numpy()
        losses.append(float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-12)).mean()))
    best = int(np.argmin(losses))
    return float(temperatures[best]), float(losses[best])
