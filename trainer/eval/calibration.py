from __future__ import annotations

from typing import Any

import numpy as np

from train.model import (
    _check_decode_rule,
    coral_decode,
    coral_interior_mass_ceiling,
    coral_logits_to_probs,
)


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
    decode: str = "rank",
) -> dict[str, Any]:
    """Choose the highest-coverage validation threshold satisfying the declared risk gate.

    If the validation data cannot satisfy the gate at the minimum required coverage,
    this returns the least-risk eligible candidate and explicitly marks target_met false.

    H-CORAL reachability constraint
    ------------------------------
    A CORAL interior class (L4/L5/L6) is a difference of two adjacent
    sigmoids, so its mass is capped at ``tanh(gap / 4)`` (see
    :func:`train.model.coral_interior_mass_ceiling`) while the two tail
    classes are unbounded.  A threshold above that cap therefore abstains on
    *every* interior decode: the published confusion matrix then shows only
    L3 and L7, which is the H-CORAL failure mode arriving through the
    abstention gate instead of through argmax.  Measured on the five v0.7
    release seeds, the unfixed selector returned 0.5189 / 0.2327 / 0.5092 /
    0.2394 / 0.2286 against caps of 0.2232 / 0.2305 / 0.2229 / 0.2383 /
    0.2228, and reported ``target_met: true`` while 0 / 3 / 0 / 2 / 0 of
    20 / 22 / 22 / 18 / 23 interior decodes survived.

    So when the decode actually produced interior levels, the candidate
    thresholds are clipped to the reachable region ``threshold <= cap``.  This
    does not weaken the under-classification gate: the gate is still evaluated
    over the same candidates and ``target_met`` still reports the outcome.  It
    only removes the region where the gate can be "satisfied" by deleting the
    interior levels from the output entirely.

    Tie-break direction
    -------------------
    Among candidates that tie on coverage the most CONSERVATIVE one is kept -- the
    highest threshold.  ``0.0`` and ``min(confidence)`` both give coverage 1.0,
    and the previous ``-record["threshold"]`` key picked ``0.0`` between them: a
    threshold the abstention gate can never fire, reported as
    ``target_met: true`` with the reason "highest coverage satisfying the
    configured validation under-classification gate".  The sibling branch of this
    same function already uses the conservative idiom (``-coverage`` inside a
    ``min`` to prefer higher coverage), so the two halves now agree.  Measured on
    a collapsed v0.7 seed-43 calibration matrix (60 confident L7 rows, 40
    confident L3 rows, under-classification 0.0), the old key returned 0.0 at
    coverage 1.0; the corrected key returns 0.949493 at the same coverage.

    Ceiling availability
    --------------------
    The cap is a property of the head's learned cut-points, not of which rows
    happen to land in the validation split, so it is computed and reported
    whenever the matrix is a CORAL reconstruction.  ``coral_interior_mass_ceiling``
    recovers it even from a tail-only matrix, so a fully collapsed model -- every
    validation row in the L3 or L7 tail, which is the regime the guard exists for
    -- still reports it.  The CLIP still only applies when the split contains an
    interior decode: clipping a tail-only split would delete every candidate
    above the cap and leave 0.0 as the only reachable threshold, which is the
    vacuous gate the tie-break fix above removes.  In that regime the cap is
    therefore disclosed rather than enforced, and the ``WARNING`` below -- which
    the clip had made unreachable -- fires so the reason string cannot read as a
    healthy gate while sitting above the cap.
    """
    decode = _check_decode_rule(decode)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or not len(labels):
        raise ValueError("probabilities must be a non-empty [N, C] matrix matching labels")
    predictions, confidences = coral_decode(probabilities, decode)
    num_classes = probabilities.shape[1]
    interior = (predictions >= 1) & (predictions <= num_classes - 2)
    interior_decoded = int(interior.sum())
    # Reported unconditionally (see "Ceiling availability"); enforced only when
    # there is an interior decode in this split to protect.
    ceiling = coral_interior_mass_ceiling(probabilities)
    min_coverage = 1.0 - max_abstention
    candidates = np.unique(np.concatenate(([0.0], confidences, [np.nextafter(confidences.max(), np.inf)])))
    if ceiling is not None and interior_decoded:
        reachable = candidates[candidates <= ceiling + 1e-12]
        if reachable.size:
            candidates = reachable
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
        # Highest coverage, then the HIGHEST threshold among the ties: a lower
        # threshold buys no extra coverage on this matrix and filters nothing on
        # any other.  0.0 is a candidate, so the old `-threshold` key could hand
        # back a gate that can never fire.
        chosen = max(feasible, key=lambda record: (record["coverage"], record["threshold"]))
        reason = "highest coverage satisfying the configured validation under-classification gate"
        target_met = True
    else:
        chosen = min(eligible, key=lambda record: (record["under_fnr"], record["error_rate"], -record["coverage"]))
        reason = "gate not met; least under-classification risk among candidates meeting minimum coverage"
        target_met = False
    if ceiling is not None and chosen["threshold"] > ceiling + 1e-12:
        reason += "; WARNING threshold exceeds the CORAL interior mass ceiling"
    return {
        **chosen,
        "target_under_fnr": float(target_under_fnr),
        "minimum_coverage": float(min_coverage),
        "target_met": target_met,
        "selection_reason": reason,
        "interior_mass_ceiling": ceiling,
        "interior_decodes": interior_decoded,
        "interior_levels_reachable": bool(
            interior_decoded and (confidences[interior] >= chosen["threshold"]).any()
        ),
        "decode_rule": decode,
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
