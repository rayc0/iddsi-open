"""Select abstention thresholds on VALIDATION data only and report risk-coverage.

This module is the coverage-targeted counterpart of
:func:`eval.calibration.calibrate_abstention_threshold` (which selects a
threshold subject to a dangerous-error gate).  Here the coverage is the
constraint: for each target coverage (default 70/80/90%) we pick the most
conservative confidence threshold that still covers at least the target
fraction of validation samples, then report a risk-coverage row.

Dangerous direction (grounded in this repository):
    ``trainer/eval/report.py`` states "Dangerous direction means predicting a
    lower numeric IDDSI level than the physical-test label", and
    ``paper/paper.md`` section 6 names the metric "Dangerous-direction FNR
    (under-classification: predicted softer than tested)".  A patient served
    food thinner than the tested level may aspirate, so ``y_pred < y_true``
    is the dangerous error direction; ``y_pred > y_true`` is merely a quality
    issue.  Level codes are ordinal and must be ordered soft -> thick.

Dangerous-direction FNR, as implemented here (precise definition):
    Treat abstention as a detector of dangerous errors.  A sample is a
    *dangerous-direction sample* when its raw (pre-abstention) prediction
    under-classifies, i.e. ``y_pred < y_true``.  Abstention "catches" such a
    sample when it abstains on it (``confidence < threshold``).  The
    dangerous-direction FNR at a threshold is

        dangerous_fnr = |{i : y_pred[i] < y_true[i] and confidence[i] >= threshold}|
                        / |{i : y_pred[i] < y_true[i]}|

    i.e. the fraction of dangerous-direction samples NOT caught by
    abstention (dangerous errors that slip through the filter and are
    confidently served to the patient).  It is ``nan`` when the split
    contains no dangerous-direction samples.  This is distinct from
    ``dangerous_underclassification_fnr_covered`` in ``eval.metrics``
    (dangerous errors over ALL covered samples) — both are reported here.

Safety rule:
    Threshold selection must NEVER run on test data.  If the split name
    contains "test" (case-insensitive substring), every entry point raises
    ``ValueError`` instead of returning anything.

All functions are deterministic (sorting only, no RNG, no network, no
model); they operate purely on per-sample numpy arrays.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: Default target coverages for threshold selection (fleet task W33).
DEFAULT_TARGET_COVERAGES: tuple[float, ...] = (0.70, 0.80, 0.90)


def split_is_test(split: str) -> bool:
    """Return True when the split name contains "test" (case-insensitive)."""

    return "test" in str(split).lower()


def _refuse_if_test(split: str) -> None:
    if split_is_test(split):
        raise ValueError(
            f"refusing to select abstention thresholds on split {split!r}: "
            "thresholds may only be selected on validation data, never on any "
            "test split (a split whose name contains 'test'). Evaluate on test "
            "only with thresholds frozen in advance on validation data."
        )


def _validate_arrays(
    y_true: Sequence[float], y_pred: Sequence[float], confidence: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.ndim != 1 or confidence.ndim != 1:
        raise ValueError("y_true, y_pred and confidence must be 1-D per-sample arrays")
    n = y_true.size
    if n == 0:
        raise ValueError("inputs must be non-empty")
    if not (y_pred.size == confidence.size == n):
        raise ValueError("y_true, y_pred and confidence must have the same length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must be finite level codes")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence scores must be finite")
    return y_true, y_pred, confidence


def select_threshold_for_coverage(
    confidence: Sequence[float], target_coverage: float
) -> dict[str, float | int]:
    """Pick the most conservative threshold covering at least ``target_coverage``.

    With ``n`` samples and ``k = ceil(target_coverage * n)``, the threshold is
    the ``k``-th largest confidence; accepting ``confidence >= threshold``
    therefore covers the ``k`` most confident samples.  With distinct
    confidences the achieved coverage is exactly ``k / n`` (the smallest
    coverage >= target); ties at the threshold are all accepted, which can
    only raise coverage above the target.
    """

    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    if confidence.size == 0:
        raise ValueError("confidence must be non-empty")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence scores must be finite")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must lie in (0, 1]")
    n = confidence.size
    k = int(np.ceil(target_coverage * n))
    k = min(max(k, 1), n)
    threshold = float(np.sort(confidence)[n - k])  # k-th largest confidence
    accepted = int((confidence >= threshold).sum())
    return {
        "n": n,
        "k": k,
        "threshold": threshold,
        "accepted": accepted,
        "coverage": accepted / n,
    }


def risk_coverage_row(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    confidence: Sequence[float],
    threshold: float,
) -> dict[str, float | int]:
    """One risk-coverage row at a fixed threshold (no split check; pure).

    Fields: coverage / accepted / abstained, accuracy and error_rate (risk)
    on covered samples, and the dangerous-direction FNR defined in the
    module docstring plus its raw counts.
    """

    y_true, y_pred, confidence = _validate_arrays(y_true, y_pred, confidence)
    accepted = confidence >= threshold
    n = int(y_true.size)
    accepted_count = int(accepted.sum())
    if accepted_count:
        incorrect = y_pred[accepted] != y_true[accepted]
        error_rate = float(incorrect.mean())
        accuracy = float((~incorrect).mean())
        dangerous_rate_covered = float((y_pred[accepted] < y_true[accepted]).mean())
    else:
        error_rate = accuracy = dangerous_rate_covered = float("nan")
    dangerous = y_pred < y_true
    dangerous_total = int(dangerous.sum())
    dangerous_covered = int((dangerous & accepted).sum())
    dangerous_caught = dangerous_total - dangerous_covered
    dangerous_fnr = (
        dangerous_covered / dangerous_total if dangerous_total else float("nan")
    )
    return {
        "threshold": float(threshold),
        "coverage": accepted_count / n,
        "accepted": accepted_count,
        "abstained": n - accepted_count,
        "n": n,
        "accuracy": accuracy,
        "error_rate": error_rate,
        # Fraction of dangerous-direction samples (y_pred < y_true) that are
        # NOT caught by abstention, i.e. accepted despite under-classifying.
        "dangerous_fnr": dangerous_fnr,
        "dangerous_errors_total": dangerous_total,
        "dangerous_errors_covered": dangerous_covered,
        "dangerous_errors_caught_by_abstention": dangerous_caught,
        # Dangerous errors over all covered samples (matches the
        # `dangerous_underclassification_fnr_covered` denominator in
        # eval.metrics.evaluate_predictions).
        "dangerous_rate_covered": dangerous_rate_covered,
    }


def select_abstention_thresholds(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    confidence: Sequence[float],
    *,
    split: str,
    target_coverages: Sequence[float] = DEFAULT_TARGET_COVERAGES,
) -> dict[str, Any]:
    """Select thresholds on VALIDATION data for each target coverage.

    Refuses (raises ``ValueError``) when ``split`` contains "test"
    (case-insensitive).  Returns ``{"split", "n", "dangerous_errors_total",
    "rows": [risk_coverage_row(...) + {"target_coverage", "k"} per target]}``.
    """

    _refuse_if_test(split)
    y_true, y_pred, confidence = _validate_arrays(y_true, y_pred, confidence)
    targets = [float(target) for target in target_coverages]
    if not targets:
        raise ValueError("target_coverages must be non-empty")
    for target in targets:
        if not 0.0 < target <= 1.0:
            raise ValueError("each target coverage must lie in (0, 1]")
    rows: list[dict[str, Any]] = []
    for target in targets:
        selection = select_threshold_for_coverage(confidence, target)
        row = risk_coverage_row(y_true, y_pred, confidence, selection["threshold"])
        row["target_coverage"] = target
        row["k"] = selection["k"]
        rows.append(row)
    dangerous_total = int((y_pred < y_true).sum())
    return {
        "split": str(split),
        "n": int(y_true.size),
        "dangerous_errors_total": dangerous_total,
        "target_coverages": targets,
        "rows": rows,
    }


def render_risk_coverage_table(result: dict[str, Any]) -> str:
    """Render a select_abstention_thresholds result as a markdown table."""

    lines = [
        f"Abstention risk-coverage (split: {result['split']}, n={result['n']}, "
        f"dangerous-direction samples={result['dangerous_errors_total']})",
        "",
        "| target coverage | threshold | coverage | accepted | abstained | "
        "accuracy | risk (error rate) | dangerous FNR | dangerous caught/total |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        caught = row["dangerous_errors_caught_by_abstention"]
        total = row["dangerous_errors_total"]
        lines.append(
            f"| {row['target_coverage']:.0%} | {row['threshold']:.4f} | "
            f"{row['coverage']:.4f} | {row['accepted']} | {row['abstained']} | "
            f"{row['accuracy']:.4f} | {row['error_rate']:.4f} | "
            f"{row['dangerous_fnr']:.4f} | {caught}/{total} |"
        )
    return "\n".join(lines)
