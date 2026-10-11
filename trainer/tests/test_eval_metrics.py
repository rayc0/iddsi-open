"""Contract tests for eval.metrics and eval.report.

These pin the metric *denominators*, which is the part of this harness that is
easy to get quietly wrong: a count that disagrees with the declared number of
events turns into a published number nobody can reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import (
    _bootstrap_ci,
    _point_metrics,
    evaluate_predictions,
    expected_calibration_error,
    risk_coverage_curve,
)
from eval.report import render_markdown_report


def _probabilities(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(5), size=n)


def test_evaluate_predictions_rejects_out_of_range_labels() -> None:
    """A label with no confusion-matrix row must not be silently dropped.

    `evaluate_predictions` builds the matrix over ``labels=[0, 1, 2, 3, 4, -1]``,
    so a label of 9 used to vanish from every count while ``n`` still claimed
    the event: the reported n and the matrix total disagreed.
    """
    probabilities = _probabilities(6)
    for bad in (np.array([0, 1, 2, 3, 4, 9]), np.array([0, 1, 2, 3, 4, -1])):
        with pytest.raises(ValueError, match="zero-based level indices 0..4"):
            evaluate_predictions(probabilities, bad, 0.0, bootstrap_samples=0)
    # The legitimate range still works.
    evaluate_predictions(_probabilities(5), np.arange(5) % 5, 0.0, bootstrap_samples=0)


def test_confusion_matrix_totals_match_the_declared_event_count() -> None:
    """Every evaluated event must appear exactly once in the confusion matrix."""
    rng = np.random.default_rng(5)
    for n in (1, 7, 40):
        probabilities = _probabilities(n, seed=n)
        labels = rng.integers(0, 5, size=n)
        result = evaluate_predictions(probabilities, labels, 0.5, bootstrap_samples=0)
        assert result["n"] == n
        assert sum(sum(row) for row in result["confusion_matrix"]) == n


def test_report_renders_every_confusion_row_so_the_table_sums_to_n() -> None:
    """The markdown table must not drop a row the metrics object carries.

    The renderer sliced ``confusion_matrix[:5]``, so a non-empty ``true_unclear``
    row was printed nowhere while ``metrics.json`` kept it and the "Events: n"
    line above still counted it.
    """
    result = {
        "n": 6,
        "threshold": 0.5,
        "metrics": {"macro_f1_with_abstentions": 0.5},
        "confidence_intervals_95": {},
        "confusion_matrix": [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        "confusion_rows": [
            "true_L3", "true_L4", "true_L5", "true_L6", "true_L7", "true_unclear",
        ],
        "confusion_columns": [
            "pred_L3", "pred_L4", "pred_L5", "pred_L6", "pred_L7", "pred_unclear",
        ],
    }
    report = render_markdown_report(result, title="T", synthetic=False)
    data_rows = [line for line in report.splitlines() if line.startswith("| L") or line.startswith("| unclear")]
    assert len(data_rows) == 6, f"expected 6 rendered rows, got {len(data_rows)}"
    table_total = sum(
        int(cell) for line in data_rows for cell in line.split("|")[2:] if cell.strip()
    )
    assert table_total == result["n"]
    assert "unclear" in report


def test_dangerous_direction_is_underclassification_only() -> None:
    """Predicted softer than tested is dangerous; predicted thicker is not.

    This is the semantic the abstention gate and the FNR metrics are built on,
    so it is pinned here rather than left implicit in a sign. A one-hot at index
    k decodes to level k under the ordinal rank rule.
    """
    labels = np.array([2, 2, 2, 2])
    probabilities = np.zeros((4, 5))
    probabilities[0, 1] = 1.0  # predicted 1 < tested 2 -> dangerous
    probabilities[1, 2] = 1.0  # exact
    probabilities[2, 3] = 1.0  # predicted 3 > tested 2 -> quality issue only
    probabilities[3, 2] = 1.0  # exact
    result = evaluate_predictions(probabilities, labels, 0.0, bootstrap_samples=0)
    covered = result["metrics"]
    # Only the under-classifying sample counts as dangerous.
    assert covered["dangerous_underclassification_fnr_covered"] == pytest.approx(0.25)
    # The over-classifying sample is an ordinary error, not a dangerous one.
    assert covered["macro_f1_with_abstentions"] < 1.0


def test_macro_f1_counts_abstentions_as_misses() -> None:
    """An abstained event must hurt the true class it belonged to."""
    probabilities = _probabilities(10, seed=2)
    labels = np.arange(10) % 5
    high = evaluate_predictions(probabilities, labels, 0.0, bootstrap_samples=0)
    low = evaluate_predictions(probabilities, labels, 1.0, bootstrap_samples=0)
    assert high["metrics"]["coverage"] == 1.0
    assert low["metrics"]["coverage"] == 0.0
    assert low["metrics"]["abstention_rate"] == 1.0
    assert (
        low["metrics"]["macro_f1_with_abstentions"]
        <= high["metrics"]["macro_f1_with_abstentions"]
    )


def test_expected_calibration_error_masks_and_handles_empty_mask() -> None:
    probabilities = _probabilities(12, seed=3)
    labels = np.arange(12) % 5
    assert expected_calibration_error(probabilities, labels) >= 0.0
    covered = np.zeros(12, dtype=bool)
    covered[:6] = True
    partial = expected_calibration_error(probabilities, labels, mask=covered)
    assert partial >= 0.0
    assert np.isnan(expected_calibration_error(probabilities, labels, mask=np.zeros(12, dtype=bool)))


def test_bootstrap_intervals_are_deterministic_and_cover_every_point_metric() -> None:
    probabilities = _probabilities(25, seed=4)
    labels = np.arange(25) % 5
    first = _bootstrap_ci(probabilities, labels, 0.3, 10, 20, 11)
    second = _bootstrap_ci(probabilities, labels, 0.3, 10, 20, 11)
    other = _bootstrap_ci(probabilities, labels, 0.3, 10, 20, 12)
    assert first == second
    assert first != other
    assert set(first) == set(_point_metrics(probabilities, labels, 0.3, 10))
    for low, high in first.values():
        assert low <= high


def test_bootstrap_resampling_is_stratified_by_level() -> None:
    """Rare levels must survive the resample, or their CI is meaningless."""
    probabilities = _probabilities(30, seed=6)
    labels = np.array([0] * 26 + [1] * 4)
    intervals = _bootstrap_ci(probabilities, labels, 0.3, 10, 10, 1)
    assert intervals  # level 1 has 4 of 30 events and still produces intervals


def test_risk_coverage_curve_spans_full_coverage() -> None:
    probabilities = _probabilities(30, seed=7)
    labels = np.arange(30) % 5
    curve = risk_coverage_curve(probabilities, labels, points=11)
    assert curve
    assert max(row["coverage"] for row in curve) == pytest.approx(1.0)
    assert min(row["coverage"] for row in curve) < 1.0
    for row in curve:
        if row["accepted"]:
            assert 0.0 <= row["error_risk"] <= 1.0


def test_evaluate_predictions_rejects_non_finite_and_negative_probabilities() -> None:
    probabilities = _probabilities(4, seed=8)
    labels = np.arange(4) % 5
    broken = probabilities.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_predictions(broken, labels, 0.5, bootstrap_samples=0)
    negative = probabilities.copy()
    negative[0, 0] = -0.1
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_predictions(negative, labels, 0.5, bootstrap_samples=0)
