"""Tests for eval.abstention — coverage-targeted threshold selection on validation only.

Worked example used throughout (n=10, level codes 0..4 = IDDSI L3..L7, lower =
softer, so ``y_pred < y_true`` is the dangerous direction):

    index     :  0     1     2     3     4     5     6     7     8     9
    confidence:  0.95  0.90  0.85  0.80  0.75  0.70  0.65  0.60  0.55  0.50
    y_true    :   4     4     3     4     2     3     1     0     2     2
    y_pred    :   4     3     3     4     2     2     1     1     0     2

Dangerous-direction samples (y_pred < y_true): index 1 (3<4), index 5 (2<3),
index 8 (0<2) -> 3 total.

At target coverage 70%: k = ceil(0.7 * 10) = 7, threshold = 7th largest
confidence = 0.65, accepted = indices 0..6.  Covered errors: index 1 (3!=4),
index 5 (2!=3) -> error rate 2/7, accuracy 5/7.  Accepted dangerous errors:
indices 1 and 5 -> dangerous FNR = 2/3; index 8 is abstained (caught).

At 80%: threshold 0.60, accepted 0..7, errors 1, 5, 7 (1!=0, safe direction)
-> risk 3/8, dangerous FNR = 2/3.

At 90%: threshold 0.55, accepted 0..8, errors 1, 5, 7, 8 -> risk 4/9,
dangerous FNR = 3/3 = 1.0 (all dangerous errors slip through).
"""

import numpy as np
import pytest

from eval.abstention import (
    render_risk_coverage_table,
    risk_coverage_row,
    select_abstention_thresholds,
    select_threshold_for_coverage,
    split_is_test,
)

CONFIDENCE = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
Y_TRUE = [4, 4, 3, 4, 2, 3, 1, 0, 2, 2]
Y_PRED = [4, 3, 3, 4, 2, 2, 1, 1, 0, 2]


def test_threshold_hits_target_coverage_exactly_on_distinct_confidences():
    for target, expected_threshold in [(0.7, 0.65), (0.8, 0.60), (0.9, 0.55)]:
        selection = select_threshold_for_coverage(CONFIDENCE, target)
        assert selection["threshold"] == expected_threshold
        assert selection["coverage"] == pytest.approx(target)
        assert selection["accepted"] == round(target * 10)


def test_ties_can_only_raise_coverage_within_one_sample():
    # 5th largest of these confidences is 0.6, which ties (two samples at 0.6),
    # so coverage exceeds the 50% target by exactly one sample.
    tied = [0.9, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2]
    selection = select_threshold_for_coverage(tied, 0.5)
    assert selection["threshold"] == pytest.approx(0.6)
    assert selection["coverage"] == pytest.approx(0.6)
    assert selection["accepted"] - 0.5 * 10 <= 1  # within 1 sample of target


def test_risk_coverage_row_hand_computed_example():
    row = risk_coverage_row(Y_TRUE, Y_PRED, CONFIDENCE, threshold=0.65)
    assert row["n"] == 10
    assert row["accepted"] == 7
    assert row["abstained"] == 3
    assert row["coverage"] == pytest.approx(0.7)
    assert row["accuracy"] == pytest.approx(5 / 7)
    assert row["error_rate"] == pytest.approx(2 / 7)
    assert row["dangerous_errors_total"] == 3
    assert row["dangerous_errors_covered"] == 2
    assert row["dangerous_errors_caught_by_abstention"] == 1
    assert row["dangerous_fnr"] == pytest.approx(2 / 3)
    assert row["dangerous_rate_covered"] == pytest.approx(2 / 7)


def test_select_abstention_thresholds_full_table():
    result = select_abstention_thresholds(
        Y_TRUE, Y_PRED, CONFIDENCE, split="validation"
    )
    assert result["split"] == "validation"
    assert result["n"] == 10
    assert result["dangerous_errors_total"] == 3
    assert result["target_coverages"] == [0.7, 0.8, 0.9]
    rows = {row["target_coverage"]: row for row in result["rows"]}

    row70 = rows[0.7]
    assert row70["threshold"] == pytest.approx(0.65)
    assert row70["coverage"] == pytest.approx(0.7)
    assert row70["error_rate"] == pytest.approx(2 / 7)
    assert row70["dangerous_fnr"] == pytest.approx(2 / 3)
    assert row70["dangerous_errors_caught_by_abstention"] == 1

    row80 = rows[0.8]
    assert row80["threshold"] == pytest.approx(0.60)
    assert row80["coverage"] == pytest.approx(0.8)
    assert row80["error_rate"] == pytest.approx(3 / 8)  # errors: 1, 5, 7 (safe direction)
    assert row80["accuracy"] == pytest.approx(5 / 8)
    assert row80["dangerous_fnr"] == pytest.approx(2 / 3)

    row90 = rows[0.9]
    assert row90["threshold"] == pytest.approx(0.55)
    assert row90["coverage"] == pytest.approx(0.9)
    assert row90["error_rate"] == pytest.approx(4 / 9)  # errors: 1, 5, 7, 8
    assert row90["accuracy"] == pytest.approx(5 / 9)
    assert row90["dangerous_fnr"] == pytest.approx(1.0)
    assert row90["dangerous_errors_caught_by_abstention"] == 0

    # Required risk-coverage fields present on every row.
    for row in result["rows"]:
        for field in (
            "target_coverage",
            "threshold",
            "coverage",
            "accepted",
            "abstained",
            "accuracy",
            "error_rate",
            "dangerous_fnr",
        ):
            assert field in row


def test_refuses_any_split_named_test():
    for bad_split in ["test", "TEST", "Test", "external_test", "testset"]:
        with pytest.raises(ValueError, match="refusing to select"):
            select_abstention_thresholds(
                Y_TRUE, Y_PRED, CONFIDENCE, split=bad_split
            )


def test_accepts_validation_like_splits():
    for good_split in ["val", "validation", "dev", "holdout_val"]:
        result = select_abstention_thresholds(
            Y_TRUE, Y_PRED, CONFIDENCE, split=good_split
        )
        assert result["split"] == good_split
        assert result["rows"]


def test_split_is_test_helper():
    assert split_is_test("test")
    assert split_is_test("TEST")
    assert split_is_test("external_test")
    assert not split_is_test("val")
    assert not split_is_test("validation")


def test_deterministic():
    first = select_abstention_thresholds(Y_TRUE, Y_PRED, CONFIDENCE, split="val")
    second = select_abstention_thresholds(Y_TRUE, Y_PRED, CONFIDENCE, split="val")
    assert first == second


def test_no_dangerous_errors_gives_nan_fnr():
    y_true = [2, 2, 2, 2]
    y_pred = [2, 3, 3, 3]  # only safe-direction errors
    result = select_abstention_thresholds(
        y_true, y_pred, [0.9, 0.8, 0.7, 0.6], split="validation", target_coverages=[0.5]
    )
    row = result["rows"][0]
    assert row["dangerous_errors_total"] == 0
    assert np.isnan(row["dangerous_fnr"])


def test_input_validation():
    with pytest.raises(ValueError, match="same length"):
        select_abstention_thresholds([1, 2, 3], [1, 2], [0.9, 0.8, 0.7], split="val")
    with pytest.raises(ValueError, match="non-empty"):
        select_abstention_thresholds([], [], [], split="val")
    with pytest.raises(ValueError, match="finite"):
        select_abstention_thresholds(
            [1, 2], [1, 2], [0.9, float("nan")], split="val"
        )
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        select_threshold_for_coverage([0.9, 0.8], 1.5)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        select_threshold_for_coverage([0.9, 0.8], 0.0)


def test_render_risk_coverage_table_contains_rows():
    result = select_abstention_thresholds(Y_TRUE, Y_PRED, CONFIDENCE, split="val")
    table = render_risk_coverage_table(result)
    assert "target coverage" in table
    assert "dangerous FNR" in table
    assert table.count("\n") >= 4
    for target in ("70%", "80%", "90%"):
        assert target in table
