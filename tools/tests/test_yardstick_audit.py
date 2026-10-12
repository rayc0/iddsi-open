#!/usr/bin/env python3
"""Tests for tools/yardstick_audit.py (lane an01_yardstick).

Pins the from-scratch quadratic weighted kappa and the CORAL rank decode that
the H-YARDSTICK re-derivation relies on, and anchors them to values that were
published before this audit ran (``results/v07_release/v07_seed*_metrics.json``
and ``results/v05_release/v04_scored.json``), so a regression in either the
audit or the release pipeline shows up as a test failure rather than as a
silently different number.

Run from repo root:
    python -m pytest tools/tests/test_yardstick_audit.py -q
or, without pytest:
    python tools/tests/test_yardstick_audit.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.yardstick_audit import coral_decode, quadratic_weighted_kappa  # noqa: E402


# --------------------------------------------------------------------------- #
# quadratic weighted kappa against hand-computed reference values
# --------------------------------------------------------------------------- #
def test_qwk_perfect_agreement_is_one():
    assert quadratic_weighted_kappa([0, 1], [0, 1]) == 1.0


def test_qwk_hand_computed_two_classes():
    # y_true=[0,0,1,1], y_pred=[0,1,1,1]
    #   O = [[.25,.25],[0,.5]], W = [[0,1],[1,0]]  ->  sum(W*O) = .25
    #   ht = [.5,.5], hp = [.25,.75]  ->  E = [[.125,.375],[.125,.375]]
    #   sum(W*E) = .5  ->  kappa = 1 - .25/.5 = .5
    assert quadratic_weighted_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == 0.5


def test_qwk_hand_computed_three_classes():
    # y_true=[0,0,1,1,2,2], y_pred=[0,1,1,2,2,2]
    #   O/6 = [[1/6,1/6,0],[0,1/6,1/6],[0,0,2/6]], W_ij = ((i-j)/2)^2
    #   sum(W*O) = (0.25 + 0.25)/6 = 1/12
    #   ht = [1/3]*3, hp = [1/6,2/6,3/6]  ->  sum(W*E) = 1/3
    #   kappa = 1 - (1/12)/(1/3) = 3/4
    got = quadratic_weighted_kappa([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2])
    assert abs(got - 0.75) < 1e-12, got


def test_qwk_is_undefined_for_a_single_label():
    assert np.isnan(quadratic_weighted_kappa([0, 0], [0, 0]))
    assert np.isnan(quadratic_weighted_kappa([0], [1]))


def test_qwk_uses_the_union_of_observed_labels():
    # sklearn's labels=None behaviour: the label set is the union, so a
    # prediction outside the true label set is still a category.
    a = quadratic_weighted_kappa([0, 0, 1, 1], [0, 1, 1, 2])
    assert 0.0 <= a <= 1.0


def test_qwk_penalises_distance():
    near = quadratic_weighted_kappa([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2])
    far = quadratic_weighted_kappa([0, 0, 1, 1, 2, 2], [2, 2, 1, 1, 0, 0])
    assert near > far


# --------------------------------------------------------------------------- #
# CORAL rank decode
# --------------------------------------------------------------------------- #
def test_coral_decode_uses_the_rank_rule_not_argmax():
    # A one-hot distribution decodes to that class under both rules.
    probs = np.eye(5, dtype=np.float64)
    predictions, confidence = coral_decode(probs)
    assert predictions.tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(confidence, 1.0)


def test_coral_decode_interior_class_is_reachable():
    # The rank rule returns an interior index when P(y>k) > 0.5 holds for
    # exactly two cut-points, even though no single class has the largest mass.
    probs = np.array(
        [
            [0.05, 0.10, 0.10, 0.10, 0.65],  # survival = [.95,.85,.75,.65] -> 4
            [0.65, 0.10, 0.10, 0.10, 0.05],  # survival = [.35,.25,.15,.05] -> 0
            [0.30, 0.05, 0.30, 0.30, 0.05],  # survival = [.70,.65,.35,.05] -> 2
        ]
    )
    predictions, _ = coral_decode(probs)
    assert predictions.tolist() == [4, 0, 2]


def test_coral_decode_normalises_rows():
    probs = np.array([[2.0, 1.0, 1.0, 1.0, 1.0]])
    predictions, confidence = coral_decode(probs)
    assert confidence[0] <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# anchors: values published before this audit ran
# --------------------------------------------------------------------------- #
def test_anchors_reproduce_published_gate_metric():
    """Re-derive each run's published wk1 from its own prediction file.

    This is the control for the whole audit: if the from-scratch quadratic
    weighted kappa or the re-implemented CORAL decode drifted from the released
    pipeline, this fails. Skipped when the run tree is not present (the
    prediction files live outside the repo, under ``runs/``).
    """
    # The prediction files live in the shared tree's runs/, not in this clone.
    # Discover it (tools/paths.py) rather than naming an absolute host path.
    from paths import find_root

    _root = find_root()
    default_runs = _root / "runs" if _root is not None else _REPO_ROOT / "runs"
    runs_root = Path(os.environ.get("IDDSI_RUNS_ROOT", str(default_runs)))
    if not runs_root.is_dir():
        print(f"    (skipped: {runs_root} not present)")
        return
    patterns = {
        "v0.4": "v04/siglip2_v04_consensus_seed{seed}",
        "v0.5": "v05/siglip2_v05_clean_seed{seed}",
        "v0.7": "v07/siglip2_v07_fullcoverage_seed{seed}",
    }
    checked = 0
    for version, pattern in patterns.items():
        for seed in ("17", "29", "43", "71", "101"):
            run_dir = runs_root / pattern.format(seed=seed)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            threshold = float(
                json.loads((run_dir / "calibration.json").read_text())["threshold"]
            )
            with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as arrays:
                probabilities = arrays["probabilities"].astype(np.float64)
                labels = arrays["labels"].astype(np.int64)
            raw, confidence = coral_decode(probabilities)
            accepted = confidence >= threshold
            got = quadratic_weighted_kappa(labels[accepted], raw[accepted])
            published = metrics["metrics"]["weighted_kappa_covered"]
            assert abs(got - published) < 1e-12, (version, seed, got, published)
            checked += 1
    assert checked == 15, checked


def _main() -> int:
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except AssertionError as exc:  # pragma: no cover - reporting path
                failures += 1
                print(f"FAIL {name}: {exc}")
            else:
                print(f"ok   {name}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
