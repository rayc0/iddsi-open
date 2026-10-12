"""H-CORAL threshold-selection defects (lane def06_coral_audit, 2026-10-12).

``eval/calibration.py:calibrate_abstention_threshold`` picks

    chosen = max(feasible, key=lambda record: (record["coverage"], -record["threshold"]))

so among candidates that tie on coverage it MINIMISES the threshold.  The
candidate list always contains ``0.0`` alongside every confidence value, and
``0.0`` and ``min(confidence)`` both give coverage 1.0.  Whenever full coverage
is feasible the selector therefore returns ``0.0``: an abstention threshold the
gate can never fire, with ``target_met: true`` and the reason string "highest
coverage satisfying the configured validation under-classification gate".

Measured on a calibration set built from the real v0.7 seed-43 head (cutpoints
[1.5553, -0.0686, -1.6474, -3.234], temperature 1.6245047927124712): 60 rows
confidently decoded L7 and 40 confidently decoded L3, under-classification 0.0,
so the selector returns threshold 0.0 at coverage 1.0 and reports the gate as
met.  On that matrix ``min(confidence)`` is 0.949493, so the most conservative
candidate achieving the same coverage is 0.949493.

The geometry constants below are read out of
``/work/runs/v07/siglip2_v07_fullcoverage_seed43/best.pt`` and
``calibration.json`` by ``lane_runs/def06_coral_audit/scripts/
read_cutpoint_geometry.py``; the true per-class interior mass ceilings at the
fitted temperature are L4 0.2448, L5 0.2383, L6 0.2394.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.calibration import calibrate_abstention_threshold
from train.model import coral_decode, coral_interior_mass_ceiling, coral_logits_to_probs

# v0.7 seed-43 head geometry, read out of best.pt + calibration.json.
SEED43_CUTPOINTS = np.array([1.5553, -0.0686, -1.6474, -3.234])
SEED43_TEMPERATURE = 1.6245047927124712


def _matrix(scores: np.ndarray, cutpoints: np.ndarray = SEED43_CUTPOINTS,
            temperature: float = SEED43_TEMPERATURE) -> np.ndarray:
    """A CORAL probability matrix from the real head at the fitted temperature."""
    logits = torch.as_tensor(np.asarray(scores, dtype=np.float64)[:, None] + cutpoints[None, :], dtype=torch.float32)
    return coral_logits_to_probs(logits / temperature).numpy()


def _collapsed_calibration_set() -> tuple[np.ndarray, np.ndarray]:
    """A validation split where every row is a confident, correct tail call."""
    scores = np.concatenate([np.full(60, 8.0), np.full(40, -8.0)])
    labels = np.asarray([4] * 60 + [0] * 40, dtype=np.int64)
    return _matrix(scores), labels


def test_selected_threshold_is_never_one_the_gate_cannot_fire() -> None:
    """A threshold of 0.0 abstains on nothing, so ``unclear`` is never emitted."""
    probabilities, labels = _collapsed_calibration_set()
    decoded, confidence = coral_decode(probabilities)
    assert not np.isin(decoded, [1, 2, 3]).any(), "this fixture must have no interior decode"

    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)

    assert selection["target_met"] is True
    assert selection["coverage"] == pytest.approx(1.0)
    assert selection["under_fnr"] == pytest.approx(0.0)
    # The most conservative candidate that achieves the same coverage.
    most_conservative = float(confidence.min())
    assert most_conservative > 0.5
    assert selection["threshold"] == pytest.approx(most_conservative), (
        f"selector returned {selection['threshold']:.6f} while "
        f"{most_conservative:.6f} achieves the same coverage {selection['coverage']:.4f}; "
        "a threshold of 0.0 disables abstention entirely and still reports "
        f"target_met={selection['target_met']} with reason {selection['selection_reason']!r}"
    )


def test_ties_on_coverage_resolve_to_the_most_conservative_threshold() -> None:
    """The tie-break must raise the threshold, not lower it.

    ``max(feasible, key=(coverage, -threshold))`` picks the smallest threshold
    among equal-coverage candidates.  The sibling branch of the same function
    uses the opposite idiom (``min(..., -coverage)`` to prefer higher coverage),
    so the two halves of one selector disagree about which direction is
    conservative.
    """
    probabilities, labels = _collapsed_calibration_set()
    _, confidence = coral_decode(probabilities)
    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)

    tied = sorted({0.0, float(confidence.min())})
    assert len(tied) == 2 and tied[0] == 0.0
    assert selection["threshold"] == pytest.approx(tied[1]), (
        f"thresholds {tied} both give coverage 1.0; the selector must return the "
        f"higher one, got {selection['threshold']:.6f}"
    )


def test_selection_reason_cannot_claim_a_healthy_gate_at_a_vacuous_threshold() -> None:
    """``target_met: true`` at threshold 0.0 is a false pass on the abstention gate."""
    probabilities, labels = _collapsed_calibration_set()
    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)
    if selection["threshold"] <= 0.0:
        pytest.fail(
            "the selector returned a vacuous threshold while reporting "
            f"target_met={selection['target_met']} / {selection['selection_reason']!r}"
        )


def test_interior_ceiling_is_reported_even_when_no_interior_decode_exists() -> None:
    """The cap is a property of the head, so it must not vanish with the split.

    ``ceiling = coral_interior_mass_ceiling(probabilities) if interior_decoded
    else None`` used to drop the cap exactly when the validation split showed no
    interior decode -- the collapsed regime the guard exists for.  The cap is now
    always computed from the matrix and always reported, so a reviewer can see it
    in ``calibration.json`` and check the threshold against it.
    """
    probabilities, labels = _collapsed_calibration_set()
    recovered = coral_interior_mass_ceiling(probabilities)
    assert recovered is not None, (
        "the interior mass ceiling is recoverable from a tail-only CORAL matrix"
    )
    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)
    assert selection["interior_mass_ceiling"] == pytest.approx(recovered)
    assert selection["interior_decodes"] == 0


def test_reason_string_flags_a_threshold_above_the_ceiling_when_the_clip_is_skipped() -> None:
    """The WARNING must be reachable, and must fire in the collapsed regime.

    The clip had made ``chosen["threshold"] > ceiling`` unreachable, so the
    function had no way to say "I could not protect the interior levels".  When
    the split carries no interior decode the clip is deliberately not applied
    (clipping a tail-only split would leave 0.0 as the only reachable threshold),
    and the reason string then has to say the threshold sits above the cap.
    """
    probabilities, labels = _collapsed_calibration_set()
    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)
    assert selection["interior_mass_ceiling"] is not None
    assert selection["threshold"] > selection["interior_mass_ceiling"]
    assert "WARNING threshold exceeds the CORAL interior mass ceiling" in selection["selection_reason"]


def test_clip_still_applies_when_the_split_has_an_interior_decode() -> None:
    """The reachability clip itself must not regress: it is the H-CORAL guard."""
    rng = np.random.default_rng(3)
    scores = np.concatenate(
        [rng.normal(6.0, 1.0, 50), rng.normal(-5.0, 1.0, 40), rng.uniform(-1.6, 1.6, 20)]
    )
    probabilities = _matrix(scores)
    labels = np.full(len(scores), 4, dtype=np.int64)
    decoded, _ = coral_decode(probabilities)
    assert np.isin(decoded, [1, 2, 3]).any(), "fixture must contain interior decodes"

    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)
    assert selection["interior_mass_ceiling"] is not None
    assert selection["threshold"] <= selection["interior_mass_ceiling"] + 1e-12
    assert "WARNING" not in selection["selection_reason"]
