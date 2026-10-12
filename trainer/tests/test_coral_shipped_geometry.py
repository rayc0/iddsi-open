"""Target/decode agreement, off-by-one, and interior reachability at the SHIPPED geometry.

``test_coral_reachability.py`` already pins these properties at the head's random
initialisation and at temperature 1.0.  This module pins them at the geometry the
v0.7 release actually shipped, so a regression cannot hide behind a difference
between "reachable in principle" and "reachable in the artefact".

Constants are read out of the release artefacts by
``lane_runs/def06_coral_audit/scripts/read_cutpoint_geometry.py``:

  ``/work/runs/v07/siglip2_v07_fullcoverage_seed{17,29,43,71,101}/best.pt``
      -> ``head.first_bias``, ``head.bias_gaps_raw``
  ``/work/runs/v07/siglip2_v07_fullcoverage_seed{17,29,43,71,101}/calibration.json``
      -> ``temperature``

Measured there: learned gaps 1.5784-1.6256 against the 1.7016 initialisation, so
every interior level stays reachable; at the fitted temperature the per-class
interior mass ceilings are L4 0.2290-0.2448, L5 0.2228-0.2383, L6 0.2238-0.2394,
while at temperature 1.0 they are 0.3753-0.3854.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from train.losses import ClassBalancedCoralLoss, effective_number_weights, ordinal_targets, smoothed_ordinal_targets
from train.model import (
    LEVELS,
    OrderedCoralHead,
    coral_decode,
    coral_interior_mass_ceiling,
    coral_logits_to_probs,
)

# seed -> (cutpoints, temperature), read from best.pt / calibration.json.
V07_GEOMETRY = {
    17: ([1.553, -0.0684, -1.6472, -3.2354], 1.6817928305074292),
    29: ([1.5577, -0.0634, -1.6424, -3.2273], 1.7411011265922485),
    43: ([1.5553, -0.0686, -1.6474, -3.234], 1.6245047927124712),
    71: ([1.5558, -0.0678, -1.6461, -3.2314], 1.7411011265922485),
    101: ([1.552, -0.0736, -1.6547, -3.241], 1.7411011265922485),
}


def _probs(scores: np.ndarray, cutpoints, temperature: float) -> np.ndarray:
    logits = torch.as_tensor(np.asarray(scores, dtype=np.float64)[:, None] + np.asarray(cutpoints)[None, :], dtype=torch.float32)
    return coral_logits_to_probs(logits / temperature).numpy()


# ---------------------------------------------------------------------------
# 1. the training target and the decode rule agree; no off-by-one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level_index", range(len(LEVELS)))
@pytest.mark.parametrize("smoothing", [0.0, 0.05, 0.1, 0.2])
def test_smoothed_target_still_decodes_to_its_own_level(level_index: int, smoothing: float) -> None:
    """Label smoothing must not move the decode by one level.

    ``t * (1 - eps) + eps / 2`` keeps the target strictly on its own side of 0.5
    for every eps < 1, so the CORAL optimum for label k still decodes to k.  A
    smoothing rule that pushed a target across 0.5 would silently shift a level.
    """
    labels = torch.tensor([level_index])
    targets = smoothed_ordinal_targets(labels, len(LEVELS), smoothing)
    assert ((targets > 0.5) | (targets < 0.5)).all(), "a target landed exactly on 0.5"
    logits = torch.logit(targets).requires_grad_(True)
    with torch.no_grad():
        decoded, _ = coral_decode(coral_logits_to_probs(logits).numpy())
    assert int(decoded[0]) == level_index
    # and the target is exactly the sign vector the rank rule counts
    assert (targets > 0.5).numpy().tolist()[0] == [(level_index > k) for k in range(len(LEVELS) - 1)]


def test_target_vector_is_the_cumulative_sign_pattern_the_decode_reads_back() -> None:
    """``#{j : P(y > j) > 0.5}`` must equal the label for every possible target."""
    for level_index in range(len(LEVELS)):
        target = ordinal_targets(torch.tensor([level_index]), len(LEVELS))
        # a logit vector at the CORAL optimum: +inf where the target is 1, -inf where 0
        logits = torch.where(target > 0.5, torch.tensor(12.0), torch.tensor(-12.0))
        decoded, _ = coral_decode(coral_logits_to_probs(logits).numpy())
        assert int(decoded[0]) == level_index


def test_manifest_level_field_round_trips_without_an_off_by_one() -> None:
    """``level`` 3..7 in the manifest is index 0..4 everywhere downstream."""
    from train.data import ManifestImageDataset

    rows = [{"image_path": f"/tmp/{level}.jpg", "level": level, "split": "test"} for level in LEVELS]
    dataset = ManifestImageDataset(rows, "test", lambda image: image)
    assert dataset.labels == list(range(len(LEVELS)))
    for index, level in enumerate(LEVELS):
        assert LEVELS[index] == level == index + 3
        assert LEVELS[level - LEVELS[0]] == level


def test_loss_target_width_matches_the_head_logit_width() -> None:
    """A head emitting 4 cut-point logits must be trained against 5 levels."""
    head = OrderedCoralHead(in_features=8, num_classes=len(LEVELS), dropout=0.0)
    features = torch.randn(4, 8)
    logits = head(features)
    assert logits.shape == (4, len(LEVELS) - 1)
    labels = torch.tensor([0, 2, 3, 4])
    targets = smoothed_ordinal_targets(labels, logits.shape[1] + 1, 0.0)
    assert targets.shape == logits.shape
    weights = effective_number_weights([0, 0, 1, 2, 3, 4], beta=0.999)
    assert weights.shape == (len(LEVELS),)
    loss = ClassBalancedCoralLoss(weights)(logits, labels)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 2. every interior level is reachable at the shipped geometry
# ---------------------------------------------------------------------------.


@pytest.mark.parametrize("seed", sorted(V07_GEOMETRY))
def test_every_level_reachable_at_the_shipped_cutpoints_and_temperature(seed: int) -> None:
    """The v0.7 release geometry, not a random initialisation."""
    cutpoints, temperature = V07_GEOMETRY[seed]
    scores = np.linspace(-14.0, 14.0, 28001)
    decoded, confidence = coral_decode(_probs(scores, cutpoints, temperature))
    assert sorted({int(v) for v in decoded}) == [0, 1, 2, 3, 4]
    for index in range(len(LEVELS)):
        window = scores[decoded == index]
        assert window.size > 0, f"L{LEVELS[index]} has an empty score window at seed {seed}"
    assert (np.diff(decoded) >= 0).all(), "the decode is not monotone in the score"


@pytest.mark.parametrize("seed", sorted(V07_GEOMETRY))
def test_interior_ceiling_matches_a_brute_force_supremum(seed: int) -> None:
    """``tanh(gap/4)`` is the supremum, checked against a dense score grid.

    The reported ceiling is ``tanh(min_j gap_j / 4T)`` -- the cap on the NARROWEST
    interior class -- which is what a threshold must stay under to keep every
    interior level reachable.  Each class's own cap is checked separately, because
    the shipped gaps differ (L4's gap is the widest in all five seeds, so L4 can
    look more confident than the reported ceiling allows).
    """
    cutpoints, temperature = V07_GEOMETRY[seed]
    scores = np.linspace(-14.0, 14.0, 28001)
    probs = _probs(scores, cutpoints, temperature)
    reported = coral_interior_mass_ceiling(probs)
    gaps = -np.diff(np.asarray(cutpoints))
    expected = float(np.tanh(gaps.min() / (4.0 * temperature)))
    assert reported == pytest.approx(expected, abs=1e-6)

    decoded, confidence = coral_decode(probs)
    for index in (1, 2, 3):
        own_ceiling = float(np.tanh(gaps[index - 1] / (4.0 * temperature)))
        observed = confidence[decoded == index]
        assert observed.size, f"L{LEVELS[index]} unreachable at seed {seed}"
        assert observed.max() <= own_ceiling + 1e-6, (
            f"L{LEVELS[index]} reached {observed.max():.6f} above its own ceiling "
            f"{own_ceiling:.6f}"
        )
        assert observed.max() == pytest.approx(own_ceiling, abs=2e-3), (
            f"L{LEVELS[index]} never approached its own ceiling {own_ceiling:.6f}"
        )
    # the reported cap is the minimum of the three per-class caps, recovered from
    # the float32 probability matrix rather than from the cut-points directly
    assert reported == pytest.approx(min(
        float(np.tanh(gaps[index - 1] / (4.0 * temperature))) for index in (1, 2, 3)
    ), abs=1e-5)


def test_temperature_scaling_is_what_drops_the_interior_ceiling_into_gate_range() -> None:
    """The mechanism behind the shipped L5/L6 collapse, pinned as a number.

    The learned gaps (~1.58) are inherited almost unchanged from the 1.7016
    initialisation, so at temperature 1.0 the interior ceiling is ~0.38.  The
    fitted v0.7 temperature (~1.62-1.74) divides the effective gap and drops it
    to ~0.22-0.24 -- the range a 5-class abstention threshold has to live in.
    """
    cutpoints, temperature = V07_GEOMETRY[43]
    scores = np.linspace(-14.0, 14.0, 28001)
    gaps = -np.diff(np.asarray(cutpoints))
    at_unit_temperature = float(np.tanh(gaps.min() / 4.0))
    at_fitted_temperature = float(np.tanh(gaps.min() / (4.0 * temperature)))
    assert at_unit_temperature > 0.35
    assert at_fitted_temperature < 0.25
    assert at_fitted_temperature < at_unit_temperature / 1.4
    # and the matrix at the fitted temperature reports the fitted ceiling
    assert coral_interior_mass_ceiling(_probs(scores, cutpoints, temperature)) == pytest.approx(
        at_fitted_temperature, abs=1e-6
    )


def test_a_threshold_above_the_interior_ceiling_deletes_every_interior_level() -> None:
    """The abstention-gate twin of H-CORAL, as an executable property.

    A threshold has to clear the WIDEST interior cap, not the reported (minimum)
    one, to black out all three levels at once.  That is the regime the shipped
    v0.7 thresholds sat in: 0.5092 / 0.5189 for seeds 29 and 101 against a widest
    interior cap of 0.2448.
    """
    cutpoints, temperature = V07_GEOMETRY[43]
    scores = np.linspace(-14.0, 14.0, 28001)
    probs = _probs(scores, cutpoints, temperature)
    ceiling = coral_interior_mass_ceiling(probs)
    gaps = -np.diff(np.asarray(cutpoints))
    widest_cap = max(float(np.tanh(gaps[index - 1] / (4.0 * temperature))) for index in (1, 2, 3))
    decoded, confidence = coral_decode(probs)
    interior = np.isin(decoded, [1, 2, 3])
    assert interior.any()
    for threshold in (widest_cap + 1e-3, 0.5, 0.9):
        assert not (interior & (confidence >= threshold)).any(), (
            f"an interior decode survived a threshold of {threshold:.4f} above the "
            f"widest interior cap {widest_cap:.4f}"
        )
    # below the narrowest cap every interior level is still reachable
    accepted = confidence >= (ceiling - 1e-3)
    assert {int(v) for v in decoded[interior & accepted]} == {1, 2, 3}
