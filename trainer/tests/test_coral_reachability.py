"""H-CORAL reachability audit (lane qa04_coral, 2026-10-12).

``test_coral_decode.py`` pins the rank rule and proves the *decode* can emit
every level.  This module pins the rest of the H-CORAL invariant, which the
decode rule alone does not deliver:

1. every interior level is reachable by the head **as shipped** (the real
   ``OrderedCoralHead`` initialisation, not a cut-point vector copied out of
   one old checkpoint);
2. the training target and the decode agree at the loss optimum, so no
   off-by-one can shift L3-L7;
3. the interior class mass has a hard ceiling ``tanh(gap / 4)`` that is
   recoverable from the probability matrix alone;
4. the abstention threshold may not sit above that ceiling, because a
   threshold above it turns every L4/L5/L6 decode into ``unclear`` and the
   published confusion matrix then shows only L3 and L7 -- the exact symptom
   of the original H-CORAL bug, arriving through the abstention gate instead
   of through argmax.

Measured on the shipped runs before the fix (all five v0.7 seeds,
``runs/v07/*/best.pt`` + ``test_predictions.npz`` +
``calibration.json``): interior decodes 20/22/22/18/23 per seed, interior
decodes accepted 0/3/0/2/0.  The declared under-classification gate reported
``target_met: true`` for all five seeds at thresholds of 0.5189 / 0.2327 /
0.5092 / 0.2394 / 0.2286 against interior mass ceilings of 0.2232 / 0.2305 /
0.2229 / 0.2383 / 0.2228.
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from eval.calibration import calibrate_abstention_threshold
from eval.metrics import evaluate_predictions
from train.losses import ordinal_targets
from train.model import (
    LEVELS,
    OrderedCoralHead,
    coral_decode,
    coral_logits_to_probs,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _head_cutpoints(head: OrderedCoralHead) -> np.ndarray:
    """The four strictly-decreasing cut-point logits the head will emit."""
    with torch.no_grad():
        gaps = F.softplus(head.bias_gaps_raw) + 1e-4
        return torch.cat(
            [head.first_bias.reshape(1), head.first_bias - torch.cumsum(gaps, dim=0)]
        ).numpy()


def _probs_for_scores(scores: np.ndarray, cutpoints: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = torch.as_tensor(scores[:, None] + cutpoints[None, :], dtype=torch.float32)
    return coral_logits_to_probs(logits / temperature).numpy()


def _v07_shaped_validation_set(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A CORAL probability matrix with the v0.7 geometry and split shape.

    72 confidently-decoded tail events that are labelled correctly, plus 28
    events whose true level is L7 but which decode to an interior level -- so
    every interior decode is a dangerous under-call.  With
    ``target_under_fnr=0.05`` the only way to satisfy the gate is to abstain
    on all 28, which is what the unfixed selector does.
    """
    head = OrderedCoralHead(in_features=8, num_classes=len(LEVELS), dropout=0.0)
    with torch.no_grad():
        head.first_bias.fill_(1.553)
        head.bias_gaps_raw.copy_(
            torch.log(torch.expm1(torch.tensor([1.5789, 1.5789, 1.5879]) - 1e-4))
        )
    cutpoints = _head_cutpoints(head)
    temperature = 1.682  # the v0.7 fitted temperature
    edges = -cutpoints / temperature  # decode=k iff edges[k-1] < score <= edges[k]

    rng = np.random.default_rng(seed)
    tail = np.concatenate([rng.normal(6.0, 1.2, 66), rng.normal(-3.0, 0.8, 6)])
    # one group per interior window, parked at the window centre, where the
    # interior class mass reaches its ceiling
    interior = np.concatenate(
        [np.full(9, (edges[k - 1] + edges[k]) / 2.0) for k in (1, 2, 3)]
    )
    scores = np.concatenate([tail, interior])
    labels = np.asarray([4] * 66 + [0] * 6 + [4] * len(interior), dtype=np.int64)
    return _probs_for_scores(scores, cutpoints, temperature), labels


# ---------------------------------------------------------------------------
# 1. reachability at the shipped initialisation
# ---------------------------------------------------------------------------


def test_every_level_reachable_at_the_real_head_initialisation() -> None:
    """The head as constructed must reach L3..L7, not just one old checkpoint."""
    torch.manual_seed(0)
    head = OrderedCoralHead(in_features=768, num_classes=len(LEVELS), dropout=0.0)
    cutpoints = _head_cutpoints(head)
    scores = np.linspace(-12.0, 12.0, 24001)
    predictions, _ = coral_decode(_probs_for_scores(scores, cutpoints))
    assert sorted({int(v) for v in predictions}) == [0, 1, 2, 3, 4]
    # and the reachable windows must actually be non-empty intervals
    for level in range(len(LEVELS)):
        window = scores[predictions == level]
        assert window.size > 0, f"level index {level} (L{LEVELS[level]}) has an empty score window"


def test_decode_is_monotone_in_the_score_for_the_shipped_head() -> None:
    torch.manual_seed(0)
    head = OrderedCoralHead(in_features=768, num_classes=len(LEVELS), dropout=0.0)
    scores = np.linspace(-12.0, 12.0, 4801)
    predictions, _ = coral_decode(_probs_for_scores(scores, _head_cutpoints(head)))
    assert (np.diff(predictions) >= 0).all()


# ---------------------------------------------------------------------------
# 2. training target and decode agree; no off-by-one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level_index", range(len(LEVELS)))
def test_training_target_and_decode_agree_at_the_loss_optimum(level_index: int) -> None:
    """At the CORAL optimum for label k the decode must return k, and the
    target must be exactly the sign pattern the decode reads back."""
    target = ordinal_targets(torch.tensor([level_index]), len(LEVELS))
    logits = torch.zeros(1, len(LEVELS) - 1, requires_grad=True)
    optimizer = torch.optim.LBFGS([logits], lr=0.5, max_iter=300, tolerance_grad=1e-12)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        decoded, _ = coral_decode(coral_logits_to_probs(logits.detach()).numpy())
    assert int(decoded[0]) == level_index
    # the decode counts #{j : logit_j > 0}; the target must be the same vector
    assert (logits.detach() > 0).numpy().tolist() == target.numpy().astype(bool).tolist()


def test_level_index_mapping_has_no_off_by_one() -> None:
    assert list(LEVELS) == [3, 4, 5, 6, 7]
    for index in range(len(LEVELS)):
        assert LEVELS[index] == index + 3
    # the manifest round trip used by ManifestImageDataset / engine.py
    for level in LEVELS:
        assert LEVELS[level - LEVELS[0]] == level


def test_interior_targets_are_strictly_greater_than_the_threshold() -> None:
    """``>`` not ``>=``: a ``>=`` would fire one threshold too many."""
    targets = ordinal_targets(torch.arange(len(LEVELS)), len(LEVELS)).numpy()
    assert targets.tolist() == [
        [int(k > j) for j in range(len(LEVELS) - 1)] for k in range(len(LEVELS))
    ]
    assert [int(row.sum()) for row in targets] == list(range(len(LEVELS)))


# ---------------------------------------------------------------------------
# 3. the interior mass ceiling is recoverable from probabilities alone
# ---------------------------------------------------------------------------


def test_interior_mass_ceiling_matches_the_head_geometry() -> None:
    from train.model import coral_interior_mass_ceiling

    torch.manual_seed(3)
    head = OrderedCoralHead(in_features=768, num_classes=len(LEVELS), dropout=0.0)
    cutpoints = _head_cutpoints(head)
    gaps = -np.diff(cutpoints)  # b_j - b_{j+1}
    expected = float(np.tanh(gaps.min() / 4.0))

    scores = np.linspace(-12.0, 12.0, 4001)
    for temperature in (0.5, 1.0, 1.682, 2.0):
        probabilities = _probs_for_scores(scores, cutpoints, temperature)
        measured = coral_interior_mass_ceiling(probabilities)
        assert measured == pytest.approx(float(np.tanh(gaps.min() / (4.0 * temperature))), abs=1e-6)
        # and no interior decode may ever exceed it (float32 rounding only)
        decoded, confidence = coral_decode(probabilities)
        interior = np.isin(decoded, [1, 2, 3])
        assert confidence[interior].max() <= measured + 1e-6
    assert expected > 0.3  # the head is initialised so every level is reachable


def test_interior_mass_ceiling_is_none_for_a_non_coral_matrix() -> None:
    """A softmax matrix is not a CORAL reconstruction; the ceiling must say so
    rather than invent a number."""
    from train.model import coral_interior_mass_ceiling

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(64, 5))
    softmaxed = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmaxed /= softmaxed.sum(axis=1, keepdims=True)
    assert coral_interior_mass_ceiling(softmaxed) is None


# ---------------------------------------------------------------------------
# 4. the abstention threshold may not black out the interior levels
# ---------------------------------------------------------------------------


def test_abstention_threshold_never_exceeds_the_interior_mass_ceiling() -> None:
    """H-CORAL, second front.  The rank rule made every level reachable by the
    decode; a threshold above the interior mass ceiling makes L4/L5/L6
    unreachable by the *published* output instead."""
    from train.model import coral_interior_mass_ceiling

    probabilities, labels = _v07_shaped_validation_set()
    ceiling = coral_interior_mass_ceiling(probabilities)
    assert ceiling is not None
    selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)
    assert selection["threshold"] <= ceiling + 1e-12, (
        f"threshold {selection['threshold']:.4f} sits above the interior mass "
        f"ceiling {ceiling:.4f}; every L4/L5/L6 decode would be reported as "
        "'unclear' (v0.7 seeds 29 and 101 selected 0.5092 / 0.5189 against "
        "ceilings of 0.2229 / 0.2232)"
    )
    decoded, confidence = coral_decode(probabilities)
    accepted = confidence >= selection["threshold"]
    assert (np.isin(decoded, [1, 2, 3]) & accepted).any(), (
        "no interior decode survives the selected threshold"
    )


def test_evaluate_predictions_reports_the_pre_abstention_decode() -> None:
    """The published artifact must not be able to show 'only L3 and L7
    predicted' while the stored probabilities decode to interior levels."""
    probabilities, labels = _v07_shaped_validation_set()
    decoded, _ = coral_decode(probabilities)
    interior_decoded = int(np.isin(decoded, [1, 2, 3]).sum())
    assert interior_decoded > 0

    result = evaluate_predictions(probabilities, labels, 0.60, bootstrap_samples=0)
    pre = result["decoded_level_counts_pre_abstention"]
    assert sum(pre.values()) == len(labels)
    assert sum(pre[f"L{level}"] for level in (4, 5, 6)) == interior_decoded
    # and the post-abstention view must not silently agree
    post = np.asarray(result["confusion_matrix"])[:, :5].sum(axis=0)
    assert result["interior_levels_reachable"] is bool(post[1:4].sum() > 0)
