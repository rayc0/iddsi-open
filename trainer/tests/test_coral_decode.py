"""Regression tests for the CORAL ordinal decode.

The first trained model (2026-09-06) scored weighted kappa 0.419 and a 30.3%
dangerous under-classification rate while emitting ONLY L3, L7 and abstain on a
balanced L4-L7 test set that contained no L3 at all. The cause was not the data
and not the head: `coral_logits_to_probs` builds interior classes as differences
of adjacent sigmoids, so with the default cut-point spacing every interior class
was capped near 0.17 while both tail classes reached ~0.26. Decoding by argmax
therefore could not return an interior level for ANY input.

These tests pin the rank rule `level = #{k : P(y > k) > 0.5}` and, critically,
assert the property argmax violated: every level must be reachable.
"""

from __future__ import annotations

import numpy as np
import torch

from train.model import coral_decode, coral_logits_to_probs


def _probs_for_score(score: float, biases: torch.Tensor) -> np.ndarray:
    return coral_logits_to_probs((score + biases).unsqueeze(0)).numpy()


def test_every_level_is_reachable_across_the_score_range() -> None:
    """The defect: argmax could only ever emit the two extreme levels."""
    biases = torch.tensor([1.5002, 0.8034, 0.1059, -0.5919])
    reached = {
        int(coral_decode(_probs_for_score(s, biases))[0][0])
        for s in np.linspace(-4.0, 4.0, 400)
    }
    assert reached == {0, 1, 2, 3, 4}, f"unreachable levels; decoder produced {sorted(reached)}"


def test_argmax_would_have_failed_on_the_same_cutpoints() -> None:
    """Guard the diagnosis itself, so a regression cannot quietly reappear."""
    biases = torch.tensor([1.5002, 0.8034, 0.1059, -0.5919])
    argmax_reached = {
        int(_probs_for_score(s, biases).argmax()) for s in np.linspace(-4.0, 4.0, 400)
    }
    assert argmax_reached == {0, 4}, (
        "argmax unexpectedly reached an interior level; the historical failure "
        f"mode has changed shape: {sorted(argmax_reached)}"
    )


def test_matches_the_cumulative_threshold_definition() -> None:
    rng = np.random.default_rng(17)
    logits = torch.from_numpy(rng.normal(size=(64, 4)).astype(np.float64))
    # Enforce the monotone (ordered) cut-point structure the head guarantees.
    logits = torch.sort(logits, dim=1, descending=True).values
    probs = coral_logits_to_probs(logits).numpy()
    predictions, confidence = coral_decode(probs)
    expected = (torch.sigmoid(logits).numpy() > 0.5).sum(axis=1)
    assert np.array_equal(predictions, expected)
    assert np.all(confidence > 0.0)
    assert np.allclose(confidence, probs[np.arange(len(predictions)), predictions])


def test_monotone_in_the_score() -> None:
    """A higher score must never decode to a lower level."""
    biases = torch.tensor([1.5, 0.8, 0.1, -0.6])
    scores = np.linspace(-4.0, 4.0, 200)
    levels = [int(coral_decode(_probs_for_score(s, biases))[0][0]) for s in scores]
    assert levels == sorted(levels)
