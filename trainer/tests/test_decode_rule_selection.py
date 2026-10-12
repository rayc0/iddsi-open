"""Tests for the selectable CORAL decode rule (lane fix03_retrain, 2026-10-12).

The v0.7 release ships the H-CORAL ordinal rank rule. This lane was asked to
retrain under candidate (c) argmax, so the two rules must be selectable without
either of them being silently redefined. These tests pin:

1. the default is the rank rule and its output is bit-for-bit unchanged, so no
   existing number moves because the knob exists;
2. ``argmax`` is selectable and is the pre-H-CORAL categorical argmax;
3. an unknown rule is rejected rather than falling back to a default;
4. the chosen rule travels into the metrics artifact and into the calibration
   record, so a reader can tell which decode produced a published number;
5. the v0.7-measured reason argmax cannot fix the L5/L6 problem: at the
   cut-point spacing the head actually learns, argmax emits no interior level
   while the rank rule does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from eval.calibration import calibrate_abstention_threshold
from eval.metrics import evaluate_predictions
from train.config import DEFAULTS, load_config, validate_config
from train.model import (
    DECODE_RULES,
    OrderedCoralHead,
    coral_decode,
    coral_decode_argmax,
    coral_interior_mass_ceiling,
    coral_logits_to_probs,
)


def _probs_for_score(score: float, biases: torch.Tensor) -> np.ndarray:
    return coral_logits_to_probs((score + biases).unsqueeze(0)).numpy()


def _v07_shaped_probabilities(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """A CORAL-shaped probability matrix with interior decodes in it."""
    rng = np.random.default_rng(seed)
    logits = torch.from_numpy(np.sort(rng.normal(size=(96, 4)), axis=1)[:, ::-1].copy())
    probabilities = coral_logits_to_probs(logits).numpy()
    labels = rng.integers(0, 5, size=len(probabilities))
    return probabilities, labels


def test_default_decode_is_rank_and_unchanged() -> None:
    """Calling `coral_decode(p)` must be identical to `coral_decode(p, "rank")`."""
    probabilities, _ = _v07_shaped_probabilities()
    bare_levels, bare_confidence = coral_decode(probabilities)
    rank_levels, rank_confidence = coral_decode(probabilities, "rank")
    assert np.array_equal(bare_levels, rank_levels)
    assert np.array_equal(bare_confidence, rank_confidence)
    assert "rank" in DECODE_RULES and DEFAULTS["evaluation"]["decode"] == "rank"


def test_argmax_rule_is_the_categorical_argmax() -> None:
    probabilities, _ = _v07_shaped_probabilities()
    levels, confidence = coral_decode(probabilities, "argmax")
    assert np.array_equal(levels, probabilities.argmax(axis=1))
    assert np.allclose(confidence, probabilities.max(axis=1))
    assert np.array_equal(levels, coral_decode_argmax(probabilities)[0])


def test_unknown_decode_rule_is_rejected() -> None:
    probabilities, labels = _v07_shaped_probabilities()
    with pytest.raises(ValueError, match="decode rule must be one of"):
        coral_decode(probabilities, "median")
    with pytest.raises(ValueError, match="decode rule must be one of"):
        evaluate_predictions(probabilities, labels, 0.3, bootstrap_samples=0, decode="median")
    with pytest.raises(ValueError, match="decode rule must be one of"):
        calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30, "median")
    with pytest.raises(ValueError, match="evaluation.decode must be"):
        validate_config({**DEFAULTS, "evaluation": {"decode": "median"}})


def test_metrics_record_the_decode_rule_that_produced_them() -> None:
    """A published number must carry the rule that made it."""
    probabilities, labels = _v07_shaped_probabilities()
    for rule in DECODE_RULES:
        result = evaluate_predictions(
            probabilities, labels, 0.2, bootstrap_samples=0, decode=rule
        )
        assert result["decode_rule"] == rule
        # The pre-abstention census is rule-dependent; both must sum to n.
        assert sum(result["decoded_level_counts_pre_abstention"].values()) == len(labels)
    assert (
        evaluate_predictions(probabilities, labels, 0.2, bootstrap_samples=0)["decode_rule"]
        == "rank"
    )


def test_calibration_records_the_decode_rule() -> None:
    probabilities, labels = _v07_shaped_probabilities()
    for rule in DECODE_RULES:
        selection = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30, rule)
        assert selection["decode_rule"] == rule
        # The H-CORAL reachability clip holds under BOTH rules: a threshold
        # above the interior mass ceiling deletes L4/L5/L6 from the output.
        ceiling = selection["interior_mass_ceiling"]
        if ceiling is not None:
            assert selection["threshold"] <= ceiling + 1e-12


def test_argmax_cannot_express_an_interior_level_at_the_learned_gap() -> None:
    """The measured reason candidate (c) does not rescue L5/L6.

    The five v0.7 release seeds learned cut-point gaps of ~0.91 (init 1.70), so
    `coral_interior_mass_ceiling` returns ~0.22 while the tail classes are
    unbounded. At that spacing the categorical argmax can only ever top the two
    tails plus the single interior level whose *both* neighbours happen to be
    below the ceiling at the same score -- and only L5 qualifies, by a margin of
    hundredths. Measured over the whole score range at gap = 0.91 the argmax
    reaches exactly {L3, L5, L7}; L4 and L6 are unreachable. That is precisely
    the wave-1 measurement on the real v0.7 held-out predictions (argmax never
    tops L4 or L6; it tops L5 2-5 times out of 108), while the rank rule emits
    18-23 interior decodes per seed.
    """
    head = OrderedCoralHead(in_features=8, num_classes=5)
    with torch.no_grad():
        # Freeze the head at the gap the v0.7 runs actually converged to.
        head.bias_gaps_raw.fill_(float(np.log(np.expm1(0.91))))
        cutpoints = torch.cat(
            [
                head.first_bias.reshape(1),
                head.first_bias - torch.cumsum(F.softplus(head.bias_gaps_raw) + 1e-4, dim=0),
            ]
        )
    scores = np.linspace(-4.0, 4.0, 4001)
    reached_rank: set[int] = set()
    reached_argmax: set[int] = set()
    probabilities_seen: list[np.ndarray] = []
    for score in scores:
        probabilities = _probs_for_score(float(score), cutpoints)
        reached_rank.add(int(coral_decode(probabilities)[0][0]))
        reached_argmax.add(int(coral_decode(probabilities, "argmax")[0][0]))
        probabilities_seen.append(probabilities[0])
    ceiling = coral_interior_mass_ceiling(np.asarray(probabilities_seen))

    assert reached_rank == {0, 1, 2, 3, 4}, "rank rule lost interior reachability"
    assert reached_argmax == {0, 2, 4}, (
        "argmax reachability changed shape; expected the measured v0.7 pattern "
        f"(only L3/L5/L7) and got {sorted(reached_argmax)}"
    )
    assert ceiling is not None and 0.20 < ceiling < 0.25, (
        f"interior mass ceiling {ceiling} is not the ~0.22 the v0.7 seeds show"
    )
