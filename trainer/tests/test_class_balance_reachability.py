"""Contract tests for the class-balance / interior-reachability diagnostics.

These pin the two things this lane's experiments are read against: the
per-level recall denominators (H-ABSTBLIND / H-KAPPACOV: an abstention is a
miss on the all-rows denominator and simply absent on the covered one), and
the balanced sampler's claim to equalise levels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.metrics import evaluate_predictions, per_level_recall
from train.engine import _balanced_sampler
from train.losses import effective_number_weights


class _FakeDataset:
    """Minimal stand-in exposing the attributes ``_balanced_sampler`` reads."""

    def __init__(self, labels: list[int]):
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)


def test_per_level_recall_counts_abstentions_as_misses_on_all_rows() -> None:
    """The two recall denominators must disagree in the abstention direction."""
    labels = np.array([0, 0, 0, 0, 1])
    raw = np.array([0, 0, 0, 1, 1])
    accepted = np.array([True, True, False, True, True])
    recall = per_level_recall(labels, raw, accepted)
    assert recall["L3"]["support"] == 4
    assert recall["L3"]["recall_all"] == pytest.approx(3 / 4)
    assert recall["L3"]["covered_support"] == 3
    assert recall["L3"]["recall_covered"] == pytest.approx(2 / 3)
    assert recall["L4"]["support"] == 1
    assert recall["L4"]["recall_all"] == pytest.approx(1.0)


def test_per_level_recall_zero_support_is_nan_not_zero() -> None:
    """A level with no support has no denominator, so it must not read as 0.00."""
    labels = np.array([0, 0, 4])
    raw = np.array([0, 4, 4])
    accepted = np.array([True, True, True])
    recall = per_level_recall(labels, raw, accepted)
    for level in ("L4", "L5", "L6"):
        assert recall[level]["support"] == 0
        assert np.isnan(recall[level]["recall_all"])
        assert np.isnan(recall[level]["recall_covered"])


def test_evaluate_predictions_reports_per_level_recall_and_argmax() -> None:
    rng = np.random.default_rng(11)
    probabilities = rng.dirichlet(np.ones(5), size=30)
    labels = rng.integers(0, 5, size=30)
    result = evaluate_predictions(probabilities, labels, 0.3, bootstrap_samples=0)
    assert set(result["per_level_recall"]) == {"L3", "L4", "L5", "L6", "L7"}
    assert set(result["argmax_level_counts"]) == {"L3", "L4", "L5", "L6", "L7"}
    assert sum(result["argmax_level_counts"].values()) == 30
    assert sum(result["decoded_level_counts_pre_abstention"].values()) == 30
    # Recall denominators must add up to the declared event count.
    assert sum(v["support"] for v in result["per_level_recall"].values()) == result["n"]


def test_balanced_sampler_draws_every_level_with_equal_probability() -> None:
    # The v0.7 train split, L3..L7 = 28/37/13/35/201, as per-sample labels.
    labels = [0] * 28 + [1] * 37 + [2] * 13 + [3] * 35 + [4] * 201
    sampler = _balanced_sampler(_FakeDataset(labels), seed=17)
    assert sampler.num_samples == len(labels)
    assert sampler.replacement
    drawn = [labels[index] for index in sampler]
    counts = {level: drawn.count(level) for level in range(5)}
    # The natural counts are 28/37/13/35/201; the sampler must flatten them to
    # ~314/5 = 62.8 each, not merely skew them less.
    assert counts[4] < labels.count(4)
    for level, count in counts.items():
        assert 45 <= count <= 82, counts


def test_balanced_sampler_rejects_a_split_with_a_level_gap() -> None:
    """A missing middle level must fail loudly, not equalise over fewer levels."""
    with pytest.raises(ValueError, match="missing \\[2\\]"):
        _balanced_sampler(_FakeDataset([0, 0, 1, 1, 3, 3, 3]), seed=1)
    with pytest.raises(ValueError, match="non-empty"):
        _balanced_sampler(_FakeDataset([]), seed=1)


def test_effective_number_weights_saturate_toward_inverse_frequency() -> None:
    """beta -> 1 is the inverse-frequency limit; the shipped beta is close to it.

    On the v0.7 train split (L3..L7 = 28/37/13/35/201) beta=0.999 already sits
    within ~10% of that limit, which is why "a stronger beta" is a small move.
    ``effective_number_weights`` takes per-sample labels, so the counts are
    expanded into a label vector before the call.
    """
    counts = [28, 37, 13, 35, 201]
    labels = [level for level, count in enumerate(counts) for _ in range(count)]
    shipped = effective_number_weights(labels, beta=0.999).numpy()
    limit = effective_number_weights(labels, beta=0.999999).numpy()
    # The weights are normalised to mean 1 over the present levels, so the
    # beta -> 1 limit is 1/count renormalised the same way.
    inverse = 1.0 / np.asarray(counts, dtype=float)
    inverse = inverse / inverse.mean()
    assert float(np.abs(shipped / limit - 1).max()) < 0.10
    assert float(np.abs(limit / inverse - 1).max()) < 1e-3
    # A weaker beta must move the L7 down-weight, i.e. the ratio must fall.
    weaker = effective_number_weights(labels, beta=0.99).numpy()
    assert weaker.max() / weaker.min() < shipped.max() / shipped.min()


def test_balanced_sampler_is_seed_reproducible() -> None:
    labels = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4
    first = list(_balanced_sampler(_FakeDataset(labels), seed=29))
    second = list(_balanced_sampler(_FakeDataset(labels), seed=29))
    other = list(_balanced_sampler(_FakeDataset(labels), seed=30))
    assert first == second
    assert first != other
