"""Differential tests for the headline statistics this project publishes.

``eval/inter_judge_agreement.py`` regenerates the release's headline
inter-judge agreement number, and its own docstring says every published figure
must come from that script rather than from memory. Its kappa formulas are
therefore checked against scikit-learn on the same inputs, so a silent formula
drift cannot reach a release card.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from eval.inter_judge_agreement import _level, kappa, load_pairs

LEVELS = [3, 4, 5, 6, 7]


def _dataset(root: Path, events: list[dict], tier2: list[dict] | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    if tier2 is not None:
        (root / "tier2_labels.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in tier2), encoding="utf-8"
        )
    return root


def _event(event_id: str, level_weak: object) -> dict:
    return {"event_id": event_id, "level_weak": level_weak}


def _tier2(event_id: str, level_tier2: object) -> dict:
    return {"event_id": event_id, "level_tier2": level_tier2}


def _pairs(n: int, seed: int = 0) -> list[tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    return [
        (f"E{index}", int(a), int(b))
        for index, (a, b) in enumerate(
            zip(rng.choice(LEVELS, n), rng.choice(LEVELS, n))
        )
    ]


@pytest.mark.parametrize("n", [2, 5, 25, 120])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_unweighted_kappa_matches_sklearn(n: int, seed: int) -> None:
    pairs = _pairs(n, seed)
    expected = cohen_kappa_score([a for _, a, _ in pairs], [b for _, _, b in pairs])
    assert kappa(pairs, weighted=False) == pytest.approx(expected)


@pytest.mark.parametrize("n", [2, 5, 25, 120])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_quadratic_kappa_matches_sklearn(n: int, seed: int) -> None:
    pairs = _pairs(n, seed)
    expected = cohen_kappa_score(
        [a for _, a, _ in pairs], [b for _, _, b in pairs], weights="quadratic"
    )
    assert kappa(pairs, weighted=True) == pytest.approx(expected)


def test_kappa_is_nan_without_pairs() -> None:
    assert np.isnan(kappa([], weighted=False))
    assert np.isnan(kappa([], weighted=True))


def test_perfect_agreement_gives_kappa_one() -> None:
    pairs = [("E0", 4, 4), ("E1", 5, 5), ("E2", 6, 6)]
    assert kappa(pairs, weighted=False) == pytest.approx(1.0)
    assert kappa(pairs, weighted=True) == pytest.approx(1.0)


def test_level_parser_accepts_only_l3_to_l7() -> None:
    for value in (3, 4, 5, 6, 7, "5", 4.0):
        assert _level(value) == int(value)
    for value in (None, "", 2, 8, -1, "abc", 4.5, 6.9, True, False, [4]):
        assert _level(value) is None


def test_non_integral_levels_do_not_truncate_into_a_neighbouring_level() -> None:
    """``int(6.9)`` is 6, which would silently join a published crosstab."""
    assert _level(6.9) is None
    assert _level(3.5) is None


def test_load_pairs_intersects_on_event_id(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [
            _event("E0", 4),
            _event("E1", 5),
            _event("E2", None),  # tier-1 abstained: excluded
            _event("E3", 6),
        ],
        [
            _tier2("E0", 4),
            _tier2("E1", 6),
            _tier2("E9", 4),  # no tier-1 label: excluded
            _tier2("E3", None),  # tier-2 abstained: excluded
        ],
    )
    pairs = load_pairs(dataset)
    assert pairs == [("E0", 4, 4), ("E1", 5, 6)]


def test_load_pairs_returns_empty_without_the_tier2_sidecar(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [_event("E0", 4)], None)
    assert load_pairs(dataset) == []


def test_load_pairs_skips_malformed_json_lines(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir(parents=True)
    (dataset / "events.jsonl").write_text(
        json.dumps(_event("E0", 4)) + "\n" + "{not json}\n" + json.dumps(_event("E1", 5)) + "\n",
        encoding="utf-8",
    )
    (dataset / "tier2_labels.jsonl").write_text(
        json.dumps(_tier2("E0", 4)) + "\n" + "{also not json}\n", encoding="utf-8"
    )
    assert load_pairs(dataset) == [("E0", 4, 4)]
