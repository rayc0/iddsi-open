#!/usr/bin/env python3
"""Tests for the corpus-hygiene denominators behind hazard H-UNUSABLE.

H-UNUSABLE quotes ONE count -- 152 unusable events -- over TWO different
denominators, and the model card used to do the arithmetic wrong ("152 of 684 =
29.8%", which is 22.2%). These tests recompute every figure in that family from
the files that produced it and then check that each document that quotes a
percentage names the denominator it belongs to.

Run from repo root:  python -m pytest tools/tests/test_denominators.py -q
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# Make `tools` importable regardless of how pytest was invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.denominator_audit import read_ids, read_manifest_ids  # noqa: E402

_SWEEP = _REPO_ROOT / "results/v05_release/labelability_sweep.json"
_IDS = _REPO_ROOT / "results/v05_release/unusable_event_ids.txt"

# Every document in this repo that quotes a figure from the labelability sweep.
_DOCS = [
    _REPO_ROOT / "hf/model_README.md",
    _REPO_ROOT / "hf/HAZARD_LOG.md",
    _REPO_ROOT / "trainer/configs/siglip2_v05.yaml",
]

# percentage -> the denominators it may legitimately be paired with
_REQUIRED_DENOMINATOR = {
    "22.2": "684",  # 152/684, share of the whole corpus
    "29.8": "510",  # 152/510, share of the both-judged subset
    "96.7": "510",  # 493/510, IS-THERE-FOOD agreement
    "85.5": "510",  # 436/510, LEVELLABLE agreement
}


def _data_dir() -> Path | None:
    """Locate the shared manifest directory (container mount or host checkout)."""
    for candidate in (Path("/work/data"), _REPO_ROOT.parents[1] / "data", _REPO_ROOT / "data"):
        if (candidate / "manifest_v04_consensus.jsonl").exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def sweep() -> dict:
    return json.loads(_SWEEP.read_text())


@pytest.fixture(scope="module")
def unusable() -> list[str]:
    return read_ids(_IDS)


# --------------------------------------------------------------------------- #
# the measurements themselves
# --------------------------------------------------------------------------- #


def test_unusable_list_has_no_duplicates(unusable):
    assert len(unusable) == 152
    assert len(set(unusable)) == 152


def test_the_two_percentages_are_both_correct_for_their_own_denominator(sweep, unusable):
    """152/684 = 22.2% and 152/510 = 29.8% -- different denominators, not a conflict."""
    n = len(set(unusable))
    assert n == 152
    assert round(100.0 * n / 684, 1) == 22.2
    assert round(100.0 * n / sweep["both"], 1) == 29.8
    assert sweep["both"] == 510


def test_agreement_rates_use_the_both_judged_denominator(sweep):
    both = sweep["both"]
    assert round(100.0 * sweep["cross_food_agree"] / both, 1) == 96.7
    assert round(100.0 * sweep["cross_lev_agree"] / both, 1) == 85.5


def test_the_152_were_actually_removed_from_684(unusable):
    """The corpus-wide denominator is the defensible one only because the rows moved."""
    data = _data_dir()
    if data is None:
        pytest.skip("shared manifest directory not found")
    pre = read_manifest_ids(data / "manifest_v04_consensus.jsonl")
    assert len(pre) == 684
    assert len(set(unusable) & set(pre)) == 152
    post = read_manifest_ids(data / "manifest_v07_weighted.jsonl")
    assert len(post) == 684 - 152 == 532
    assert len(set(unusable) & set(post)) == 0


def test_sweep_aggregate_is_not_self_consistent(sweep):
    """683 + 512 - 510 = 685 > 684, so 'both = 510' is not a clean subset.

    This is why 29.8% may not be presented as a corpus rate: the aggregate output
    does not record which of the 152 unusable events the 510 contains.
    """
    union = sweep["step_events"] + sweep["mm_events"] - sweep["both"]
    assert union == 685
    assert union > 684


# --------------------------------------------------------------------------- #
# what the documents are allowed to say
# --------------------------------------------------------------------------- #


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


@pytest.mark.parametrize("doc", _DOCS, ids=lambda p: p.name)
def test_every_quoted_percentage_names_its_denominator(doc):
    """A percentage with no denominator on or beside its line is the original bug."""
    lines = _lines(doc)
    offenders = []
    for i, line in enumerate(lines):
        for pct, denom in _REQUIRED_DENOMINATOR.items():
            if pct not in line:
                continue
            window = lines[max(0, i - 1) : i + 2]
            if not any(denom in w for w in window):
                offenders.append(f"{doc.name}:{i + 1}: {pct}% without {denom} nearby")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("doc", _DOCS, ids=lambda p: p.name)
def test_no_percentage_is_paired_with_the_other_denominator(doc):
    """'152 of 684 = 29.8%' is the exact defect this lane exists to remove."""
    lines = _lines(doc)
    offenders = []
    for i, line in enumerate(lines):
        if "29.8" in line and "684" in line and "510" not in line:
            offenders.append(f"{doc.name}:{i + 1}: 29.8% sits next to 684 but not 510")
        if "22.2" in line and "510" in line and "684" not in line:
            offenders.append(f"{doc.name}:{i + 1}: 22.2% sits next to 510 but not 684")
    assert not offenders, "\n".join(offenders)


def test_original_defect_string_is_gone():
    for doc in _DOCS:
        assert "152 of 684 = 29.8%" not in doc.read_text(), doc.name


def test_quoted_values_match_the_recomputed_ones(sweep, unusable):
    """The docs may not drift from what the source files compute."""
    both, n = sweep["both"], len(set(unusable))
    expected = {
        "22.2": round(100.0 * n / 684, 1),
        "29.8": round(100.0 * n / both, 1),
        "96.7": round(100.0 * sweep["cross_food_agree"] / both, 1),
        "85.5": round(100.0 * sweep["cross_lev_agree"] / both, 1),
    }
    for doc in _DOCS:
        for quoted, computed in expected.items():
            if quoted in doc.read_text():
                assert f"{quoted}%" in doc.read_text()
                assert f"{computed:.1f}" == quoted


def test_docs_do_not_claim_both_families_judged_all_684(sweep):
    """The sweep judged 683 (Step-5) and 512 (MiniMax); it never judged 684 twice."""
    for doc in _DOCS:
        text = doc.read_text()
        assert "judged all 684 events with two independent model" not in text
        assert "judged all 684 events with both families" not in text
    assert sweep["step_events"] == 683 and sweep["mm_events"] == 512
