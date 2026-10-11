#!/usr/bin/env python3
"""Tests for tools/audit/audit_cell_accounting.py (BASE_FOOD_LIST.md cell accounting).

These pin the 267-of-300 accounting so a future edit to the food list or to the
card set that breaks it fails loudly instead of silently changing the documented
denominator.

Run from repo root:  python -m pytest tools/tests/test_cell_accounting.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT = _REPO_ROOT / "tools" / "audit" / "audit_cell_accounting.py"
_REPORT = _REPO_ROOT / "tools" / "audit" / "cell_accounting.json"

_spec = importlib.util.spec_from_file_location("audit_cell_accounting", _AUDIT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.fixture(scope="module")
def report() -> dict:
    # Recompute rather than trust the stored artefact.
    audit.main()
    return json.loads(_REPORT.read_text())


def test_food_list_parses_to_60_rows_with_no_gaps_or_dups(report):
    assert report["food_rows_parsed"] == 60
    assert report["food_numbering_gaps"] == []
    assert report["food_rows_duplicated"] == []


def test_planned_grid_is_300_cells(report):
    assert report["planned_cells"] == 300


def test_card_count_matches_the_index_claim(report):
    assert report["cards_on_disk"] == 267
    assert report["index_md_claim"].endswith("(267 cards)")


def test_the_33_absent_cells_are_all_and_only_the_na_cells(report):
    assert report["absent_cells"] == 33
    assert report["absent_cells_explained_by_na"] == 33
    assert report["absent_cells_unexplained"] == 0


def test_no_card_exists_for_an_na_cell(report):
    assert report["cards_present_for_na_cells"] == []


def test_no_card_exists_outside_the_planned_grid(report):
    assert report["unexpected_cards"] == []


def test_nothing_is_duplicated(report):
    assert report["duplicate_card_filenames"] == []
    assert report["byte_identical_card_groups"] == []


def test_absent_cells_split_by_level(report):
    assert report["absent_cells_per_level"] == {"3": 1, "4": 2, "5": 5, "6": 9, "7": 16}


def test_food_coverage_splits_42_full_and_18_partial(report):
    assert report["foods_with_all_five_cards"] == 42
    assert report["foods_with_absent_cells"] == 18
    assert 42 + 18 == 60


def test_no_cell_merely_mentions_na(report):
    # A cell whose note mentions N/A without starting with it would let a
    # silently dropped cell hide behind a partial mention.
    assert report["cells_mentioning_na_without_leading_na"] == []


def test_documented_accounting_section_exists():
    text = (_REPO_ROOT / "docs" / "BASE_FOOD_LIST.md").read_text()
    assert "267 of 300" in text, "BASE_FOOD_LIST.md must state the 267-of-300 accounting"
    assert "deferred" in text and "duplicated" in text, (
        "the section must say explicitly whether the 33 cells are deferred or duplicated"
    )
