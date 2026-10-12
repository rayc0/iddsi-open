#!/usr/bin/env python3
"""Tests for tools/verify_paper_claims.py -- the paper-reconciliation gate.

Lane rel06_paper (2026-10-12) reconciled paper/paper.md and paper/sections/
against what wave 1 established. Four rules came out of it, and each one is
enforced here so a future edit cannot quietly undo the reconciliation:

  R1  every paragraph quoting a kappa names its YARDSTICK and its POPULATION
      (covered rows / all rows / a no-abstention pair set) -- hazard H-YARDEVID;
  R2  the 152 unusable events are stated with BOTH denominators, 152/684
      (whole corpus) and 152/510 (both-judged subset) -- hazard H-UNUSABLE;
  R3  no pass claim covering all three models is made without naming the metric
      it passed on -- the wk1 (covered-rows) and all-rows metrics disagree;
  R4  the L5/L6 decode collapse is reported as a limitation with a denominator.

The second half of the test file checks that the NUMBERS the paper quotes are
the numbers the source files compute, so prose and evidence cannot drift.

Run from repo root:  python -m pytest tools/tests/test_paper_claims.py -q
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT = _REPO_ROOT / "tools" / "verify_paper_claims.py"

_spec = importlib.util.spec_from_file_location("verify_paper_claims", _SCRIPT)
vpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpc)

PAPER_MD = _REPO_ROOT / "paper" / "paper.md"
LIMITATIONS = _REPO_ROOT / "paper" / "sections" / "limitations.md"
RELATED_WORK = _REPO_ROOT / "paper" / "sections" / "related_work.md"


@pytest.fixture(scope="module")
def numbers() -> dict:
    """Recompute every figure from the committed source files."""
    return vpc.compute_numbers()


@pytest.fixture(scope="module")
def lint() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-P", str(_SCRIPT), "--out", "/tmp/_paper_claims_test.json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(Path("/tmp/_paper_claims_test.json").read_text())
    return payload["lint"]["problems"]


# --------------------------------------------------------------------------- #
# R1 -- kappa naming (H-YARDEVID)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doc", [PAPER_MD, LIMITATIONS, RELATED_WORK], ids=lambda p: p.name)
def test_no_kappa_is_quoted_without_its_yardstick_and_population(doc):
    offenders = [
        p for p in vpc.lint(doc) if "no yardstick" in p or "no population" in p
    ]
    assert not offenders, "\n".join(offenders)


def test_the_covered_and_all_rows_metrics_are_both_named_for_v07():
    """The headline v0.7 kappa must appear on both bases, not one."""
    text = " ".join(PAPER_MD.read_text().split())
    assert "quadratic weighted kappa, **covered rows**" in text
    assert "quadratic weighted kappa, **all 108 rows**" in text
    # and the bias matrix must be named as the all-rows quantity
    assert (
        "`results/v07_release/bias_matrix.json` holds the all-rows kappa, "
        "not the gate metric" in text
    )


# --------------------------------------------------------------------------- #
# R2 -- corpus denominators (H-UNUSABLE)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doc", [PAPER_MD, LIMITATIONS, RELATED_WORK], ids=lambda p: p.name)
def test_paper_states_both_corpus_denominators(doc):
    text = doc.read_text()
    assert "152/684" in text, f"{doc.name}: 152/684 (whole corpus) missing"
    assert "152/510" in text, f"{doc.name}: 152/510 (both-judged subset) missing"


@pytest.mark.parametrize("doc", [PAPER_MD, LIMITATIONS, RELATED_WORK], ids=lambda p: p.name)
def test_no_denominator_is_mispaired(doc):
    assert not vpc.lint_denominators(doc)


# --------------------------------------------------------------------------- #
# R3 -- "all three pass" must name its metric
# --------------------------------------------------------------------------- #
def test_three_model_pass_claim_names_its_metric():
    offenders = [
        p for p in vpc.lint(PAPER_MD) + vpc.lint(LIMITATIONS) + vpc.lint(RELATED_WORK)
        if "pass claim" in p
    ]
    assert not offenders, "\n".join(offenders)


def test_the_pass_verdict_is_metric_dependent(numbers):
    """The fact the paper now discloses, recomputed from the per-seed table."""
    cell = numbers["all_three_pass_is_metric_dependent"]
    assert cell["v05_wk1_covered_range"][0] >= 0.70, cell
    assert cell["v05_all_rows_mean"] < 0.70, cell
    assert cell["v05_all_rows_seeds_over_0.70"] < 5, cell


# --------------------------------------------------------------------------- #
# R4 -- the decode collapse is reported
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doc", [PAPER_MD, LIMITATIONS], ids=lambda p: p.name)
def test_decode_collapse_is_reported_as_a_limitation(doc):
    assert not vpc.lint_decode_collapse(doc)


def test_l5_and_l6_are_never_emitted_in_any_seed(numbers):
    summary = numbers["v07_summary"]
    assert summary["L5_never_emitted_in_all_5_seeds"] is True
    assert summary["L6_never_emitted_in_all_5_seeds"] is True
    for seed, levels in summary["levels_never_emitted_by_seed"].items():
        assert "L5" in levels and "L6" in levels, (seed, levels)


# --------------------------------------------------------------------------- #
# the numbers themselves
# --------------------------------------------------------------------------- #
def test_v07_covered_kappa_range_reproduces_the_published_one(numbers):
    """0.838-0.860 is `weighted_kappa_covered` on 108 events, ~80% coverage."""
    lo, hi = numbers["v07_summary"]["weighted_kappa_covered_range"]
    assert (round(lo, 3), round(hi, 3)) == (0.838, 0.860)
    assert numbers["v07_summary"]["n_test_events"] == 108
    assert numbers["v07_summary"]["coverage_range"] == [0.787, 0.8519]


def test_all_rows_kappa_is_a_different_and_lower_quantity(numbers):
    cell = numbers["cross_yardstick"]["v0.7 | v0.7 full-coverage consensus"]
    assert cell["all_rows_range"] == [0.7927, 0.8253]
    assert cell["wk1_covered_range"] == [0.8375, 0.8604]
    assert cell["all_rows_range"][1] < cell["wk1_covered_range"][1]


def test_bias_matrix_cells_are_identified_as_all_rows_kappa(numbers):
    ident = numbers["bias_matrix_identified"]
    identified = ident[
        "v0.5 (trained: cleaned consensus) | v0.7 consensus (full coverage)"
    ]["identified_as"].lower()
    assert "all held-out rows" in identified
    assert "abstentions included in the denominator" in identified
    assert ident["v0.5 (trained: cleaned consensus) | v0.7 consensus (full coverage)"]["value"] == (
        pytest.approx(0.671259, abs=1e-6)
    )


def test_v01_decode_fix_kappa_is_the_covered_rows_metric(numbers):
    v01 = numbers["v01_decode_fix"]
    assert v01["n_held_out"] == 109
    assert v01["weighted_kappa_covered"] == pytest.approx(0.4227, abs=1e-4)
    assert v01["coverage"] == pytest.approx(0.8807, abs=1e-4)
    assert "covered" in v01["kappa_population"]


def test_seed43_per_level_recall_matches_the_published_table(numbers):
    seed43 = [r for r in numbers["v07_per_seed"] if r["seed"] == 43][0]
    recall = seed43["per_level_recall_decided"]
    assert recall == {"L3": 0.9, "L4": 0.5, "L5": 0.0, "L6": 0.0, "L7": 1.0}
    assert seed43["n_decided"] == 92
    # precision is undefined for a level that is never predicted
    assert seed43["per_level_precision_decided"]["L5"] is None
    assert seed43["per_level_precision_decided"]["L6"] is None


def test_corpus_denominators_are_both_correct(numbers):
    cd = numbers["corpus_denominators"]
    assert cd["unusable_events"] == 152
    assert cd["whole_corpus"] == "152/684" and cd["whole_corpus_pct"] == 22.2
    assert cd["both_judged_subset"] == "152/510" and cd["both_judged_subset_pct"] == 29.8
    assert cd["corpus_after_cleaning"] == "684 - 152 = 532"
    assert cd["sweep_union_check"] == 685  # 683 + 512 - 510, so 510 is not a clean subset


def test_inter_judge_kappas_have_no_coverage_filter(numbers):
    ij = numbers["inter_judge"]
    assert ij["n_pairs"] == 575
    assert ij["kappa_unweighted"] == 0.1062
    assert ij["kappa_quadratic"] == 0.2766
    assert "no coverage filter" in ij["population"]


def test_always_l7_baseline_is_quoted_with_its_denominator(numbers):
    """70 of 108 test events are L7, so the baseline is 64.8% exact."""
    assert numbers["v07_summary"]["always_L7_baseline_seed43"] == pytest.approx(70 / 108, abs=1e-9)


# --------------------------------------------------------------------------- #
# the gate as a whole
# --------------------------------------------------------------------------- #
def test_lint_passes_on_every_paper_file(lint):
    assert lint == [], "\n".join(lint)
