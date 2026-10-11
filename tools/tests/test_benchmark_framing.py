"""Regression guard for the benchmark-framing numbers.

These assertions pin the exact figures published in `docs/BENCHMARK_FRAMING.md`
and `hf/model_README.md` (v0.7 section) to the numbers recomputed from
`data/manifest_v07_weighted.jsonl` and `runs/v07/*`. If a future lane changes a
label, a split, a decode rule or a threshold, these fail rather than letting a
stale number sit next to an external benchmark.

Run: docker exec -w /work/lanes/an05_external iddsi-train python -m pytest tools/tests/test_benchmark_framing.py -q
CPU only, no GPU.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = next((p for p in (Path("/work"), Path("/home/tun/iddsi")) if (p / "data/manifest_v07_weighted.jsonl").exists()), None)
pytestmark = pytest.mark.skipif(ROOT is None, reason="v0.7 manifest not reachable")

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
OUT = None if ROOT is None else ROOT / "lane_runs/an05_external/benchmark_framing.json"


def _run(script: str) -> dict:
    subprocess.run([sys.executable, "-I", str(TOOLS / script)], check=True, cwd=str(TOOLS))
    name = "benchmark_framing.json" if script == "benchmark_framing.py" else "label_noise_band.json"
    return json.loads((ROOT / "lane_runs/an05_external" / name).read_text())


@pytest.fixture(scope="module")
def framing() -> dict:
    return _run("benchmark_framing.py")


@pytest.fixture(scope="module")
def band() -> dict:
    return _run("label_noise_band.py")


def test_self_check_reproduces_published_metrics(framing):
    """The reading path must be the published one: recomputed kappa and coverage
    have to match each run's metrics.json exactly."""
    assert framing["self_checks"], "no seeds checked"
    for c in framing["self_checks"]:
        assert c["abs_diff"] == 0.0, c
        assert c["recomputed_coverage"] == c["published_coverage"], c


def test_test_split_shape(framing):
    s = framing["summary"]
    assert s["n_test_events"] == 108
    assert s["test_level_distribution"] == {"3": 10, "4": 11, "5": 7, "6": 10, "7": 70}


def test_headline_same_basis_as_the_external_anchor(framing):
    o = framing["summary"]["ours_same_basis"]
    assert o["exact_match_pct_covered"] == pytest.approx([80.89887640449437, 84.88372093023256], abs=1e-6)
    assert o["mae_covered_levels"] == pytest.approx([0.2441860465116279, 0.29213483146067415], abs=1e-6)
    assert o["coverage_pct"] == pytest.approx([78.70370370370371, 85.18518518518519], abs=1e-6)
    assert o["n_covered_range"] == [85, 92]


def test_alternative_denominators(framing):
    a = framing["summary"]["ours_alternative_denominators"]
    assert a["exact_match_pct_allN_abstain_as_miss"] == pytest.approx([66.66666666666667, 70.37037037037037], abs=1e-6)
    assert a["macro_exact_match_pct_covered"] == pytest.approx([37.5, 48.0], abs=1e-6)


def test_pooled_figure_and_ci(framing):
    p = framing["summary"]["pooled_all_5_seeds"]
    assert p["n_covered_events_across_5_seeds"] == 437
    assert p["exact_match_pct_covered"] == pytest.approx(83.52402745995424, abs=1e-6)
    assert p["exact_match_pct_covered_ci95"] == pytest.approx([79.86270022883295, 86.95652173913044], abs=1e-6)
    assert p["mae_covered_levels"] == pytest.approx(0.2631578947368421, abs=1e-6)


def test_class_prior_baselines(framing):
    b = framing["summary"]["baselines_on_the_same_108_events"]
    assert b["always_L7"]["exact_match_pct"] == pytest.approx(64.81481481481481, abs=1e-6)
    assert b["always_L7"]["mae_levels"] == pytest.approx(0.8981481481481481, abs=1e-6)
    assert b["sample_from_test_prior"]["exact_match_pct"] == pytest.approx(45.18175582990398, abs=1e-6)


def test_interior_levels_are_never_correct(framing):
    """L5 and L6 exact match is 0.0% in every seed - the finding that keeps the
    exact-match number honest."""
    for s in framing["per_seed"]:
        assert s["per_class_covered"]["5"]["exact"] == 0.0, s["seed"]
        assert s["per_class_covered"]["6"]["exact"] == 0.0, s["seed"]
        assert s["per_class_covered"]["7"]["exact"] == 1.0, s["seed"]


def test_printed_release_card_row_is_the_abstention_bypassed_basis(framing):
    """The row hf/model_README.md printed against the EdUHK anchor is CORAL over
    all 108 events with abstention bypassed - and no other rule reproduces it."""
    v = framing["summary"]["decode_variant_audit"]["verdict"]
    assert v["coral_over_all_108_events_abstention_bypassed"]["exact_match_matches_printed"] is True
    assert v["coral_over_all_108_events_abstention_bypassed"]["mae_matches_printed"] is True
    for name, verdict in v.items():
        if name != "coral_over_all_108_events_abstention_bypassed":
            assert not (verdict["exact_match_matches_printed"] and verdict["mae_matches_printed"]), name


def test_v05_cross_check_confirms_the_basis_is_systemic(framing):
    c = framing["summary"]["decode_variant_audit"]["v05_cleaned_314row_model_cross_check"]
    assert c["verdict"]["coral_allN_matches_printed"] is True
    assert c["verdict"]["argmax_allN_matches_printed"] is False


def test_label_noise_band(band):
    assert band["n_test_events"] == 108
    assert band["n_split_test_events_relabelled"] == 22
    assert band["exact_match_pct_covered_flipped"] == pytest.approx([79.7752808988764, 87.05882352941177], abs=1e-6)
    assert band["mae_covered_levels_flipped"] == pytest.approx([0.18823529411764706, 0.3258426966292135], abs=1e-6)


def test_published_doc_numbers_match_the_json(framing, band):
    """Every figure quoted in docs/BENCHMARK_FRAMING.md must be the one in the JSON."""
    doc = (ROOT / "lanes/an05_external/docs/BENCHMARK_FRAMING.md").read_text()
    s = framing["summary"]
    for token in [
        "80.90", "84.88", "0.244", "0.292", "83.52", "[79.86, 86.96]",
        "66.67", "70.37", "37.5", "48.0", "64.81", "0.898", "45.18",
        "79.78", "87.06", "0.188", "0.326", "69.44", "74.07",
    ]:
        assert token in doc, token
    assert "0.809" in doc and "61.74" in doc and "70.43" in doc
    assert s["n_test_events"] == 108
