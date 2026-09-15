from __future__ import annotations

import json
from pathlib import Path

from flowtest.bench.run_bench import GRID, build_grid, run_bench


def test_grid_is_deterministic_and_varied() -> None:
    grid = build_grid()
    assert grid == build_grid()
    expected_len = 1
    for values in GRID.values():
        expected_len *= len(values)
    assert len(grid) == expected_len == 108
    assert grid[0] == {
        "residual_ml": 0.5,
        "tilt_deg": 0.0,
        "bubbles": False,
        "lighting_factor": 1.0,
    }


def test_bench_small_n_produces_expected_summary_structure(tmp_path: Path) -> None:
    report = run_bench(n=8, out_dir=tmp_path / "out")

    # Files written.
    assert (tmp_path / "out" / "bench_summary.json").exists()
    markdown = (tmp_path / "out" / "bench_summary.md").read_text(encoding="utf-8")
    assert "# Flow-Test synthetic bench (n=8)" in markdown
    assert "## Confusion" in markdown

    # Top-level structure.
    assert report["n"] == 8
    assert len(report["cases"]) == 8
    for key in ("scope", "grid", "fixed_params", "seed_rule", "overall", "per_bucket", "confusion_counts", "abstention_reasons"):
        assert key in report

    # Per-case structure, including the real abstention mechanism fields.
    for case in report["cases"]:
        assert case["status"] in ("graded", "abstain")
        assert case["ground_truth_level"] in ("L0", "L1", "L2", "L3", "L4")
        assert case["volume_bucket"] in ("near_boundary", "far_from_boundary")
        assert isinstance(case["abstention_reasons"], list)
        if case["status"] == "graded":
            assert case["predicted_level"] in ("L0", "L1", "L2", "L3", "L4")
            assert case["abstention_reasons"] == []
        else:
            assert case["predicted_level"] is None
            assert case["abstention_reasons"]  # abstain always carries codes

    # Per-bucket structure covers every documented dimension.
    for dimension in ("residual_ml", "tilt_deg", "bubbles", "lighting_factor", "volume_bucket"):
        assert dimension in report["per_bucket"]
        for stats in report["per_bucket"][dimension].values():
            assert stats["n"] >= 1
            assert 0.0 <= stats["abstention_rate"] <= 1.0

    # Overall counts are consistent with the cases.
    overall = report["overall"]
    assert overall["n"] == 8
    assert overall["graded"] + overall["abstained"] == 8
    total_confusion = sum(entry["count"] for entry in report["confusion_counts"])
    assert total_confusion == 8

    # JSON round-trips.
    payload = json.loads((tmp_path / "out" / "bench_summary.json").read_text(encoding="utf-8"))
    assert payload["n"] == 8
