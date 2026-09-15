from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from eval.inter_rater import inter_rater_baseline, load_manifest_rows, main

FIXTURE = Path(__file__).parent / "fixtures" / "inter_rater_manifest.jsonl"


def test_fixture_loads() -> None:
    rows = load_manifest_rows(FIXTURE)
    assert len(rows) == 12
    assert all("event_id" in row for row in rows)


def test_inter_rater_baseline_counts_and_agreement() -> None:
    result = inter_rater_baseline(FIXTURE, "level_tested_RD", "level_tested_SLP")
    assert result["n_rows"] == 12
    assert result["n_both_rated"] == 8
    # Abstentions are reported separately, not silently dropped.
    abst = result["abstentions"]
    assert abst["rater_a_abstained"] == 1  # RD null on FIX_RATER_11
    assert abst["rater_b_abstained"] == 2  # SLP null on FIX_RATER_09/10
    assert abst["both_abstained"] == 1
    assert abst["either_abstained"] == 4
    assert abst["rater_a_abstained_event_ids"] == ["FIX_RATER_11"]
    assert sorted(abst["rater_b_abstained_event_ids"]) == ["FIX_RATER_09", "FIX_RATER_10"]
    assert abst["both_abstained_event_ids"] == ["FIX_RATER_12"]
    assert result["raw_agreement"] == pytest.approx(6 / 8)
    a = np.array([3, 3, 4, 5, 5, 6, 7, 4])
    b = np.array([3, 4, 4, 5, 6, 6, 7, 4])
    assert result["weighted_kappa"] == pytest.approx(cohen_kappa_score(a, b, weights="quadratic"))


def test_confusion_matrix_shape_and_totals() -> None:
    result = inter_rater_baseline(FIXTURE, "level_tested_RD", "level_tested_SLP")
    matrix = result["confusion_matrix"]
    assert result["confusion_labels"] == list(range(8))
    assert len(matrix) == 8 and all(len(row) == 8 for row in matrix)
    assert sum(sum(row) for row in matrix) == 8  # only doubly-rated rows
    assert matrix[3][3] == 1  # (3,3)
    assert matrix[3][4] == 1  # (3,4)
    assert matrix[4][4] == 2  # (4,4) twice
    assert matrix[5][5] == 1  # (5,5)
    assert matrix[5][6] == 1  # (5,6)
    assert matrix[6][6] == 1  # (6,6)
    assert matrix[7][7] == 1  # (7,7)


def test_all_abstentions_leave_kappa_undefined(tmp_path: Path) -> None:
    manifest = tmp_path / "events.jsonl"
    manifest.write_text(
        '{"event_id": "E1", "level_tested_RD": null, "level_tested_SLP": null}\n'
        '{"event_id": "E2", "level_tested_RD": 4, "level_tested_SLP": null}\n',
        encoding="utf-8",
    )
    result = inter_rater_baseline(manifest)
    assert result["n_both_rated"] == 0
    assert result["weighted_kappa"] is None
    assert result["raw_agreement"] is None
    assert "no events rated by both raters" in result["warning"]


def test_missing_column_and_bad_levels_raise(tmp_path: Path) -> None:
    manifest = tmp_path / "events.jsonl"
    manifest.write_text(
        '{"event_id": "E1", "level_tested_RD": 4, "level_tested_SLP": 4}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rater column"):
        inter_rater_baseline(manifest, "level_tested_RD", "not_a_column")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event_id": "E1", "level_tested_RD": 9, "level_tested_SLP": 4}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="level 0..7"):
        inter_rater_baseline(bad)


def test_cli_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(FIXTURE), "--rater-a", "level_tested_RD", "--rater-b", "level_tested_SLP"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_both_rated"] == 8
    assert payload["weighted_kappa"] is not None
