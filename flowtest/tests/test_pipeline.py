from __future__ import annotations

import json

import pytest

from flowtest.grade import main
from flowtest.pipeline import FlowTestConfig, _level_for_volume, grade_video
from flowtest.synthetic import SyntheticSpec, evaluate_synthetic, generate_video


@pytest.mark.parametrize(
    ("residual", "flowed", "expected"),
    [(0.99, 9.01, "L0"), (1.0, 9.0, "L1"), (3.99, 6.01, "L1"), (4.0, 6.0, "L2"), (7.99, 2.01, "L2"), (8.0, 2.0, "L3"), (10.0, 0.0, "L4")],
)
def test_level_mapping(residual: float, flowed: float, expected: str) -> None:
    assert _level_for_volume(residual, flowed, FlowTestConfig()) == expected


def test_end_to_end_non_boundary_video(tmp_path) -> None:
    video = generate_video(tmp_path / "level2.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0))
    result = grade_video(video)
    assert result.status == "graded", result.to_dict()
    assert result.iddsi_level == "L2"
    assert result.release_method == "nozzle_release"
    assert result.estimated_residual_ml == pytest.approx(6.0, abs=0.20)


def test_repeated_grade_is_deterministic(tmp_path) -> None:
    video = generate_video(tmp_path / "repeat.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0))
    assert grade_video(video).to_dict() == grade_video(video).to_dict()


def test_boundary_abstention(tmp_path) -> None:
    video = generate_video(tmp_path / "boundary.mp4", SyntheticSpec(residual_ml=4.25, fps=8.0))
    result = grade_video(video)
    assert result.status == "abstain"
    assert result.iddsi_level is None
    assert "near_level_boundary" in result.abstention_reasons
    assert result.estimated_residual_ml == pytest.approx(4.25, abs=0.20)


def test_wrong_syringe_abstention(tmp_path) -> None:
    video = generate_video(
        tmp_path / "wrong.mp4",
        SyntheticSpec(residual_ml=6.0, fps=8.0, barrel_length_mm=70.0),
    )
    result = grade_video(video)
    assert result.status == "abstain"
    assert "wrong_or_unverified_syringe" in result.abstention_reasons


def test_non_vertical_abstention(tmp_path) -> None:
    video = generate_video(tmp_path / "tilted.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0, tilt_deg=8.0))
    result = grade_video(video)
    assert result.status == "abstain"
    assert "syringe_not_vertical" in result.abstention_reasons


def test_bubbles_abstention(tmp_path) -> None:
    video = generate_video(tmp_path / "bubbles.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0, bubbles=True))
    result = grade_video(video)
    assert result.status == "abstain"
    assert "bubbles_or_lumps_detected" in result.abstention_reasons


def test_occlusion_abstention(tmp_path) -> None:
    video = generate_video(tmp_path / "occluded.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0, occlusion=True))
    result = grade_video(video)
    assert result.status == "abstain"
    assert "syringe_occluded" in result.abstention_reasons


def test_short_video_abstention(tmp_path) -> None:
    video = generate_video(
        tmp_path / "short.mp4",
        SyntheticSpec(residual_ml=6.0, fps=8.0, post_release_s=8.0),
    )
    result = grade_video(video)
    assert result.status == "abstain"
    assert "insufficient_duration" in result.abstention_reasons


def test_no_flow_maps_to_level4(tmp_path) -> None:
    video = generate_video(tmp_path / "level4.mp4", SyntheticSpec(residual_ml=10.0, fps=8.0))
    result = grade_video(video)
    assert result.status == "graded", result.to_dict()
    assert result.iddsi_level == "L4"


def test_synthetic_mae_report_is_explicitly_synthetic(tmp_path) -> None:
    report = evaluate_synthetic(tmp_path, residuals=[0.25, 2.5, 6.0, 9.0])
    assert report["count"] == 4
    assert report["mae_ml"] is not None
    assert float(report["mae_ml"]) <= 0.20
    assert "not real-video validation" in str(report["scope"])


def test_cli_writes_json(tmp_path, monkeypatch) -> None:
    video = generate_video(tmp_path / "cli.mp4", SyntheticSpec(residual_ml=2.5, fps=8.0))
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["flowtest.grade", str(video), "--out", str(output)])
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "graded"
    assert payload["iddsi_level"] == "L1"
