"""Tests for the Flow-Test video grading handler and its UI renderer.

Fixtures are tiny CPU-generated synthetic MP4s produced by the flowtest
package's own ``synthetic.generate_video`` — the same approach flowtest's own
tests use — so no camera footage or network access is needed.  cv2 is required
(by flowtest); the tests skip cleanly if it is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="flowtest grader requires OpenCV")

import crowd.app as app_module
from crowd.flowtest_ui import grade_flow_test

# Make the inner flowtest package importable the same way crowd.flowtest_ui
# does (sibling ``flowtest/`` directory on sys.path).
_FLOWTEST_DIR = Path(__file__).resolve().parent.parent.parent / "flowtest"
if str(_FLOWTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOWTEST_DIR))

from flowtest.synthetic import SyntheticSpec, generate_video  # noqa: E402


def _make_video(tmp_path: Path, **spec_kwargs) -> Path:
    spec = SyntheticSpec(**spec_kwargs)
    return generate_video(tmp_path / "fixture.mp4", spec)


def test_grade_flow_test_returns_level_for_clean_capture(tmp_path: Path) -> None:
    video = _make_video(tmp_path, residual_ml=6.0)
    result = grade_flow_test(video)

    assert result["status"] == "graded"
    assert result["iddsi_level"] == "L2"  # 4–<8 mL residual band
    assert isinstance(result["estimated_residual_ml"], float)
    assert result["abstention_reasons"] == []


def test_grade_flow_test_abstains_without_fiducial(tmp_path: Path) -> None:
    video = _make_video(tmp_path, residual_ml=6.0, fiducial=False)
    result = grade_flow_test(video)

    assert result["status"] == "abstain"
    assert result["iddsi_level"] is None
    assert "scale_not_verified" in result["abstention_reasons"]
    # Human-readable descriptions are guaranteed present for every code.
    descriptions = result["diagnostics"]["abstention_code_descriptions"]
    for code in result["abstention_reasons"]:
        assert code in descriptions
        assert descriptions[code]


def test_grade_flow_test_error_on_missing_video() -> None:
    result = grade_flow_test(None)
    assert result["status"] == "error"
    assert result["error"]


def test_render_graded_result_shows_level_bilingually(tmp_path: Path) -> None:
    video = _make_video(tmp_path, residual_ml=6.0)
    markdown = app_module.render_flow_test_markdown(grade_flow_test(video))

    assert "L2" in markdown
    assert "流量測試量度" in markdown  # 繁中 line present
    assert "research only" in markdown


def test_render_abstain_result_shows_reasons_and_never_a_pass(tmp_path: Path) -> None:
    video = _make_video(tmp_path, residual_ml=6.0, fiducial=False)
    markdown = app_module.render_flow_test_markdown(grade_flow_test(video))

    assert "Abstain" in markdown
    assert "棄權" in markdown
    assert "scale_not_verified" in markdown
    assert "repeat the official physical test" in markdown
    # An abstain must not be rendered as a level/pass.
    assert "pass" not in markdown.lower()


def test_render_error_result_is_bilingual() -> None:
    markdown = app_module.render_flow_test_markdown(grade_flow_test(None))
    assert "無法讀取影片" in markdown
    assert "Could not read the video" in markdown


def test_video_ui_handler_returns_json_and_bilingual_rendering(tmp_path: Path) -> None:
    video = _make_video(tmp_path, residual_ml=6.0, fiducial=False)

    payload, markdown = app_module.handle_flowtest_video(video)

    assert payload["status"] == "abstain"
    assert "scale_not_verified" in payload["abstention_reasons"]
    assert payload["diagnostics"]["abstention_code_descriptions"]["scale_not_verified"]
    assert "Abstention reasons" in markdown
    assert "棄權原因" in markdown
    assert "scale_not_verified" in markdown


def test_video_ui_handler_missing_input_returns_error_json() -> None:
    payload, markdown = app_module.handle_flowtest_video(None)
    assert payload["status"] == "error"
    assert payload["abstention_reasons"] == []
    assert "Could not read the video" in markdown
    assert "無法讀取影片" in markdown


def test_app_exports_flow_test_entrypoint() -> None:
    # The Gradio app module must expose the wired grader entrypoint.
    assert app_module.grade_flow_test is grade_flow_test
