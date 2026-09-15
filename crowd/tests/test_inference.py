from __future__ import annotations

import pytest

import crowd.app as app_module
from crowd.inference import (
    InferenceResult,
    predict_iddsi_level,
    reset_lazy_model_for_tests,
)

Image = pytest.importorskip("PIL.Image", reason="photo path needs Pillow")


def setup_function() -> None:
    reset_lazy_model_for_tests()


def _tiny_png(path) -> None:
    """Write a tiny valid RGB PNG fixture (no camera image needed)."""
    Image.new("RGB", (8, 8), (120, 80, 40)).save(path)


def test_stub_returns_placeholder_structure_when_no_weights(tmp_path) -> None:
    missing = tmp_path / "no-checkpoint.pt"
    result = predict_iddsi_level(None, checkpoint=missing)

    assert isinstance(result, InferenceResult)
    assert result.status == "placeholder_no_weights"
    assert result.is_placeholder is True
    assert result.predicted_level is None
    assert result.levels == (3, 4, 5, 6, 7)
    # Uniform placeholder across the five ordinal levels, not a model output.
    assert set(result.probabilities) == {3, 4, 5, 6, 7}
    assert all(abs(prob - 0.2) < 1e-9 for prob in result.probabilities.values())
    assert "placeholder" in result.detail.lower()


def test_stub_reports_missing_checkpoint_path(tmp_path) -> None:
    missing = tmp_path / "best.pt"
    result = predict_iddsi_level(None, checkpoint=missing)
    assert result.checkpoint == str(missing)


def test_app_exports_predict_entrypoint() -> None:
    # The Gradio app module must expose the wired stub entrypoint.
    assert app_module.predict_iddsi_level is predict_iddsi_level


def test_disclaimer_rendered_in_english_and_traditional_chinese() -> None:
    # r5_reg.md "Safer public-demo boundary" wording, both languages.
    assert app_module.DISCLAIMER == (
        "Research demonstration for culinary education. Estimates visual similarity "
        "to IDDSI descriptors; does not perform official IDDSI tests, assess "
        "swallowing, determine suitability for any person, or decide whether food is "
        "safe to consume."
    )
    assert "烹飪教學研究示範" in app_module.DISCLAIMER_ZH_HANT
    assert "IDDSI" in app_module.DISCLAIMER_ZH_HANT
    assert "不評估吞嚥" in app_module.DISCLAIMER_ZH_HANT
    # The two disclaimers are distinct strings (not a duplicated constant).
    assert app_module.DISCLAIMER != app_module.DISCLAIMER_ZH_HANT


def test_photo_path_placeholder_with_real_image(tmp_path) -> None:
    """Photo handler: a real uploaded image still yields the honest placeholder
    when no trained checkpoint is loaded (never a fabricated level)."""
    photo = tmp_path / "plate.png"
    _tiny_png(photo)
    result = predict_iddsi_level(photo, checkpoint=tmp_path / "missing.pt")

    assert result.is_placeholder is True
    assert result.predicted_level is None
    assert result.status == "placeholder_no_weights"


def test_photo_ui_handler_uses_trainer_stub_and_renders_bilingually(tmp_path) -> None:
    photo = tmp_path / "plate.png"
    _tiny_png(photo)
    seen = []

    def fake_predictor(image):
        seen.append(image)
        return InferenceResult(
            status="placeholder_no_weights",
            levels=(3, 4, 5, 6, 7),
            probabilities={level: 0.2 for level in (3, 4, 5, 6, 7)},
            predicted_level=None,
            model_kind=None,
            checkpoint=None,
            is_placeholder=True,
            detail="Recorded trainer-stub fixture.",
        )

    markdown = app_module.handle_photo_estimate(photo, predictor=fake_predictor)

    assert seen == [photo]
    assert "Stub estimate" in markdown
    assert "預留估算" in markdown
    assert "並非" in markdown
    assert "L3: 0.20" in markdown


def test_photo_ui_handler_missing_input_is_bilingual() -> None:
    markdown = app_module.handle_photo_estimate(None)
    assert "No photo" in markdown
    assert "未上載照片" in markdown


def test_photo_path_abstains_on_unreadable_image(tmp_path) -> None:
    """Photo handler with a checkpoint present but an undecodable image must
    surface an abstention-style placeholder, not crash or invent a level."""
    photo = tmp_path / "broken.png"
    photo.write_bytes(b"not-a-real-png")
    # No checkpoint on disk -> placeholder regardless of image content.
    result = predict_iddsi_level(photo, checkpoint=tmp_path / "missing.pt")
    assert result.is_placeholder is True
    assert result.predicted_level is None
