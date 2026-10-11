"""Tests for the IDDSI-Open Space inference core (``space/core.py``).

CPU-only: no GPU, no network. Run from the repository root:

    python -m pytest -q space/tests

What these tests defend, in order of importance:

1. **No silent wrong level.** With no trained weights the demo must emit no
   level at all, not a plausible-looking one.  In particular a *uniform*
   distribution decoded by the rank rule gives L5, so the placeholder path must
   never reach the decoder.
2. **The CORAL port cannot drift** from ``trainer/train/model.py``.
3. **Fail-closed.** Every failure mode returns the abstention string and an
   empty/placeholder distribution, never a number.
4. **The weighted path works** when a checkpoint exists, and honours the
   abstention gate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINER_DIR = _REPO_ROOT / "trainer"
if str(_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINER_DIR))

from space import core  # noqa: E402

try:  # torch is present in the training container; the tests that need it skip otherwise.
    import torch
except ImportError:  # pragma: no cover - depends on the environment.
    torch = None  # type: ignore[assignment]

needs_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")

LEVELS = (3, 4, 5, 6, 7)


# --------------------------------------------------------------------------
# 1. No silent wrong level
# --------------------------------------------------------------------------


def test_placeholder_emits_no_level_and_a_uniform_distribution(tmp_path, monkeypatch):
    monkeypatch.setenv("IDDSI_SPACE_CHECKPOINT", str(tmp_path / "does_not_exist.pt"))
    monkeypatch.delenv("IDDSI_SPACE_CHECKPOINT", raising=False)

    result = core.predict(_synthetic_image())

    assert result.status == "placeholder_no_weights"
    assert result.is_placeholder is True
    assert result.abstained is True
    assert result.decoded_level is None
    assert result.confidence is None
    assert result.probabilities == {level: pytest.approx(0.2) for level in LEVELS}
    assert result.abstention_text == "unclear — perform physical test"


def test_uniform_distribution_would_decode_to_l5_which_is_why_we_never_decode():
    """The trap: the rank rule applied to a uniform distribution yields L5.

    If a future edit makes the placeholder path call the decoder, this test
    fails and the regression is caught before it ships as a silent L5.
    """

    import numpy as np

    uniform = np.full((1, 5), 0.2)
    indices, _ = core.coral_decode(uniform)
    assert int(indices[0]) == 2  # -> L5
    assert LEVELS[int(indices[0])] == 5
    # ...and the shipped placeholder path therefore must not produce a level.
    assert core.predict(_synthetic_image()).decoded_level is None


def test_placeholder_banner_and_disclaimer_are_always_present():
    result = core.predict(_synthetic_image())
    assert "PLACEHOLDER" in result.banner
    assert result.disclaimer == core.DISCLAIMER
    assert result.disclaimer_zh_hant == core.DISCLAIMER_ZH_HANT
    rendered = result.to_markdown()
    assert core.DISCLAIMER in rendered
    assert core.DISCLAIMER_ZH_HANT in rendered
    assert "PLACEHOLDER" in rendered
    assert "L5" not in result.verdict_line


# --------------------------------------------------------------------------
# 2. The CORAL port cannot drift from the trainer
# --------------------------------------------------------------------------


@needs_torch
def test_coral_port_is_numerically_identical_to_the_trainer():
    from train.model import coral_decode as trainer_decode
    from train.model import coral_logits_to_probs as trainer_probs

    generator = torch.Generator().manual_seed(20261012)
    worst_probs = worst_decode = 0.0
    for _ in range(200):
        logits = torch.randn(1, 4, generator=generator) * 3.0
        worst_probs = max(
            worst_probs,
            float((core.coral_logits_to_probs(logits) - trainer_probs(logits)).abs().max()),
        )
        space_probs = core.coral_logits_to_probs(logits).numpy()
        space_index, space_conf = core.coral_decode(space_probs)
        trainer_index, trainer_conf = trainer_decode(space_probs)
        assert int(space_index[0]) == int(trainer_index[0])
        worst_decode = max(worst_decode, float(abs(space_conf[0] - trainer_conf[0])))
    assert worst_probs == 0.0
    assert worst_decode == 0.0


def test_disclaimer_is_verbatim_the_controlled_public_demo_boundary():
    """The wording is controlled; assert it against crowd/app.py, its source.

    ``crowd/app.py`` records the English string as the source of truth for
    r5_reg.md "Safer public-demo boundary".  Comparing the imported constants
    (rather than grepping the file) means a re-wrapping or a stray space in
    either place is caught.
    """

    crowd_app = _REPO_ROOT / "crowd" / "app.py"
    if not crowd_app.is_file():  # pragma: no cover - repo layout guard.
        pytest.skip("crowd/app.py not present")
    if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - conftest guarantees this.
        sys.path.insert(0, str(_REPO_ROOT))
    import crowd.app as crowd_module

    assert core.DISCLAIMER == crowd_module.DISCLAIMER
    assert core.DISCLAIMER_ZH_HANT == crowd_module.DISCLAIMER_ZH_HANT
    # The abstention string is a control principle in the hazard log.
    hazard_log = _REPO_ROOT / "hf" / "HAZARD_LOG.md"
    if hazard_log.is_file():
        assert core.ABSTENTION_TEXT in hazard_log.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 3. Fail-closed
# --------------------------------------------------------------------------


def test_corrupt_checkpoint_fails_closed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch payload")
    monkeypatch.setenv("IDDSI_SPACE_CHECKPOINT", str(checkpoint))

    result = core.predict(_synthetic_image())

    assert result.abstained is True
    assert result.decoded_level is None
    assert result.status == "error"
    assert result.probabilities == {}


@needs_torch
def test_weight_mismatch_fails_closed(tmp_path, monkeypatch):
    """A checkpoint whose metadata claims SigLIP-2 but whose weights are a
    TinyCNN must fail closed, not load partially and emit a confident level."""

    from train.model import OrdinalImageModel

    checkpoint = tmp_path / "mismatched.pt"
    torch.save(
        {
            "metadata": {"model_kind": "siglip2", "model_name": "unused", "levels": list(LEVELS)},
            "state_dict": OrdinalImageModel(kind="tiny_cnn").state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setenv("IDDSI_SPACE_CHECKPOINT", str(checkpoint))

    result = core.predict(_synthetic_image())

    assert result.status == "error"
    assert result.abstained is True
    assert result.decoded_level is None
    assert result.probabilities == {}


@needs_torch
def test_truncated_state_dict_fails_closed(tmp_path, monkeypatch):
    """A checkpoint missing the CORAL head must fail closed."""

    from train.model import OrdinalImageModel

    state = OrdinalImageModel(kind="tiny_cnn").state_dict()
    truncated = {key: value for key, value in state.items() if not key.startswith("head.")}
    assert truncated, "expected the head keys to be separable from the backbone keys"

    checkpoint = tmp_path / "truncated.pt"
    torch.save(
        {
            "metadata": {"model_kind": "tiny_cnn", "model_name": "", "levels": list(LEVELS)},
            "state_dict": truncated,
        },
        checkpoint,
    )
    monkeypatch.setenv("IDDSI_SPACE_CHECKPOINT", str(checkpoint))

    result = core.predict(_synthetic_image())

    assert result.status == "error"
    assert result.abstained is True
    assert result.decoded_level is None


# --------------------------------------------------------------------------
# 4. The weighted path works when a checkpoint exists
# --------------------------------------------------------------------------


@needs_torch
def test_weighted_path_returns_a_valid_distribution_and_honours_the_gate(tmp_path, monkeypatch):
    from train.model import OrdinalImageModel

    model = OrdinalImageModel(kind="tiny_cnn")
    model.eval()
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "metadata": {"model_kind": "tiny_cnn", "model_name": "", "levels": list(LEVELS)},
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setenv("IDDSI_SPACE_CHECKPOINT", str(checkpoint))

    image = _synthetic_image()

    # A permissive gate: the distribution must be a real, normalised one and
    # the decoded level must be inside L3..L7.
    permissive = core.predict(image, abstain_threshold=0.0)
    assert permissive.status == "ok"
    assert permissive.is_placeholder is False
    assert permissive.abstained is False
    assert permissive.decoded_level in LEVELS
    assert sum(permissive.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(permissive.probabilities) == set(LEVELS)
    assert permissive.banner == ""
    assert "not a texture test" in permissive.verdict_line

    # An impossible gate: must abstain and name no level.
    strict = core.predict(image, abstain_threshold=1.0)
    assert strict.status == "ok"
    assert strict.abstained is True
    assert strict.decoded_level is None
    assert "abstention threshold" in strict.banner


@needs_torch
def test_default_threshold_is_the_most_conservative_v07_calibration():
    """Pin the provenance of the default gate to the shipped calibration files."""

    import statistics

    thresholds = []
    for path in sorted((_REPO_ROOT / "results" / "v07_release").glob("v07_seed*_calibration.json")):
        thresholds.append(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    assert len(thresholds) == 5, f"expected 5 v0.7 calibrations, found {len(thresholds)}"
    assert core.DEFAULT_ABSTENTION_THRESHOLD == pytest.approx(max(thresholds))
    assert core.DEFAULT_ABSTENTION_THRESHOLD > statistics.median(thresholds)


@needs_torch
def test_model_input_keys_match_the_trainer_forward_signature():
    """The Space must forward exactly the keys ``engine._forward`` forwards.

    A key dropped here is a silent architecture mismatch; a key added here is a
    TypeError at inference time.  Either way the Space would serve a different
    model than the trainer trained.
    """

    import re

    engine = _REPO_ROOT / "trainer" / "train" / "engine.py"
    if not engine.is_file():  # pragma: no cover - repo layout guard.
        pytest.skip("trainer/train/engine.py not present")
    source = engine.read_text(encoding="utf-8")
    match = re.search(
        r'for key in \(([^)]*)\).*?if key in batch', source, flags=re.DOTALL
    )
    assert match, "could not locate the key tuple in engine._forward"
    trainer_keys = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert set(core._MODEL_INPUT_KEYS) == trainer_keys


@needs_torch
def test_flatten_fallback_matches_the_real_siglip2_processor():
    """The offline fallback must reproduce the processor's NaViT flattening.

    ``_preprocess`` prefers ``AutoImageProcessor``.  When the processor cannot
    be loaded (an offline Space with no backbone directory) the fallback below
    must produce the *same* tensor, or the weighted path would silently change
    meaning between environments.
    """

    backbone = Path(os.getenv("IDDSI_SPACE_BACKBONE", core.DEFAULT_BACKBONE))
    if not backbone.is_dir():  # pragma: no cover - depends on the checkout.
        pytest.skip(f"backbone not present: {backbone}")

    from transformers import AutoImageProcessor  # noqa: PLC0415

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(20261012)
    image = Image.fromarray((rng.random((300, 300, 3)) * 255).astype("uint8"), mode="RGB")

    processor = AutoImageProcessor.from_pretrained(str(backbone))
    expected = processor(images=image, return_tensors="pt")

    fallback = core._preprocess_flattened_patches(torch, image)

    assert set(expected) <= set(fallback) | {"pixel_attention_mask"}
    assert fallback["pixel_values"].shape == expected["pixel_values"].shape
    assert torch.allclose(fallback["pixel_values"], expected["pixel_values"], atol=2e-2)
    assert torch.equal(fallback["spatial_shapes"].long(), expected["spatial_shapes"].long())
    assert fallback["pixel_values"].shape[1] == 196  # 14 x 14 patches at 224 px / 16 px


# --------------------------------------------------------------------------
# Space packaging
# --------------------------------------------------------------------------


def test_space_readme_front_matter_is_valid_and_declares_gradio():
    """``space/README.md`` must carry valid HF Space SDK front matter.

    A malformed front matter makes Hugging Face fall back to the default
    Streamlit SDK, which then fails to find an ``app.py`` entrypoint and the
    Space never starts.  This is the one packaging defect that is invisible
    until deploy time.
    """

    readme = _REPO_ROOT / "space" / "README.md"
    assert readme.is_file(), "space/README.md is missing"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "Space front matter must be the first thing in the file"

    closing = text.find("\n---\n", 4)
    assert closing > 0, "Space front matter is not closed by a --- line"
    front_matter = text[4:closing]

    import yaml  # noqa: PLC0415

    parsed = yaml.safe_load(front_matter)
    assert isinstance(parsed, dict), "front matter must parse to a mapping"

    for key in ("title", "emoji", "colorFrom", "colorTo", "sdk", "sdk_version", "app_file"):
        assert key in parsed, f"front matter is missing required key {key!r}"
    assert parsed["sdk"] == "gradio", f"expected sdk: gradio, got {parsed['sdk']!r}"
    assert parsed["app_file"] == "app.py"
    assert (_REPO_ROOT / "space" / parsed["app_file"]).is_file()

    requirements = (_REPO_ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    pinned = [
        line.split("==")[0].strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#") and "==" in line
    ]
    assert "gradio" in pinned, "requirements.txt must pin gradio"
    assert parsed["sdk_version"] in requirements, (
        f"front matter declares sdk_version {parsed['sdk_version']} but "
        "requirements.txt does not install it"
    )


def test_space_never_describes_itself_as_a_safety_tool():
    """The project's one non-negotiable wording rule, checked on every file
    that actually ships to Hugging Face."""

    banned = (
        "swallowing-safety classifier",
        "swallowing safety classifier",
        "safe to eat",
        "safe for the patient",
        "diagnos",
    )
    shipped = ["app.py", "core.py", "README.md", "requirements.txt"]
    for name in shipped:
        path = _REPO_ROOT / "space" / name
        assert path.is_file(), f"expected shipped file is missing: {name}"
        text = path.read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase not in text, f"space/{name} contains {phrase!r}"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _synthetic_image():
    """A 64x64 RGB image. Content is irrelevant: the placeholder path never
    touches the photo, and the weighted-path tests use a random-init head."""

    import numpy as np
    from PIL import Image

    array = (np.random.default_rng(7).random((64, 64, 3)) * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")
