"""Tests for the opt-in augmentation added by the exp04_aug lane.

The load-bearing claim is that ``data.augment`` absent reproduces the v0.7
transform bit-for-bit, so an augmentation study changes augmentation and
nothing else.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from train.augment import AugmentSettings, PhotometricAugment, from_mapping
from train.data import build_transform

# The shared SigLIP-2 base checkpoint lives in the shared tree's models_local/,
# a few levels above this lane clone. Search for it rather than naming an
# absolute host path (the release leak scan BLOCKs on those).
def _local_model_dir() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        *( [Path(os.environ["IDDSI_MODEL_DIR"])] if os.environ.get("IDDSI_MODEL_DIR") else [] ),
        *(p / "models_local" / "siglip2-base-patch16-224" for p in here.parents[:6]),
    ]
    return next((c for c in candidates if c.is_dir()), None)


LOCAL_MODEL = _local_model_dir()


def _photo(size: int = 240, seed: int = 0) -> Image.Image:
    """A synthetic plate-ish photo: smooth colour gradient plus noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack(
        [
            (xx / size * 200 + 20),
            (yy / size * 160 + 40),
            ((xx + yy) / (2 * size) * 120 + 60),
        ],
        axis=-1,
    )
    base = np.clip(base + rng.normal(0, 12, base.shape), 0, 255)
    return Image.fromarray(base.astype("uint8"), "RGB")


def _v07_siglip2_transform(image: Image.Image, model_name: str) -> torch.Tensor:
    """The pre-exp04_aug siglip2 transform, written out literally.

    Kept here so a future edit to ``build_transform`` that changes the default
    behaviour fails this test instead of silently moving the baseline.
    """
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name)
    if bool(torch.rand(()) < 0.5):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]


def _v07_basic_transform(image: Image.Image, size: int) -> torch.Tensor:
    """The pre-exp04_aug ``BasicImageTransform``, written out literally."""
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    if bool(torch.rand(()) < 0.5):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def test_absent_augment_key_means_flip_only():
    assert from_mapping(None) is None
    assert from_mapping({}) is None


def test_default_settings_are_stricter_than_shipped():
    settings = AugmentSettings()
    settings.validate()
    # Shipped v0.7: flip only. Everything below must actually do something.
    assert settings.crop_scale[0] < 1.0
    assert settings.brightness > 0 and settings.contrast > 0 and settings.saturation > 0
    assert settings.hue > 0 and settings.blur_p > 0 and settings.blur_radius > 0


def test_photometric_augment_is_deterministic_per_torch_seed():
    image = _photo()
    augment = PhotometricAugment(AugmentSettings(crop_scale=(0.6, 0.6), hue=0.1))
    torch.manual_seed(5)
    first = np.asarray(augment(image))
    torch.manual_seed(5)
    second = np.asarray(augment(image))
    torch.manual_seed(6)
    third = np.asarray(augment(image))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_photometric_augment_crop_scale_is_an_area_fraction():
    """crop_scale scales AREA, so a 0.5 area fraction is a 0.707 side."""
    image = _photo(size=300)
    augment = PhotometricAugment(
        AugmentSettings(crop_scale=(0.5, 0.5), crop_aspect=(1.0, 1.0), hue=0.0, blur_p=0.0)
    )
    torch.manual_seed(3)
    out = augment(image)
    assert out.mode == "RGB"
    assert out.size == (212, 212)


def test_photometric_augment_never_grows_the_image():
    augment = PhotometricAugment()
    torch.manual_seed(3)
    out = augment(_photo(size=8))
    width, height = out.size
    assert 1 <= width <= 8 and 1 <= height <= 8


def test_hue_jitter_stays_inside_the_colour_circle():
    augment = PhotometricAugment(AugmentSettings(hue=0.5))
    torch.manual_seed(11)
    out = augment(_photo(size=64))
    assert np.asarray(out).dtype == np.uint8


@pytest.mark.parametrize(
    "mapping",
    [
        {"brightness": 1.5},
        {"contrast": -0.1},
        {"hue": 0.9},
        {"blur_p": 2.0},
        {"blur_radius": -1.0},
        {"crop_scale": [0.0, 1.0]},
        {"crop_scale": [0.9, 0.8]},
        {"crop_scale": [1.0, 1.2]},
        {"crop_aspect": [0.0, 4.0]},
        {"crop_aspect": [2.0, 1.0]},
        {"horizontal_flip_p": 1.5},
        {"crop_scale": [0.5]},
    ],
)
def test_invalid_settings_are_rejected(mapping):
    with pytest.raises(ValueError):
        AugmentSettings.from_mapping(mapping).validate()


def test_unknown_key_is_a_hard_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="unknown data.augment keys"):
        AugmentSettings.from_mapping({"brigtness": 0.3})


def test_from_mapping_coerces_yaml_lists_to_float_pairs():
    settings = AugmentSettings.from_mapping({"crop_scale": [0.5, 0.9], "brightness": 1e-1})
    assert settings.crop_scale == (0.5, 0.9)
    assert settings.brightness == pytest.approx(0.1)


def test_build_transform_baseline_is_bit_identical_to_v07():
    """No augment_config must give exactly the old resize+flip+normalise."""
    image = _photo(size=96)
    for seed in range(6):
        torch.manual_seed(seed)
        via_builder = build_transform("tiny_cnn", "test", 32, True)(image)
        torch.manual_seed(seed)
        reference = _v07_basic_transform(image, 32)
        assert torch.equal(via_builder, reference)


def test_build_transform_eval_split_ignores_augment_config():
    """A strong augment_config must never reach val/test."""
    image = _photo(size=96)
    mapping = {"crop_scale": [0.2, 0.2], "brightness": 0.9, "hue": 0.2, "blur_p": 1.0}
    for seed in range(4):
        torch.manual_seed(seed)
        eval_out = build_transform("tiny_cnn", "test", 32, False, mapping)(image)
        torch.manual_seed(seed)
        plain = build_transform("tiny_cnn", "test", 32, False)(image)
        assert torch.equal(eval_out, plain)


def test_build_transform_train_split_does_apply_augment_config():
    image = _photo(size=96)
    mapping = {
        "crop_scale": [0.3, 0.3], "brightness": 0.0, "contrast": 0.0,
        "saturation": 0.0, "hue": 0.0, "blur_p": 0.0,
    }
    changed = 0
    for seed in range(6):
        torch.manual_seed(seed)
        augmented = build_transform("tiny_cnn", "test", 32, True, mapping)(image)
        torch.manual_seed(seed)
        plain = build_transform("tiny_cnn", "test", 32, True)(image)
        changed += int(not torch.equal(augmented, plain))
    assert changed == 6


@pytest.mark.skipif(LOCAL_MODEL is None, reason="local SigLIP-2 weights not present")
def test_siglip2_default_matches_the_v07_transform_exactly():
    """The strongest control: the shipped path is bit-identical to v0.7."""
    image = _photo(size=260)
    flipped = 0
    for seed in range(8):
        torch.manual_seed(seed)
        via_builder = build_transform("siglip2", str(LOCAL_MODEL), 224, True)(image)
        torch.manual_seed(seed)
        reference = _v07_siglip2_transform(image, str(LOCAL_MODEL))
        assert torch.equal(via_builder["pixel_values"], reference)
        flipped += int(
            not torch.equal(
                reference,
                build_transform("siglip2", str(LOCAL_MODEL), 224, False)(image)["pixel_values"],
            )
        )
    # Sanity: the comparison above is only meaningful if the flip actually fires.
    assert flipped >= 1


@pytest.mark.skipif(LOCAL_MODEL is None, reason="local SigLIP-2 weights not present")
def test_siglip2_augmented_output_keeps_processor_shape_and_scale():
    """SigLIP-2 patchifies to [n_patches, patch_dim]; augmentation must not change that."""
    image = _photo(size=260)
    mapping = {"crop_scale": [0.55, 1.0], "brightness": 0.3, "hue": 0.05}
    torch.manual_seed(4)
    pixels = build_transform("siglip2", str(LOCAL_MODEL), 224, True, mapping)(image)["pixel_values"]
    torch.manual_seed(4)
    plain = build_transform("siglip2", str(LOCAL_MODEL), 224, True)(image)["pixel_values"]
    assert pixels.shape == plain.shape == (196, 768)
    # Same processor, so the same [-1, 1] normalisation and the same dynamic range.
    assert pixels.min() >= -1.0001 and pixels.max() <= 1.0001
    assert not torch.equal(pixels, plain)


def test_engine_passes_the_augment_config_to_the_train_transform(tmp_path: Path):
    """End-to-end on synthetic CPU fixtures: the key reaches build_transform."""
    from fixtures.generate import generate_fixture
    from train.engine import run_experiment

    manifest = generate_fixture(tmp_path / "fixture", per_level_per_split=1, size=24)
    cfg = {
        "experiment": {"name": "aug_e2e", "output_dir": str(tmp_path / "runs"), "seeds": [3]},
        "data": {
            "manifest": str(manifest), "image_size": 24, "num_workers": 0, "batch_size": 5,
            "level_field": "level", "split_field": "split", "group_field": "group_id",
            "synthetic": True,
            "augment": {"crop_scale": [0.5, 0.5], "brightness": 0.3, "hue": 0.05},
        },
        "model": {"kind": "tiny_cnn", "name": "test", "dropout": 0.0, "pretrained": False},
        "training": {
            "weight_decay": 0.0, "class_balance_beta": 0.99, "gradient_clip_norm": 1.0,
            "amp": False, "device": "cpu",
            "phases": [{"name": "head", "epochs": 1, "lr": 0.001, "unfreeze_last_blocks": 0}],
        },
        "calibration": {
            "target_under_fnr": 0.4, "max_abstention": 0.4,
            "temperature_min": 0.5, "temperature_max": 2.0, "temperature_steps": 5,
        },
        "evaluation": {"bootstrap_samples": 5, "bootstrap_seed": 2},
    }
    summaries = run_experiment(cfg)
    assert (Path(summaries[0]["output_dir"]) / "metrics.json").exists()
