"""Config-driven train-time augmentation for the manifest pipeline.

The v0.7 recipe trains with a single random horizontal flip at p=0.5 drawn
inside :func:`train.data.build_transform`. Nothing else touches the pixels: no
crop, no colour jitter, no blur. This module adds an opt-in pipeline that is
strictly stronger, so an augmentation study is a one-key config change.

Three properties are load-bearing:

* Every operation happens in PIL space *before* the model's own image
  processor, so the SigLIP-2 resize and normalisation are byte-identical to the
  un-augmented path. Two augmentation settings therefore differ in the
  augmentation and in nothing else.
* Randomness is drawn from the same per-worker ``torch`` RNG the shipped flip
  uses, so a (seed, num_workers, batch_size) triple reproduces exactly.
* The class is deliberately NOT :class:`train.realdata.SeededAugment`. That
  class owns its own resize and normalisation, so dropping it in here would
  silently rescale the inputs of the manifest pipeline and confound the
  comparison with the augmentation change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

#: Config keys understood by :meth:`AugmentSettings.from_mapping`. Anything
#: else is a typo and is rejected rather than silently ignored: a misspelled
#: augmentation key that quietly disables augmentation would corrupt a study.
KEYS = (
    "horizontal_flip_p",
    "crop_scale",
    "crop_aspect",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "blur_p",
    "blur_radius",
)


@dataclass(frozen=True)
class AugmentSettings:
    """Magnitudes for the photometric pipeline. Ranges are sampled uniformly.

    The defaults are a *stronger-than-shipped* food-photo preset: they assume
    the plate roughly fills the frame, that hue and saturation carry real
    signal about the food, and that the dominant nuisance factors in web food
    photography are crop/zoom, exposure, white balance and focus. Every value
    therefore stays inside the range torchvision uses for ImageNet, and none is
    large enough to erase the colour the level labels depend on.
    """

    horizontal_flip_p: float = 0.5
    crop_scale: tuple[float, float] = (0.55, 1.0)
    crop_aspect: tuple[float, float] = (0.75, 1.3333333333333333)
    brightness: float = 0.30
    contrast: float = 0.30
    saturation: float = 0.35
    hue: float = 0.05
    blur_p: float = 0.15
    blur_radius: float = 1.5

    def validate(self) -> None:
        if not 0.0 <= self.horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be in [0, 1]")
        for name in ("brightness", "contrast", "saturation"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0.0 <= self.hue <= 0.5:
            raise ValueError("hue must be in [0, 0.5] (fraction of the hue circle)")
        if not 0.0 <= self.blur_p <= 1.0:
            raise ValueError("blur_p must be in [0, 1]")
        if self.blur_radius < 0.0:
            raise ValueError("blur_radius must be >= 0")
        for name in ("crop_scale", "crop_aspect"):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"{name} must satisfy 0 < low <= high")
        if self.crop_scale[1] > 1.0:
            raise ValueError("crop_scale[1] must be <= 1.0")
        if self.crop_aspect[0] <= 0.0:
            raise ValueError("crop_aspect[0] must be > 0")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AugmentSettings":
        """Build settings from a YAML mapping, rejecting unknown keys."""
        unknown = sorted(set(mapping) - set(KEYS))
        if unknown:
            raise ValueError(f"unknown data.augment keys: {unknown}; expected any of {list(KEYS)}")
        values: dict[str, Any] = {}
        for name in KEYS:
            if name not in mapping:
                continue
            value = mapping[name]
            if name in {"crop_scale", "crop_aspect"}:
                pair = tuple(float(item) for item in value)  # type: ignore[union-attr]
                if len(pair) != 2:
                    raise ValueError(f"{name} must be a [low, high] pair")
                values[name] = pair
            else:
                values[name] = float(value)
        return cls(**values)


def _uniform(magnitude: float) -> float:
    """A factor in ``[1 - magnitude, 1 + magnitude)``."""
    return 1.0 + (2.0 * float(torch.rand(())) - 1.0) * magnitude


def _shift_hue(image: Image.Image, delta: float) -> Image.Image:
    """Rotate hue by ``delta`` turns of the colour circle (PIL HSV is 0..255)."""
    if delta == 0.0:
        return image
    hsv = np.asarray(image.convert("HSV")).astype("float32")
    hsv[:, :, 0] = (hsv[:, :, 0] + delta * 255.0) % 256.0
    shifted = Image.fromarray(hsv.round().clip(0, 255).astype("uint8"), "HSV")
    return shifted.convert("RGB")


class PhotometricAugment:
    """Random resized crop + horizontal flip + colour jitter (+ light blur).

    PIL in, PIL out. The caller's image processor still performs the resize to
    the model's input resolution and the normalisation, so this composes with
    ``siglip2``, ``resnet18`` and ``tiny_cnn`` alike.
    """

    def __init__(self, settings: AugmentSettings | None = None):
        self.settings = settings or AugmentSettings()
        self.settings.validate()

    def __call__(self, image: Image.Image) -> Image.Image:
        cfg = self.settings
        image = image.convert("RGB")
        width, height = image.size
        if width < 2 or height < 2:
            return image

        # Random resized crop: area scaled by crop_scale, aspect jittered
        # geometrically so the distribution of shapes is symmetric in log space.
        scale = float(torch.rand(())) * (cfg.crop_scale[1] - cfg.crop_scale[0]) + cfg.crop_scale[0]
        log_aspect = (
            float(torch.rand(())) * (math.log(cfg.crop_aspect[1]) - math.log(cfg.crop_aspect[0]))
            + math.log(cfg.crop_aspect[0])
        )
        aspect = math.exp(log_aspect)
        area = width * height * scale
        crop_w = min(width, max(1, int(round(math.sqrt(area * aspect)))))
        crop_h = min(height, max(1, int(round(math.sqrt(area / aspect)))))
        left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
        top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
        image = image.crop((left, top, left + crop_w, top + crop_h))

        if cfg.horizontal_flip_p and float(torch.rand(())) < cfg.horizontal_flip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if cfg.brightness:
            image = ImageEnhance.Brightness(image).enhance(_uniform(cfg.brightness))
        if cfg.contrast:
            image = ImageEnhance.Contrast(image).enhance(_uniform(cfg.contrast))
        if cfg.saturation:
            image = ImageEnhance.Color(image).enhance(_uniform(cfg.saturation))
        if cfg.hue:
            image = _shift_hue(image, (2.0 * float(torch.rand(())) - 1.0) * cfg.hue)
        if cfg.blur_p and float(torch.rand(())) < cfg.blur_p:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(torch.rand(())) * cfg.blur_radius))
        return image


def from_mapping(mapping: Mapping[str, Any] | None) -> PhotometricAugment | None:
    """``None`` (key absent) means "ship as v0.7 shipped": flip only."""
    if not mapping:
        return None
    return PhotometricAugment(AugmentSettings.from_mapping(mapping))
