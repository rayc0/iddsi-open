"""Pure inference core for the IDDSI-Open Hugging Face Space.

This module holds everything the Space needs that is *not* Gradio, so it can be
unit-tested on CPU with no UI dependency (``space/tests/test_space_core.py``).

Two execution paths, deliberately separated:

``placeholder_no_weights``
    No trained checkpoint is reachable.  The CORAL head is **unavailable**, so
    this path returns a *uniform* L3-L7 distribution, ``decoded_level=None``,
    and ``abstained=True``.  It never emits a level.  This is the path that
    ships today: the v0.7 weights are withheld (see ``hf/model_README.md``,
    "Why the weights are withheld"), so a demo that printed a level here would
    be printing an untrained number as if it were a measurement.

``ok``
    A checkpoint was found and loaded.  The SigLIP-2 backbone runs, the CORAL
    head emits cut-point logits, and the level is decoded with the ordinal
    **rank rule** ``level = #{k : P(y>k) > 0.5}`` -- NOT argmax over the
    reconstructed categorical distribution (see ``coral_decode`` below, and
    hazard H-CORAL in ``hf/HAZARD_LOG.md``).

Both paths return the controlled disclaimer text unchanged.  Nothing in this
module issues a safe/unsafe verdict, and nothing here is a swallowing-safety
assessment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: IDDSI levels the ordinal head predicts.  Mirrors ``train.model.LEVELS``.
LEVELS: tuple[int, ...] = (3, 4, 5, 6, 7)

#: Controlled disclaimer, English.  Verbatim from ``crowd/app.py`` (``DISCLAIMER``),
#: which records it as the English source of truth for r5_reg.md "Safer
#: public-demo boundary".  Do not edit the wording here: edit it there and
#: mirror it, so the two public faces cannot drift apart.
DISCLAIMER = (
    "Research demonstration for culinary education. Estimates visual similarity "
    "to IDDSI descriptors; does not perform official IDDSI tests, assess "
    "swallowing, determine suitability for any person, or decide whether food is "
    "safe to consume."
)

#: Traditional-Chinese rendering of the same disclaimer, verbatim from
#: ``crowd/app.py`` (``DISCLAIMER_ZH_HANT``).  Identical meaning.
DISCLAIMER_ZH_HANT = (
    "烹飪教學研究示範。本工具僅估算食物外觀與 IDDSI 描述之間的視覺相似度；"
    "並不執行 IDDSI 官方測試、不評估吞嚥能力、不判斷食物是否適合任何人士，"
    "亦不決定食物是否安全食用。"
)

#: Required abstention output (``hf/HAZARD_LOG.md`` control principles).  This
#: string is never replaced by a level estimate.
ABSTENTION_TEXT = "unclear — perform physical test"

#: Shown instead of a level whenever the demo is running without trained
#: weights.  Deliberately louder than a footnote.
PLACEHOLDER_BANNER = (
    "PLACEHOLDER HEAD — NO TRAINED WEIGHTS. The v0.7 SigLIP-2 CORAL weights are "
    "withheld, so the numbers below are a UNIFORM PLACEHOLDER, not a model "
    "output. No IDDSI level is predicted for your photo. This panel exists to "
    "demonstrate the interface and the fail-closed abstention path only."
)

#: Default confidence threshold for accepting a decoded level.
#:
#: Provenance: the maximum ``threshold`` across the five v0.7 validation
#: calibrations, read from ``results/v07_release/v07_seed{17,29,43,71,101}_calibration.json``
#: this run: 0.2327 / 0.5092 / 0.2394 / 0.2286 / 0.5189.  The maximum is used
#: rather than the median (0.2394) because abstention is the *safe* failure
#: direction here -- the dangerous error is under-classification (predicting a
#: softer texture than truth), so the most conservative threshold of the five
#: seeds is the defensible default.  Override with ``IDDSI_SPACE_ABSTAIN_THRESHOLD``.
DEFAULT_ABSTENTION_THRESHOLD = 0.5189002752304077

#: The keys the trainer's ``_forward`` passes to the model
#: (``trainer/train/engine.py:22``).  SigLIP-2 is NaViT-style: its processor
#: returns *flattened patches* (``pixel_values`` of shape ``[n_patches,
#: 3*patch_size**2]``) plus a ``spatial_shapes`` tensor, and the vision tower
#: indexes positional embeddings from ``spatial_shapes``.  Dropping it makes the
#: forward pass raise, which is why this tuple is explicit rather than "just the
#: pixels".
_MODEL_INPUT_KEYS = ("pixel_values", "pixel_attention_mask", "spatial_shapes")

#: Fallback geometry for ``google/siglip2-base-patch16-224`` (224 px / 16 px
#: patch -> 196 patches), used only when the image processor cannot be loaded.
_FALLBACK_IMAGE_SIZE = 224
_FALLBACK_PATCH_SIZE = 16
_FALLBACK_IMAGE_MEAN = (0.5, 0.5, 0.5)
_FALLBACK_IMAGE_STD = (0.5, 0.5, 0.5)

#: Local SigLIP-2 backbone.  Override with ``IDDSI_SPACE_BACKBONE``.
DEFAULT_BACKBONE = "/work/models_local/siglip2-base-patch16-224"

#: Where to look for trained weights.  Override with ``IDDSI_SPACE_CHECKPOINT``.
#: Both a ``.pt`` payload (``{"metadata": ..., "state_dict": ...}``) and a raw
#: state-dict ``.pt`` are accepted.
DEFAULT_CHECKPOINT = "/work/artifacts/iddsi_v07_best.pt"

_TRAINER_MODEL_SOURCE = "trainer/train/model.py"


# --------------------------------------------------------------------------
# CORAL ordinal head -- verbatim port of trainer/train/model.py
# --------------------------------------------------------------------------


def coral_logits_to_probs(logits: Any) -> Any:
    """Convert monotone P(y > k) logits into probabilities for L3..L7.

    Verbatim port of ``coral_logits_to_probs`` in ``trainer/train/model.py:13``.
    ``space/tests/test_space_core.py::test_coral_matches_trainer`` asserts this
    port is numerically identical to the trainer implementation, so the Space
    cannot drift from the model it claims to serve.
    """
    import torch

    survival = torch.sigmoid(logits)
    probs = torch.cat(
        [1.0 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]],
        dim=1,
    )
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def coral_decode(probabilities: Any) -> tuple[Any, Any]:
    """Decode CORAL probabilities by the ordinal RANK rule.

    Verbatim port of ``coral_decode`` in ``trainer/train/model.py:23``.

    The level index is ``#{k : P(y > k) > 0.5}`` (Cao, Mirjalili & Raschka),
    NOT ``argmax`` over the reconstructed categorical distribution.

    Why this matters: :func:`coral_logits_to_probs` builds interior classes as
    *differences of adjacent sigmoids*, so P(interior) is bounded by roughly
    ``tanh(gap / 4)`` while the two tail classes keep their full mass. With the
    default cut-point spacing (~0.69) every interior class is capped near 0.17
    while both tails reach ~0.26, so ``argmax`` provably can never return an
    interior level for ANY input. The first trained run emitted only L3, L7 and
    abstain because of this, on balanced L4-L7 data with no L3 at all.

    Returns ``(level_index, confidence)`` where confidence is the probability
    mass on the decoded level.
    """
    import numpy as np

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a [N, C] matrix")
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    # P(y > k) for k = 0 .. C-2, recovered from the categorical form.
    survival = 1.0 - np.cumsum(probs, axis=1)[:, :-1]
    predictions = (survival > 0.5).sum(axis=1).astype(np.int64)
    confidence = probs[np.arange(len(predictions)), predictions]
    return predictions, confidence


# --------------------------------------------------------------------------
# Output record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceOutput:
    """One Space inference result.

    ``status`` is one of:
      - ``"placeholder_no_weights"`` -- no trained checkpoint reachable;
        probabilities are uniform and ``decoded_level`` is ``None``.
      - ``"ok"`` -- a checkpoint was loaded and produced the distribution.
      - ``"unavailable"`` -- torch/transformers/PIL could not be imported.
      - ``"error"`` -- inference raised; the result is fail-closed.

    ``abstained`` is True whenever the demo declines to name a level.  It is
    True on every path except a confident ``"ok"``.
    """

    status: str
    levels: tuple[int, ...]
    probabilities: dict[int, float]
    decoded_level: int | None
    confidence: float | None
    abstained: bool
    abstention_text: str
    banner: str
    disclaimer: str
    disclaimer_zh_hant: str
    is_placeholder: bool
    detail: str
    checkpoint: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict_line(self) -> str:
        """The single line a user reads first.  Never a level when abstaining."""

        if self.abstained:
            return f"**{self.abstention_text}**"
        assert self.decoded_level is not None
        return (
            f"**Closest visual descriptor: L{self.decoded_level}** "
            f"(confidence {self.confidence:.3f}) — visual similarity only, "
            "not a texture test and not a verdict."
        )

    @property
    def verdict_plain(self) -> str:
        """``verdict_line`` without markdown emphasis, for plain-text widgets."""

        return self.verdict_line.replace("**", "")

    def to_markdown(self) -> str:
        """Render the result as markdown, banner first, disclaimer last."""

        lines: list[str] = []
        if self.banner:
            lines += [f"> ⚠️ {self.banner}", ""]
        lines += [self.verdict_line, ""]
        lines += ["| IDDSI level | probability |", "|---|---:|"]
        for level in self.levels:
            lines.append(f"| L{level} | {self.probabilities.get(level, float('nan')):.4f} |")
        lines += ["", f"*{self.detail}*", "", "---", f"**Disclaimer.** {self.disclaimer}"]
        lines += ["", f"**免責聲明。** {self.disclaimer_zh_hant}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Placeholder path
# --------------------------------------------------------------------------


def _placeholder_output(detail: str, *, checkpoint: str | None = None) -> SpaceOutput:
    """Uniform L3-L7 distribution with no decoded level.

    A uniform distribution is the only honest placeholder: any non-uniform
    number would have to come from untrained weights, and an untrained CORAL
    head produces confidently-wrong levels -- exactly the silent-wrong-level
    failure this Space must not ship.  Note also that decoding a uniform
    distribution with the rank rule yields L5, which is why the placeholder
    path never calls the decoder at all.
    """

    uniform = 1.0 / len(LEVELS)
    return SpaceOutput(
        status="placeholder_no_weights",
        levels=LEVELS,
        probabilities={int(level): uniform for level in LEVELS},
        decoded_level=None,
        confidence=None,
        abstained=True,
        abstention_text=ABSTENTION_TEXT,
        banner=PLACEHOLDER_BANNER,
        disclaimer=DISCLAIMER,
        disclaimer_zh_hant=DISCLAIMER_ZH_HANT,
        is_placeholder=True,
        detail=detail,
        checkpoint=checkpoint,
    )


def _closed_output(detail: str, *, status: str = "error") -> SpaceOutput:
    """Fail closed: abstain, emit no level, emit no probabilities."""

    return SpaceOutput(
        status=status,
        levels=LEVELS,
        probabilities={},
        decoded_level=None,
        confidence=None,
        abstained=True,
        abstention_text=ABSTENTION_TEXT,
        banner="Inference failed; no level is reported. " + PLACEHOLDER_BANNER,
        disclaimer=DISCLAIMER,
        disclaimer_zh_hant=DISCLAIMER_ZH_HANT,
        is_placeholder=True,
        detail=detail,
    )


# --------------------------------------------------------------------------
# Weighted path
# --------------------------------------------------------------------------


def resolve_checkpoint(checkpoint: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the checkpoint path to use, or ``None`` when no weights exist."""

    candidate = (
        Path(checkpoint).expanduser()
        if checkpoint
        else Path(os.getenv("IDDSI_SPACE_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser()
    )
    return candidate if candidate.is_file() else None


def _load_backbone_and_head(torch: Any, checkpoint: Path) -> tuple[Any, Any, Any]:
    """Build ``OrdinalImageModel`` from a checkpoint payload.

    Imports the trainer's own model class rather than re-implementing it, so
    the Space cannot diverge from the architecture it claims to serve.
    """

    import sys

    trainer_dir = Path(__file__).resolve().parent.parent / "trainer"
    if trainer_dir.is_dir() and str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    from train.model import OrdinalImageModel  # noqa: PLC0415

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model = OrdinalImageModel(
        kind=metadata.get("model_kind", "siglip2"),
        name=metadata.get("model_name", os.getenv("IDDSI_SPACE_BACKBONE", DEFAULT_BACKBONE)),
        pretrained=False,
    )
    model.load_state_dict(state)
    model.eval()
    return model, metadata, payload


def _open_rgb(image: Any) -> Any:
    """Coerce a Gradio upload / path / PIL image into an RGB PIL image."""

    from PIL import Image as PILImage  # noqa: PLC0415

    if hasattr(image, "convert"):
        return image.convert("RGB")
    return PILImage.open(image).convert("RGB")


def _preprocess(torch: Any, image: Any, model_name: str, model_kind: str = "siglip2") -> dict[str, Any]:
    """Turn an uploaded photo into the model inputs the trainer feeds it.

    Returns a dict restricted to :data:`_MODEL_INPUT_KEYS`, mirroring
    ``trainer/train/data.py::build_transform`` (which branches on ``kind`` and
    returns whatever the processor supplies) and
    ``trainer/train/engine.py::_forward`` (which passes those keys straight
    through to ``OrdinalImageModel``).

    The SigLIP-2 tower is NaViT-style: its processor returns *flattened patches*
    -- ``pixel_values`` of shape ``[n_patches, 3 * patch_size**2]`` -- plus a
    ``spatial_shapes`` tensor the tower uses to rebuild positional embeddings.
    Feeding pixels alone raises ``AttributeError: 'NoneType' object has no
    attribute 'shape'`` inside ``resize_positional_embeddings``, so the whole
    processor output is forwarded rather than just ``pixel_values``.
    """

    pil = _open_rgb(image)

    if model_kind != "siglip2":
        # Non-transformer backbones take a plain CHW tensor; the trainer uses
        # BasicImageTransform with per-architecture statistics.
        return {"pixel_values": _basic_transform(torch, pil, model_kind)}

    try:
        from transformers import AutoImageProcessor  # noqa: PLC0415

        processor = AutoImageProcessor.from_pretrained(model_name)
        encoded = processor(images=pil, return_tensors="pt")
        inputs = {
            key: value
            for key, value in encoded.items()
            if key in _MODEL_INPUT_KEYS
        }
        if "pixel_values" in inputs and "spatial_shapes" not in inputs:
            raise ValueError(
                "image processor returned pixel_values without spatial_shapes; "
                "the SigLIP-2 tower cannot run on this input"
            )
        if inputs:
            return inputs
    except ImportError:
        raise  # a missing transformers is an environment problem, not a data one
    except Exception:
        # Fall through to the manual flatten below.
        pass

    return _preprocess_flattened_patches(torch, pil)


#: Per-architecture image statistics, mirroring ``BasicImageTransform`` in
#: ``trainer/train/data.py`` (ImageNet stats for ResNet-18, 0.5/0.5 otherwise).
_BASIC_TRANSFORM_STATS = {
    "resnet18": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "default": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def _basic_transform(torch: Any, pil: Any, model_kind: str) -> Any:
    """The trainer's ``BasicImageTransform`` for non-transformer backbones."""

    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    mean, std = _BASIC_TRANSFORM_STATS.get(model_kind, _BASIC_TRANSFORM_STATS["default"])
    image = pil.resize((_FALLBACK_IMAGE_SIZE, _FALLBACK_IMAGE_SIZE), PILImage.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std_tensor = torch.tensor(std, dtype=torch.float32)[:, None, None]
    return ((tensor - mean_tensor) / std_tensor).unsqueeze(0)


def _preprocess_flattened_patches(torch: Any, pil: Any) -> dict[str, Any]:
    """Manual NaViT flatten, used when the image processor cannot be loaded.

    Produces exactly what ``Siglip2ImageProcessor`` produces for a square image:
    resized to ``image_size``, rescaled to [0, 1], normalised, then split into
    ``patch_size`` squares flattened row-major into ``[n_patches,
    3 * patch_size**2]``, with ``spatial_shapes = [[grid, grid]]``.
    """

    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    image_size = _FALLBACK_IMAGE_SIZE
    patch_size = _FALLBACK_PATCH_SIZE
    mean = _FALLBACK_IMAGE_MEAN
    std = _FALLBACK_IMAGE_STD
    resample = PILImage.Resampling.BILINEAR
    backbone = Path(os.getenv("IDDSI_SPACE_BACKBONE", DEFAULT_BACKBONE))
    config_path = backbone / "preprocessor_config.json"
    if config_path.is_file():
        try:
            settings = json.loads(config_path.read_text(encoding="utf-8"))
            patch_size = int(settings.get("patch_size", patch_size))
            mean = tuple(settings.get("image_mean", mean))
            std = tuple(settings.get("image_std", std))
            # PIL resample enum: 0 NEAREST, 1 LANCZOS, 2 BILINEAR, 3 BICUBIC.
            resample = int(settings.get("resample", resample))
        except Exception:
            pass  # keep the documented defaults

    array = np.asarray(
        pil.resize((image_size, image_size), resample), dtype=np.float32
    ) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    array = (array - mean_array) / std_array

    grid = image_size // patch_size
    patches = (
        array.reshape(grid, patch_size, grid, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * grid, patch_size * patch_size * 3)
    )
    return {
        "pixel_values": torch.from_numpy(patches).unsqueeze(0),
        "spatial_shapes": torch.tensor([[grid, grid]], dtype=torch.long),
    }


def predict(
    image: Any,
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    abstain_threshold: float | None = None,
) -> SpaceOutput:
    """Run one Space inference.  Never raises; never emits a level when unsure.

    ``abstain_threshold`` defaults to :data:`DEFAULT_ABSTENTION_THRESHOLD`.
    """

    threshold = (
        DEFAULT_ABSTENTION_THRESHOLD
        if abstain_threshold is None
        else float(abstain_threshold)
    )

    resolved = resolve_checkpoint(checkpoint)
    if resolved is None:
        return _placeholder_output(
            "No trained checkpoint found at "
            f"{checkpoint or os.getenv('IDDSI_SPACE_CHECKPOINT', DEFAULT_CHECKPOINT)}; "
            "returning a uniform placeholder distribution (not a model output).",
            checkpoint=str(checkpoint or DEFAULT_CHECKPOINT),
        )

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on the Space image.
        return _closed_output(
            f"Model dependencies unavailable ({type(exc).__name__}: {exc}).",
            status="unavailable",
        )

    try:
        model, metadata, _payload = _load_backbone_and_head(torch, resolved)
        model_name = str(metadata.get("model_name", DEFAULT_BACKBONE))
        model_kind = str(metadata.get("model_kind", "siglip2"))
        inputs = _preprocess(torch, image, model_name, model_kind)
        with torch.no_grad():
            logits = model(**inputs)
            probs = coral_logits_to_probs(logits).cpu().numpy()
        if probs.shape != (1, len(LEVELS)):
            return _closed_output(
                f"Head returned shape {probs.shape}, expected (1, {len(LEVELS)})."
            )
        indices, confidences = coral_decode(probs)
        index = int(indices[0])
        if not 0 <= index < len(LEVELS):
            return _closed_output(f"Decoded level index {index} outside 0..{len(LEVELS) - 1}.")
        prob_map = {
            int(level): float(probs[0, position])
            for position, level in enumerate(LEVELS)
        }
        total = sum(prob_map.values())
        if not abs(total - 1.0) < 1e-3:
            return _closed_output(f"Probabilities sum to {total:.6f}, expected 1.0.")
        confidence = float(confidences[0])
        level = int(LEVELS[index])
        abstained = confidence < threshold
        return SpaceOutput(
            status="ok",
            levels=LEVELS,
            probabilities=prob_map,
            decoded_level=None if abstained else level,
            confidence=confidence,
            abstained=abstained,
            abstention_text=ABSTENTION_TEXT,
            banner="" if not abstained else (
                f"Confidence {confidence:.3f} is below the abstention threshold "
                f"{threshold:.3f} (v0.7 validation calibration, "
                "results/v07_release/v07_seed101_calibration.json), so no level "
                "is reported for this photo."
            ),
            disclaimer=DISCLAIMER,
            disclaimer_zh_hant=DISCLAIMER_ZH_HANT,
            is_placeholder=False,
            detail=(
                "Ordinal CORAL head output decoded with the rank rule "
                "level = #{k : P(y>k) > 0.5}; visual-similarity research signal only."
            ),
            checkpoint=str(resolved),
            extras={"temperature": metadata.get("temperature")},
        )
    except Exception as exc:
        return _closed_output(f"{type(exc).__name__}: {exc}")


def describe_configuration() -> Mapping[str, Any]:
    """Report what the Space would do at import time (used by tests/README)."""

    return {
        "levels": list(LEVELS),
        "checkpoint_env": "IDDSI_SPACE_CHECKPOINT",
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "checkpoint_resolves": resolve_checkpoint() is not None,
        "backbone_env": "IDDSI_SPACE_BACKBONE",
        "default_backbone": DEFAULT_BACKBONE,
        "abstain_threshold": DEFAULT_ABSTENTION_THRESHOLD,
        "decode_rule": "level = #{k : P(y>k) > 0.5}",
        "coral_source": _TRAINER_MODEL_SOURCE,
    }


def available_levels() -> Sequence[int]:
    """The L3-L7 labels, in soft -> thick order."""

    return LEVELS
