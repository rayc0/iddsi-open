"""Stub inference wiring for the CROWD-CAPTURE research preview.

This module connects the Gradio UI to the trainer package's model interface
(``train.model.OrdinalImageModel`` / ``coral_logits_to_probs`` / ``LEVELS``).

It is a *stub*: the model is loaded lazily on first use, and when no trained
checkpoint is present it returns a placeholder result.  No accuracy,
safety, or IDDSI-compliance claim is made or implied by this output — the
result is a visual-similarity research signal only (see ``app.DISCLAIMER``).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The trainer package lives in the sibling ``trainer/`` directory and is
# importable as the top-level ``train`` package once that directory is on
# sys.path (see trainer/pyproject.toml: packages = ["train*", "eval*", ...]).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINER_DIR = _REPO_ROOT / "trainer"

# Default checkpoint location; override with CROWD_MODEL_CHECKPOINT.
_DEFAULT_CHECKPOINT = _REPO_ROOT / "trainer" / "artifacts" / "best.pt"

# IDDSI levels the ordinal head predicts, mirrored from train.model.LEVELS so
# the placeholder path works even when torch/transformers are unavailable.
_FALLBACK_LEVELS = (3, 4, 5, 6, 7)


@dataclass(frozen=True)
class InferenceResult:
    """Structured output of one stub inference call.

    ``status`` is one of:
      - ``"placeholder_no_weights"`` — no checkpoint found; probabilities are
        a uniform placeholder, NOT a model output.
      - ``"ok"`` — a checkpoint was loaded and produced the probabilities.
      - ``"unavailable"`` — model dependencies (torch etc.) are missing.
    """

    status: str
    levels: tuple[int, ...]
    probabilities: dict[int, float]
    predicted_level: int | None
    model_kind: str | None
    checkpoint: str | None
    is_placeholder: bool
    detail: str
    extras: dict[str, Any] = field(default_factory=dict)


class _LazyModel:
    """Load the trainer model on first use; never at import time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._model: Any = None
        self._levels: tuple[int, ...] = _FALLBACK_LEVELS
        self._torch: Any = None
        self._coral_logits_to_probs: Any = None
        self._load_error: str | None = None

    def _import_trainer_interface(self) -> bool:
        """Import the trainer model interface; return True on success."""
        if self._torch is not None:
            return True
        try:
            import sys

            if str(_TRAINER_DIR) not in sys.path:
                sys.path.insert(0, str(_TRAINER_DIR))
            import torch  # noqa: PLC0415
            from train import LEVELS, coral_logits_to_probs  # noqa: PLC0415
            from train.model import OrdinalImageModel  # noqa: PLC0415

            self._torch = torch
            self._levels = tuple(int(v) for v in LEVELS)
            self._coral_logits_to_probs = coral_logits_to_probs
            self._OrdinalImageModel = OrdinalImageModel  # type: ignore[attr-defined]
            return True
        except Exception as exc:  # pragma: no cover - depends on environment.
            self._load_error = f"{type(exc).__name__}: {exc}"
            return False

    def resolve_checkpoint(self, checkpoint: str | os.PathLike[str] | None) -> Path | None:
        """Return the checkpoint path to use, or None if no weights exist."""
        candidate = (
            Path(checkpoint).expanduser()
            if checkpoint
            else Path(
                os.getenv("CROWD_MODEL_CHECKPOINT", str(_DEFAULT_CHECKPOINT))
            ).expanduser()
        )
        return candidate if candidate.is_file() else None

    def get(self, checkpoint: str | os.PathLike[str] | None = None) -> tuple[Any, Path] | None:
        """Return (model, checkpoint_path), loading lazily; None if no weights."""
        with self._lock:
            if self._loaded:
                return (self._model, self._checkpoint_path) if self._model is not None else None  # type: ignore[attr-defined]
            self._loaded = True
            resolved = self.resolve_checkpoint(checkpoint)
            if resolved is None:
                return None
            if not self._import_trainer_interface():
                return None
            try:
                payload = self._torch.load(resolved, map_location="cpu", weights_only=True)
                metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                model = self._OrdinalImageModel(  # type: ignore[attr-defined]
                    kind=metadata.get("model_kind", "tiny_cnn"),
                    name=metadata.get("model_name", ""),
                    pretrained=False,
                )
                state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                model.load_state_dict(state)
                model.eval()
                self._model = model
                self._checkpoint_path = resolved  # type: ignore[attr-defined]
                return self._model, resolved
            except Exception as exc:  # pragma: no cover - corrupt/mismatched weights.
                self._load_error = f"{type(exc).__name__}: {exc}"
                return None


_LAZY = _LazyModel()


def _placeholder_result(detail: str, checkpoint: str | None) -> InferenceResult:
    uniform = 1.0 / len(_FALLBACK_LEVELS)
    return InferenceResult(
        status="placeholder_no_weights",
        levels=_FALLBACK_LEVELS,
        probabilities={level: uniform for level in _FALLBACK_LEVELS},
        predicted_level=None,
        model_kind=None,
        checkpoint=checkpoint,
        is_placeholder=True,
        detail=detail,
    )


def predict_iddsi_level(
    image: Any,
    *,
    checkpoint: str | os.PathLike[str] | None = None,
) -> InferenceResult:
    """Run stub ordinal inference on one image.

    Stub contract: if no trained checkpoint is available, return a uniform
    placeholder distribution with ``is_placeholder=True`` and
    ``predicted_level=None``.  This function makes no accuracy claim; callers
    must surface ``app.DISCLAIMER`` alongside any rendered output.
    """

    resolved = _LAZY.resolve_checkpoint(checkpoint)
    if resolved is None:
        return _placeholder_result(
            "No trained checkpoint found; returning a uniform placeholder "
            "distribution (not a model output).",
            str(checkpoint or _DEFAULT_CHECKPOINT),
        )

    loaded = _LAZY.get(checkpoint)
    if loaded is None:
        if _LAZY._load_error and _LAZY._torch is None:
            return InferenceResult(
                status="unavailable",
                levels=_FALLBACK_LEVELS,
                probabilities={},
                predicted_level=None,
                model_kind=None,
                checkpoint=str(resolved),
                is_placeholder=True,
                detail=f"Model dependencies unavailable: {_LAZY._load_error}",
            )
        return _placeholder_result(
            f"Checkpoint could not be loaded ({_LAZY._load_error or 'unknown error'}); "
            "returning a uniform placeholder distribution.",
            str(resolved),
        )

    model, checkpoint_path = loaded
    torch = _LAZY._torch

    tensor = _to_tensor(image, torch)
    with torch.no_grad():
        logits = model(pixel_values=tensor)
        probs = _LAZY._coral_logits_to_probs(logits)[0]
    levels = _LAZY._levels
    prob_map = {int(level): float(probs[index]) for index, level in enumerate(levels)}
    predicted = max(prob_map, key=prob_map.get)  # type: ignore[arg-type]
    return InferenceResult(
        status="ok",
        levels=levels,
        probabilities=prob_map,
        predicted_level=int(predicted),
        model_kind=getattr(model, "kind", None),
        checkpoint=str(checkpoint_path),
        is_placeholder=False,
        detail="Ordinal CORAL head output; visual-similarity research signal only.",
    )


def _to_tensor(image: Any, torch: Any) -> Any:
    """Normalize a file path / PIL image / tensor into a batched CHW tensor."""

    if torch.is_tensor(image):
        tensor = image.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to load image inputs.") from exc

    pil = Image.open(image).convert("RGB") if not hasattr(image, "convert") else image.convert("RGB")
    pil = pil.resize((224, 224))
    import numpy as np  # noqa: PLC0415

    array = np.asarray(pil, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def reset_lazy_model_for_tests() -> None:
    """Reset the lazy singleton (test isolation only)."""

    with _LAZY._lock:
        _LAZY._loaded = False
        _LAZY._model = None
        _LAZY._load_error = None
        _LAZY._torch = None
