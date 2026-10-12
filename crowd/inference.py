"""Stub inference wiring for the CROWD-CAPTURE research preview.

This module connects the Gradio UI to the trainer package's model interface
(``train.model.OrdinalImageModel`` / ``coral_logits_to_probs`` / ``LEVELS``).

It is a *stub*: the model is loaded lazily on first use, and when no trained
checkpoint is present it returns a placeholder result.  No accuracy,
safety, or IDDSI-compliance claim is made or implied by this output — the
result is a visual-similarity research signal only (see ``app.DISCLAIMER``).

The model source can be a local checkpoint path or a ``hub:`` reference to the
released Hugging Face repo.  Resolving a ``hub:`` reference goes through
``trainer/eval/hub_source.py``, so the capture pipeline reads the *published*
bytes (architecture config, calibrated threshold) and records which endpoint
and commit served them.  The release publishes no weights — that is a locked
decision (wk1 kill rule, dangerous under-classification 33%), and this module
verifies it from the Hub file list instead of assuming it.  With no weights to
load the estimate stays the honest placeholder: ``predicted_level=None``,
never a fabricated level.
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

# Default model source. A ``hub:`` reference pulls the released model-source
# artifacts from the Hub (see ``hub/HOW_TO_CONSUME.md``); a plain path is a
# local checkpoint. Override with CROWD_MODEL_SOURCE.
DEFAULT_MODEL_SOURCE = (
    "hub:results/fix1_siglip2_coral_seed17/resolved_config.yaml"
)
MODEL_SOURCE_ENV = "CROWD_MODEL_SOURCE"
CHECKPOINT_ENV = "CROWD_MODEL_CHECKPOINT"

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


@dataclass(frozen=True)
class ModelSource:
    """Where the capture pipeline's model came from, and what the Hub serves.

    ``weights_published`` is measured from the Hub file list, not assumed: the
    release withholds the weights under the wk1 kill rule, and a consumer that
    reports a placeholder must be able to say *why* it is a placeholder. When
    the Hub cannot be reached the field is ``None`` ("unknown"), never
    ``False`` — an unreachable endpoint is not evidence.
    """

    spec: str
    kind: str  # "local" | "hub"
    local_path: str | None
    repo: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    endpoint_provenance: str | None = None
    resolved_commit: str | None = None
    weights_published: bool | None = None
    published_weight_files: tuple[str, ...] = ()
    architecture: dict[str, Any] = field(default_factory=dict)
    calibrated_threshold: float | None = None
    detail: str = ""

    @property
    def has_weights(self) -> bool:
        """True only when a local checkpoint file is actually present."""
        return self.local_path is not None and Path(self.local_path).is_file()

    def provenance_line(self) -> str:
        if self.kind != "hub":
            return f"local model source: {self.spec}"
        return (
            f"hub model source: {self.repo}@{self.revision} "
            f"(commit {self.resolved_commit or 'unresolved'}) via {self.endpoint} "
            f"— {self.endpoint_provenance}"
        )


def _hub_source_module() -> Any:
    """Import ``eval.hub_source``, putting the sibling trainer/ on sys.path."""
    import sys

    if str(_TRAINER_DIR) not in sys.path:
        sys.path.insert(0, str(_TRAINER_DIR))
    from eval import hub_source  # noqa: PLC0415

    return hub_source


def default_model_source() -> str:
    """The model source to use when the caller does not name one.

    ``CROWD_MODEL_SOURCE`` may hold a ``hub:`` reference (the documented
    consumption path) or a local checkpoint path. Unset keeps the historical
    offline behaviour, so a deployment that never opted into the Hub is not
    silently made network-dependent.
    """
    return os.getenv(MODEL_SOURCE_ENV, "").strip() or os.getenv(
        CHECKPOINT_ENV, str(_DEFAULT_CHECKPOINT)
    )


def resolve_model_source(
    spec: str | os.PathLike[str] | None = None,
    *,
    endpoint: str | None = None,
    cache_dir: str | None = None,
    require_network: bool = False,
) -> ModelSource:
    """Resolve a model source, pulling the released artifacts when it is a ``hub:`` ref.

    A ``hub:`` reference is fetched through ``eval.hub_source``: the named
    artifact (normally the released ``resolved_config.yaml``) plus the Hub file
    list, so the caller learns the architecture, the calibrated threshold and
    whether any weights are published. Raises when the Hub cannot be reached —
    there is no silent local substitution.
    """
    spec = str(spec) if spec is not None else default_model_source()
    hub = _hub_source_module()
    if not hub.is_ref(spec):
        return ModelSource(
            spec=spec,
            kind="local",
            local_path=str(Path(spec).expanduser()),
            detail="Local model source; no Hub fetch performed.",
        )

    ref = hub.parse_ref(spec)
    path, receipt = hub.resolve_local(
        spec, kind="model source", repo=ref.repo, revision=ref.revision,
        endpoint=endpoint, cache_dir=cache_dir, require_network=require_network,
    )
    architecture: dict[str, Any] = {}
    threshold: float | None = None
    try:
        import yaml  # noqa: PLC0415

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model_section = payload.get("model") if isinstance(payload, dict) else None
        if isinstance(model_section, dict):
            architecture = {str(k): v for k, v in model_section.items()}
        calibration = payload.get("calibration") if isinstance(payload, dict) else None
        if isinstance(calibration, dict) and "threshold" in calibration:
            threshold = float(calibration["threshold"])
    except Exception:  # noqa: BLE001 - a config we cannot parse is not fatal here
        architecture = {}

    weights = hub.published_weight_files(
        ref.repo, revision=ref.revision, endpoint=endpoint
    )
    if weights:
        detail = (
            f"The release publishes {len(weights)} weight file(s); this stub does "
            "not auto-load them. Set CROWD_MODEL_CHECKPOINT to a local checkpoint "
            "to run the real head."
        )
    else:
        detail = (
            "No weights are published for this release (wk1 kill rule; dangerous "
            "under-classification 33%). The estimate below is the uniform "
            "placeholder, not a model output."
        )
    return ModelSource(
        spec=spec,
        kind="hub",
        local_path=str(path),
        repo=ref.repo,
        revision=ref.revision,
        endpoint=receipt.endpoint if receipt else None,
        endpoint_provenance=receipt.endpoint_provenance if receipt else None,
        resolved_commit=receipt.resolved_commit if receipt else None,
        weights_published=bool(weights),
        published_weight_files=tuple(weights),
        architecture=architecture,
        calibrated_threshold=threshold,
        detail=detail,
    )


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
    source: str | os.PathLike[str] | None = None,
    endpoint: str | None = None,
    cache_dir: str | None = None,
    require_network: bool = False,
) -> InferenceResult:
    """Run stub ordinal inference on one image.

    Stub contract: if no trained checkpoint is available, return a uniform
    placeholder distribution with ``is_placeholder=True`` and
    ``predicted_level=None``.  This function makes no accuracy claim; callers
    must surface ``app.DISCLAIMER`` alongside any rendered output.

    ``source=`` (or ``CROWD_MODEL_SOURCE``) may name a ``hub:`` reference, in
    which case the released artifacts are pulled from the Hub first and the
    result carries the serving endpoint and commit in ``extras``.  The release
    publishes no weights, so a ``hub:`` source yields the placeholder with that
    provenance attached — never a fabricated level.
    """
    if checkpoint is None and source is None:
        source = default_model_source()
    if source is not None:
        return _predict_with_source(
            image, str(source), endpoint=endpoint, cache_dir=cache_dir,
            require_network=require_network,
        )
    return _predict_with_checkpoint(image, checkpoint)


def _predict_with_source(
    image: Any,
    spec: str,
    *,
    endpoint: str | None,
    cache_dir: str | None,
    require_network: bool = False,
) -> InferenceResult:
    """Resolve a model source (local path or ``hub:`` ref), then predict."""
    resolved = resolve_model_source(
        spec, endpoint=endpoint, cache_dir=cache_dir, require_network=require_network
    )
    if resolved.kind == "local":
        return _predict_with_checkpoint(image, resolved.local_path)
    if resolved.has_weights and str(resolved.local_path).lower().endswith(
        (".pt", ".pth", ".safetensors", ".bin", ".ckpt")
    ):
        # A future release that publishes weights: the fetched file *is* the
        # checkpoint, so load it rather than reporting a placeholder.
        return _predict_with_checkpoint(image, resolved.local_path)
    return _hub_placeholder(image, resolved)


def _hub_placeholder(image: Any, resolved: ModelSource) -> InferenceResult:
    """The honest placeholder for a Hub model source that ships no weights."""
    if resolved.weights_published is None:
        status = "unavailable"
        detail = (
            f"Could not determine whether the release publishes weights "
            f"({resolved.detail}); refusing to report a level."
        )
    elif resolved.weights_published:
        status = "placeholder_no_weights"
        detail = (
            f"{resolved.detail} Published weight file(s): "
            f"{', '.join(resolved.published_weight_files)}."
        )
    else:
        status = "placeholder_no_weights"
        detail = resolved.detail
    result = _placeholder_result(detail, resolved.spec)
    return InferenceResult(
        status=status,
        levels=result.levels,
        probabilities=result.probabilities,
        predicted_level=None,
        model_kind=(resolved.architecture or {}).get("kind"),
        checkpoint=resolved.local_path,
        is_placeholder=True,
        detail=f"{detail} Provenance — {resolved.provenance_line()}.",
        extras={
            "model_source": resolved.spec,
            "model_source_kind": resolved.kind,
            "hub_repo": resolved.repo,
            "hub_revision": resolved.revision,
            "hub_endpoint": resolved.endpoint,
            "hub_endpoint_provenance": resolved.endpoint_provenance,
            "hub_resolved_commit": resolved.resolved_commit,
            "weights_published": resolved.weights_published,
            "published_weight_files": list(resolved.published_weight_files),
            "architecture": resolved.architecture,
            "calibrated_threshold": resolved.calibrated_threshold,
        },
    )


def _predict_with_checkpoint(
    image: Any,
    checkpoint: str | os.PathLike[str] | None,
) -> InferenceResult:
    """Load a local checkpoint (if present) and predict; placeholder otherwise."""

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
