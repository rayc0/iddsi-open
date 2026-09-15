"""Real-data loader for IDDSI-Open dataschema events.

Reads a dataset directory laid out per ``dataschema/schema/event.schema.json``::

    dataset_dir/
      events.jsonl          one event per line, schema_version 1.0.0
      media/...             files referenced by event["media_files"][*]["path"]

Source tags
-----------
The base schema declares ``source`` as an enum of ``{"real", "synthetic"}``
(see ``dataschema/schema/event.schema.json``). The project data strategy
(``dataschema/docs/DATASHEET.md``) refines ``real`` into three operational
tags, and the real pilot capture on Spark
(``~/iddsi/data/realweak_pilot/events.jsonl``) already emits
``source="real-weak"`` with ``level_weak`` / ``weak_label_record`` extension
fields. This loader therefore accepts:

- ``synthetic``    pipeline-development material, label from ``level_declared``
- ``real-weak``    third-party images, visual weak label from ``level_weak``
- ``real-crowd``   voluntary smartphone uploads, weak label from ``level_weak``
- ``real-tested``  full capture protocol with physical tests, label from
                   ``level_adjudicated`` (fallback ``level_tested_RD`` then
                   ``level_tested_SLP``)
- ``real``         base-schema value; treated as ``real-tested`` because the
                   base schema only exists for physically tested captures

Grouping
--------
The schema provides ``recipe_id``, ``batch_id`` and ``kitchen_id`` (all
required). The group key used for split enforcement is the triple
``(recipe_id, batch_id, kitchen_id)``: a batch of one recipe cooked in one
kitchen must never straddle train/val. ``group_split`` assigns whole groups to
splits; ``validate_no_group_leak`` additionally verifies that any pre-assigned
``split`` fields in the events file respect the same constraint.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Source tags
# ---------------------------------------------------------------------------

SOURCE_SYNTHETIC = "synthetic"
SOURCE_REAL_WEAK = "real-weak"
SOURCE_REAL_CROWD = "real-crowd"
SOURCE_REAL_TESTED = "real-tested"

#: Canonical operational source tags.
SOURCE_TAGS: tuple[str, ...] = (
    SOURCE_SYNTHETIC,
    SOURCE_REAL_WEAK,
    SOURCE_REAL_CROWD,
    SOURCE_REAL_TESTED,
)

#: Base-schema ``source="real"`` only exists for physically tested captures.
_SOURCE_ALIASES: dict[str, str] = {"real": SOURCE_REAL_TESTED}

#: Splits recognised by the dataschema (``external_test`` maps to ``test``).
SCHEMA_SPLITS: tuple[str, ...] = ("train", "val", "external_test")
_SPLIT_ALIASES: dict[str, str] = {"external_test": "test"}

#: Label fields tried in order per canonical source tag.
_LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    SOURCE_SYNTHETIC: ("level_declared",),
    SOURCE_REAL_WEAK: ("level_weak",),
    SOURCE_REAL_CROWD: ("level_weak",),
    SOURCE_REAL_TESTED: ("level_adjudicated", "level_tested_RD", "level_tested_SLP"),
}

DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    SOURCE_SYNTHETIC: 0.25,
    SOURCE_REAL_WEAK: 0.5,
    SOURCE_REAL_CROWD: 1.0,
    SOURCE_REAL_TESTED: 2.0,
}


def canonical_source(raw: Any) -> str:
    tag = _SOURCE_ALIASES.get(str(raw), str(raw))
    if tag not in SOURCE_TAGS:
        raise ValueError(f"unknown event source {raw!r}; expected one of {SOURCE_TAGS} or 'real'")
    return tag


def canonical_split(raw: Any) -> str:
    split = _SPLIT_ALIASES.get(str(raw), str(raw))
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown event split {raw!r}")
    return split


# ---------------------------------------------------------------------------
# Event records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecord:
    """One loadable image sample derived from a dataschema event."""

    event_id: str
    source: str  # canonical source tag
    split: str  # train | val | test (external_test folded into test)
    label: int | None  # resolved IDDSI level, None when unlabelled
    group_key: tuple[str, str, str]  # (recipe_id, batch_id, kitchen_id)
    image_path: str  # absolute path to the image file
    capture_role: str


def _resolve_label(event: dict[str, Any], source: str) -> int | None:
    for field_name in _LABEL_FIELDS[source]:
        value = event.get(field_name)
        if value is not None:
            level = int(value)
            if not 0 <= level <= 7:
                raise ValueError(f"event {event.get('event_id')!r} has out-of-range level {level}")
            return level
    return None


def _select_image(
    event: dict[str, Any], capture_role: str
) -> dict[str, Any] | None:
    media = event.get("media_files") or []
    images = [m for m in media if m.get("media_type") == "image"]
    for item in images:
        if item.get("capture_role") == capture_role:
            return item
    return images[0] if images else None


def load_events(
    dataset_dir: str | Path,
    *,
    capture_role: str = "plate_photo",
    keep_unlabeled: bool = False,
) -> list[EventRecord]:
    """Load ``events.jsonl`` from ``dataset_dir`` into flat image records.

    Events without an image media file are skipped. Events whose label cannot
    be resolved for their source tag are dropped unless ``keep_unlabeled`` is
    true (then ``label is None``; they are excluded from supervised sampling
    weights by default).
    """
    root = Path(dataset_dir).resolve()
    events_path = root / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"events.jsonl not found under {root}")
    records: list[EventRecord] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            event_id = str(event.get("event_id", f"line_{line_no}"))
            source = canonical_source(event.get("source"))
            split = canonical_split(event.get("split"))
            media = _select_image(event, capture_role)
            if media is None:
                continue
            image_path = Path(media["path"])
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            label = _resolve_label(event, source)
            if label is None and not keep_unlabeled:
                continue
            records.append(
                EventRecord(
                    event_id=event_id,
                    source=source,
                    split=split,
                    label=label,
                    group_key=(
                        str(event["recipe_id"]),
                        str(event["batch_id"]),
                        str(event["kitchen_id"]),
                    ),
                    image_path=str(image_path),
                    capture_role=str(media.get("capture_role", capture_role)),
                )
            )
    if not records:
        raise ValueError(f"no loadable image events found in {events_path}")
    return records


# ---------------------------------------------------------------------------
# Group-aware splitting
# ---------------------------------------------------------------------------


def group_split(
    records: Sequence[EventRecord],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Assign whole (recipe, batch, kitchen) groups to train or val.

    Returns ``(train_indices, val_indices)``. Deterministic given ``seed``.
    Raises if a group would be forced into both splits (impossible here since
    assignment is per-group) or if either side ends up empty.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.group_key, []).append(index)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, round(len(keys) * val_fraction)) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])
    train_idx = [i for key in keys if key not in val_keys for i in groups[key]]
    val_idx = [i for key in keys if key in val_keys for i in groups[key]]
    if not train_idx or not val_idx:
        raise ValueError(
            f"group split degenerate: {len(keys)} group(s) cannot fill both train and val"
        )
    return sorted(train_idx), sorted(val_idx)


def validate_no_group_leak(records: Sequence[EventRecord]) -> None:
    """Verify pre-assigned ``split`` fields keep every group in one split."""
    by_group: dict[tuple[str, str, str], set[str]] = {}
    for record in records:
        by_group.setdefault(record.group_key, set()).add(record.split)
    leaked = {g: s for g, s in by_group.items() if len(s) > 1}
    if leaked:
        example = sorted(leaked.items())[:3]
        raise ValueError(f"group leakage across splits: {example}")


# ---------------------------------------------------------------------------
# Per-source sampling weights
# ---------------------------------------------------------------------------


def sample_weights(
    records: Sequence[EventRecord],
    source_weights: dict[str, float] | None = None,
    *,
    unlabeled_weight: float = 0.0,
) -> list[float]:
    """Per-sample weights from a ``source -> weight`` mapping.

    Unknown sources raise; sources absent from the mapping default to 1.0.
    Unlabelled records get ``unlabeled_weight`` (0 excludes them from a
    supervised ``WeightedRandomSampler``).
    """
    weights = dict(DEFAULT_SOURCE_WEIGHTS if source_weights is None else source_weights)
    for key in weights:
        canonical_source(key)  # validate mapping keys up front
    result: list[float] = []
    for record in records:
        if record.label is None:
            result.append(float(unlabeled_weight))
        else:
            result.append(float(weights.get(record.source, 1.0)))
    return result


def build_weighted_sampler(
    records: Sequence[EventRecord],
    source_weights: dict[str, float] | None = None,
    *,
    seed: int = 0,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    weights = sample_weights(records, source_weights)
    if not any(w > 0 for w in weights):
        raise ValueError("all sampling weights are zero; nothing to draw")
    return WeightedRandomSampler(
        weights,
        num_samples=num_samples or len(records),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# Augmentations (PIL/numpy only; torchvision is not a trainer dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation hyperparameters. All ranges are sampled uniformly."""

    image_size: int = 224
    horizontal_flip_p: float = 0.5
    brightness: float = 0.2  # factor in [1-b, 1+b]
    contrast: float = 0.2
    saturation: float = 0.2
    blur_p: float = 0.1  # probability of Gaussian blur
    blur_radius: float = 1.0  # max radius
    crop_scale: tuple[float, float] = (0.8, 1.0)  # area fraction range

    def validate(self) -> None:
        if self.image_size < 8:
            raise ValueError("image_size too small")
        for name in ("brightness", "contrast", "saturation"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0.0 <= self.horizontal_flip_p <= 1.0 or not 0.0 <= self.blur_p <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        lo, hi = self.crop_scale
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError("crop_scale must satisfy 0 < lo <= hi <= 1")


class SeededAugment:
    """Deterministic augmentation pipeline.

    ``__call__(image, seed)`` draws every random decision from
    ``random.Random(seed)`` / ``numpy.random.default_rng(seed)``, so the same
    (image, seed) pair always yields the same tensor. Training code passes a
    per-(epoch, index) seed; tests pass a fixed seed to assert determinism.
    """

    def __init__(
        self,
        config: AugmentConfig | None = None,
        *,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        enabled: bool = True,
    ):
        self.config = config or AugmentConfig()
        self.config.validate()
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.enabled = enabled

    def _augment(self, image: Image.Image, rng: random.Random) -> Image.Image:
        cfg = self.config
        w, h = image.size
        scale = rng.uniform(*cfg.crop_scale)
        area = w * h * scale
        aspect = rng.uniform(3.0 / 4.0, 4.0 / 3.0)
        crop_w = min(w, max(1, int(round((area * aspect) ** 0.5))))
        crop_h = min(h, max(1, int(round((area / aspect) ** 0.5))))
        left = rng.randint(0, w - crop_w)
        top = rng.randint(0, h - crop_h)
        image = image.crop((left, top, left + crop_w, top + crop_h))
        if rng.random() < cfg.horizontal_flip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if cfg.brightness:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - cfg.brightness, 1 + cfg.brightness))
        if cfg.contrast:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(1 - cfg.contrast, 1 + cfg.contrast))
        if cfg.saturation:
            image = ImageEnhance.Color(image).enhance(rng.uniform(1 - cfg.saturation, 1 + cfg.saturation))
        if rng.random() < cfg.blur_p:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, cfg.blur_radius)))
        return image

    def __call__(self, image: Image.Image, seed: int = 0) -> torch.Tensor:
        rng = random.Random(seed)
        image = image.convert("RGB")
        if self.enabled:
            image = self._augment(image, rng)
        image = image.resize(
            (self.config.image_size, self.config.image_size), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EventsDataset(Dataset[dict[str, Any]]):
    """Torch dataset over dataschema events.

    Each item yields ``pixel_values`` (or the transform's dict output merged
    in), ``label`` (-1 when unlabelled), ``event_id``, ``source`` and
    ``group_id`` (``recipe/batch/kitchen`` joined by ``/``). Source tags are
    exposed per sample so training loops can log or re-weight by provenance.
    """

    def __init__(
        self,
        records: Sequence[EventRecord],
        transform: Callable[..., torch.Tensor | dict[str, torch.Tensor]] | None = None,
        *,
        seed: int = 0,
    ):
        if not records:
            raise ValueError("EventsDataset requires at least one record")
        self.records = list(records)
        self.transform = transform or SeededAugment(enabled=False)
        self.seed = seed

    @classmethod
    def from_dir(
        cls,
        dataset_dir: str | Path,
        *,
        split: str | None = None,
        capture_role: str = "plate_photo",
        keep_unlabeled: bool = False,
        transform: Callable[..., torch.Tensor | dict[str, torch.Tensor]] | None = None,
        seed: int = 0,
    ) -> "EventsDataset":
        records = load_events(dataset_dir, capture_role=capture_role, keep_unlabeled=keep_unlabeled)
        if split is not None:
            wanted = canonical_split(split)
            records = [r for r in records if r.split == wanted]
            if not records:
                raise ValueError(f"no records for split {wanted!r} in {dataset_dir}")
        return cls(records, transform, seed=seed)

    @property
    def sources(self) -> list[str]:
        return [r.source for r in self.records]

    @property
    def labels(self) -> list[int]:
        return [(-1 if r.label is None else r.label) for r in self.records]

    @property
    def group_ids(self) -> list[str]:
        return ["/".join(r.group_key) for r in self.records]

    def subset(self, indices: Sequence[int]) -> "EventsDataset":
        return EventsDataset([self.records[i] for i in indices], self.transform, seed=self.seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            try:
                transformed = self.transform(image, seed=self.seed + index)
            except TypeError:
                transformed = self.transform(image)
        model_inputs = transformed if isinstance(transformed, dict) else {"pixel_values": transformed}
        return {
            **model_inputs,
            "label": torch.tensor(-1 if record.label is None else record.label, dtype=torch.long),
            "event_id": record.event_id,
            "source": record.source,
            "group_id": "/".join(record.group_key),
        }


def iter_source_counts(records: Sequence[EventRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (for per-epoch reseeding)."""
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF
