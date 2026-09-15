from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .model import LEVELS


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("manifest must be .jsonl or .csv")
    for row in rows:
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        row["image_path"] = str(image_path)
    return rows


def validate_manifest(
    rows: list[dict[str, Any]], level_field: str = "level", split_field: str = "split", group_field: str = "group_id"
) -> None:
    if not rows:
        raise ValueError("manifest is empty")
    required_splits = {"train", "val", "test"}
    seen_splits: set[str] = set()
    groups_by_split: dict[str, set[str]] = {split: set() for split in required_splits}
    for index, row in enumerate(rows):
        missing = {"image_path", level_field, split_field, group_field} - row.keys()
        if missing:
            raise ValueError(f"manifest row {index} lacks fields: {sorted(missing)}")
        level = int(row[level_field])
        if level not in LEVELS:
            raise ValueError(f"manifest row {index} has invalid IDDSI level {level}; expected L3-L7")
        split = str(row[split_field])
        if split not in required_splits:
            raise ValueError(f"manifest row {index} has invalid split {split!r}")
        seen_splits.add(split)
        groups_by_split[split].add(str(row[group_field]))
    if seen_splits != required_splits:
        raise ValueError(f"manifest must contain train, val, and test; found {sorted(seen_splits)}")
    for left in required_splits:
        for right in required_splits:
            if left >= right:
                continue
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                example = sorted(overlap)[:3]
                raise ValueError(f"group leakage between {left} and {right}: {example}")


class BasicImageTransform:
    def __init__(
        self,
        image_size: int,
        augment: bool = False,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if self.augment and bool(torch.rand(()) < 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def build_transform(
    kind: str, model_name: str, image_size: int, augment: bool
) -> Callable[[Image.Image], torch.Tensor | dict[str, torch.Tensor]]:
    if kind != "siglip2":
        if kind == "resnet18":
            return BasicImageTransform(
                image_size, augment, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        return BasicImageTransform(image_size, augment)
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError("SigLIP-2 preprocessing requires `pip install -e '.[models]'`") from exc
    processor = AutoImageProcessor.from_pretrained(model_name)

    def transform(image: Image.Image) -> dict[str, torch.Tensor]:
        if augment and bool(torch.rand(()) < 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        processed = processor(images=image.convert("RGB"), return_tensors="pt")
        # SigLIP-2 processors may supply masks/spatial shapes in addition to pixels.
        return {
            key: value[0]
            for key, value in processed.items()
            if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }

    return transform


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        split: str,
        transform: Callable[[Image.Image], torch.Tensor | dict[str, torch.Tensor]],
        level_field: str = "level",
        split_field: str = "split",
    ):
        self.rows = [row for row in rows if str(row[split_field]) == split]
        self.transform = transform
        self.level_field = level_field

    @property
    def labels(self) -> list[int]:
        return [int(row[self.level_field]) - LEVELS[0] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            transformed = self.transform(image)
        model_inputs = transformed if isinstance(transformed, dict) else {"pixel_values": transformed}
        return {
            **model_inputs,
            "label": torch.tensor(int(row[self.level_field]) - LEVELS[0], dtype=torch.long),
            "event_id": str(row.get("event_id", index)),
        }
