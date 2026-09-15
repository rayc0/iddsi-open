"""Contract tests for the W20 dataschema real-data loader."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from train.realdata import (
    AugmentConfig,
    EventsDataset,
    SeededAugment,
    build_weighted_sampler,
    group_split,
    load_events,
    sample_weights,
    validate_no_group_leak,
)


def _event(
    event_id: str,
    *,
    source: str,
    split: str,
    level: int | None,
    image_path: str,
    group_number: int,
) -> dict[str, object]:
    """Return a complete base-schema event plus the loader's weak-label field."""
    event: dict[str, object] = {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "recipe_id": f"recipe_{group_number}",
        "batch_id": f"batch_{group_number}",
        "kitchen_id": f"kitchen_{group_number}",
        "phone_id": "phone_1",
        "captured_at": "2026-01-01T00:00:00Z",
        "sample_kind": "food",
        "level_declared": level if source == "synthetic" else None,
        "level_tested_RD": None,
        "level_tested_SLP": None,
        "level_adjudicated": None,
        "media_files": [
            {
                "path": image_path,
                "media_type": "image",
                "capture_role": "plate_photo",
            }
        ],
        "test_type": ["plate_photo"],
        "capture_conditions": {
            "operator_id": "operator_1",
            "lighting_category": "controlled",
            "lighting_notes": "",
            "plate_category": "light_plain",
            "plate_notes": "",
            "serving_temperature_c": None,
            "cuisine_tags": ["fixture"],
        },
        "ground_truth_record": {
            "rd_assessment_id": None,
            "slp_assessment_id": None,
            "adjudication_status": "not_applicable",
            "adjudicator_code": None,
            "adjudication_notes": "",
        },
        "notes": "test fixture",
        "split": split,
        "source": source,
        "consent_recorded": True,
        "contains_face": False,
    }
    if source == "real-weak":
        event["level_weak"] = level
    return event


def _write_source_dir(
    root: Path,
    name: str,
    source: str,
    levels: list[int | None],
    *,
    group_offset: int,
) -> Path:
    source_dir = root / name
    media_dir = source_dir / "media"
    media_dir.mkdir(parents=True)
    events = []
    for index, level in enumerate(levels):
        relative_path = f"media/{name}_{index}.jpg"
        Image.new(
            "RGB",
            (12, 10),
            color=(20 + index * 30, 40 + index * 20, 60 + index * 10),
        ).save(source_dir / relative_path, format="JPEG")
        events.append(
            _event(
                f"{name}_{index}",
                source=source,
                split="train" if index % 2 == 0 else "val",
                level=level,
                image_path=relative_path,
                group_number=group_offset + index,
            )
        )
    (source_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    return source_dir


@pytest.fixture
def source_dirs(tmp_path: Path) -> tuple[Path, Path]:
    synthetic = _write_source_dir(
        tmp_path, "synthetic", "synthetic", [0, 1, 2, None], group_offset=0
    )
    real_weak = _write_source_dir(
        tmp_path, "real_weak", "real-weak", [3, 4, 5], group_offset=10
    )
    return synthetic, real_weak


def test_dataset_length_labels_and_group_split(source_dirs: tuple[Path, Path]) -> None:
    synthetic, real_weak = source_dirs
    records = load_events(synthetic) + load_events(real_weak)
    dataset = EventsDataset(records)

    assert len(dataset) == 6
    assert dataset.labels == [0, 1, 2, 3, 4, 5]
    assert dataset.sources == ["synthetic"] * 3 + ["real-weak"] * 3

    train_indices, val_indices = group_split(records, val_fraction=0.5, seed=7)
    train_groups = {records[index].group_key for index in train_indices}
    val_groups = {records[index].group_key for index in val_indices}
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)


def test_source_weighted_sampler_matches_requested_proportions(
    source_dirs: tuple[Path, Path],
) -> None:
    synthetic, real_weak = source_dirs
    records = load_events(synthetic) + load_events(real_weak)
    sampler = build_weighted_sampler(
        records,
        {"synthetic": 1.0, "real-weak": 3.0},
        seed=11,
        num_samples=12_000,
    )

    weak_draws = sum(records[index].source == "real-weak" for index in sampler)
    # Three records per source and a 1:3 per-record weight imply 75% weak draws.
    assert weak_draws / 12_000 == pytest.approx(0.75, abs=0.02)


def test_preassigned_group_leakage_is_rejected(source_dirs: tuple[Path, Path]) -> None:
    synthetic, _ = source_dirs
    records = load_events(synthetic)
    leaked = [records[0], replace(records[1], group_key=records[0].group_key, split="val")]

    with pytest.raises(ValueError, match="group leakage across splits"):
        validate_no_group_leak(leaked)


def test_null_label_is_dropped_or_retained_as_unsupervised(
    source_dirs: tuple[Path, Path],
) -> None:
    synthetic, _ = source_dirs

    assert len(load_events(synthetic)) == 3
    records = load_events(synthetic, keep_unlabeled=True)
    assert len(records) == 4
    assert records[-1].label is None
    assert sample_weights(records, {"synthetic": 2.0}) == [2.0, 2.0, 2.0, 0.0]

    item = EventsDataset(records)[-1]
    assert item["label"].item() == -1


@pytest.mark.parametrize("enabled", [False, True])
def test_augmentation_toggle_runs_on_cpu(
    source_dirs: tuple[Path, Path], enabled: bool
) -> None:
    synthetic, _ = source_dirs
    records = load_events(synthetic)
    transform = SeededAugment(
        AugmentConfig(
            image_size=16,
            horizontal_flip_p=1.0,
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            blur_p=1.0,
            blur_radius=0.5,
            crop_scale=(0.8, 1.0),
        ),
        enabled=enabled,
    )

    tensor = EventsDataset(records, transform=transform, seed=23)[0]["pixel_values"]
    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert torch.isfinite(tensor).all()
