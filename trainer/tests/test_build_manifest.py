"""Contract tests for train.build_manifest — the dataschema -> trainer bridge.

This module decides which rows are trainable and how whole (recipe, batch,
kitchen) groups are dealt into train/val/test. Both decisions are silent by
nature, so they are pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from train.build_manifest import (
    SINK_RULE,
    build_rows,
    main,
    read_event_metadata,
    stratified_group_split,
)
from train.data import load_manifest, validate_manifest
from train.model import LEVELS
from train.realdata import EventRecord, load_events

#: (level_weak, weak_label_record.rule, level_relation)
#: Four groups per level for L3/L4 (one carrying the sink rule, one carrying an
#: `at_most` relation) and three for L5-L7, so every level reaches all three
#: splits under the default 0.2/0.2 fractions.
SPEC: list[tuple[int, str, str]] = [
    (3, "visible_liquid_flow", "exact"),
    (3, SINK_RULE, "exact"),
    (3, "visible_liquid_flow", "exact"),
    (3, "visible_liquid_flow", "exact"),
    (4, "visible_liquid_flow", "at_most"),
    (4, "puree_consistency", "exact"),
    (4, "puree_consistency", "exact"),
    (4, "puree_consistency", "exact"),
    (5, "puree_consistency", "exact"),
    (5, "minced_and_moist", "exact"),
    (5, "minced_and_moist", "exact"),
    (6, "soft_and_bite_sized", "exact"),
    (6, "soft_and_bite_sized", "exact"),
    (6, "soft_and_bite_sized", "exact"),
    (7, "regular_texture", "exact"),
    (7, "regular_texture", "exact"),
    (7, "regular_texture", "exact"),
]


def _write_dataset(
    root: Path, spec: list[tuple[int, str, str]], *, duplicate_phash: int = 0
) -> Path:
    """Write a minimal dataschema dataset directory for ``spec``.

    ``duplicate_phash`` is how many of the final events reuse an earlier
    event's perceptual hash.
    """
    media = root / "media"
    media.mkdir(parents=True, exist_ok=True)
    events = []
    for index, (level, rule, relation) in enumerate(spec):
        relative = f"media/img_{index}.jpg"
        Image.new("RGB", (12, 10), color=(20 + index * 12, 60, 90)).save(
            root / relative, format="JPEG"
        )
        phash = f"hash_{index}"
        if duplicate_phash and index >= len(spec) - duplicate_phash:
            phash = f"hash_{index - (len(spec) - duplicate_phash)}"
        events.append(
            {
                "schema_version": "1.0.0",
                "event_id": f"E{index}",
                "recipe_id": f"rec_{index}",
                "batch_id": f"bat_{index}",
                "kitchen_id": "kitchen_1",
                "phone_id": "phone_1",
                "captured_at": "2026-01-01T00:00:00Z",
                "sample_kind": "food",
                "level_weak": level,
                "media_files": [
                    {"path": relative, "media_type": "image", "capture_role": "plate_photo"}
                ],
                "test_type": ["plate_photo"],
                "split": "train",
                "source": "real-weak",
                "consent_recorded": True,
                "contains_face": False,
                "phash": phash,
                "weak_label_record": {"rule": rule, "level_relation": relation, "confidence": 0.7},
            }
        )
    (root / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    return root


def _records(levels: list[int | None]) -> list[EventRecord]:
    return [
        EventRecord(
            f"e{index}",
            "real-weak",
            "train",
            level,
            (f"recipe_{index}", f"batch_{index}", f"kitchen_1"),
            f"/tmp/e{index}.jpg",
            "plate_photo",
        )
        for index, level in enumerate(levels)
    ]


# ---------------------------------------------------------------------------
# stratified_group_split
# ---------------------------------------------------------------------------


def test_groups_with_no_label_go_to_train_instead_of_crashing() -> None:
    """An all-unlabelled group used to raise IndexError from Counter([])."""
    records = _records([None, None, 4, 5])
    assignment = stratified_group_split(
        records, val_fraction=0.25, test_fraction=0.25, seed=1
    )
    assert set(assignment) == set(range(4))
    # Unlabelled records cannot be stratified, so they stay in train.
    assert assignment[0] == "train"
    assert assignment[1] == "train"
    assert assignment[2] in {"train", "val", "test"}
    assert assignment[3] in {"train", "val", "test"}


def test_a_two_group_level_bucket_still_reaches_val() -> None:
    """Two groups used to be dealt entirely into train.

    A level that never reaches ``val`` is invisible to temperature calibration
    and to the dangerous under-classification gate — the gate that decides
    whether the run's headline FNR is trustworthy. Two groups now give one to
    val and keep one in train.
    """
    records = _records([4, 4, 5, 5, 6, 6])
    assignment = stratified_group_split(
        records, val_fraction=0.2, test_fraction=0.2, seed=1
    )
    for level_records in ((0, 1), (2, 3), (4, 5)):
        splits = {assignment[i] for i in level_records}
        assert splits == {"val", "train"}, f"bucket {level_records} got {splits}"


def test_small_buckets_never_empty_train() -> None:
    """Guaranteeing val must not cost the level its training data."""
    for levels in ([4, 4], [4, 4, 5, 5], [4, 4, 4]):
        assignment = stratified_group_split(
            _records(levels), val_fraction=0.2, test_fraction=0.2, seed=1
        )
        assert "train" in set(assignment.values()), levels


@pytest.mark.parametrize("n_groups", list(range(2, 31)))
def test_every_bucket_keeps_train_and_reaches_both_measurement_splits(n_groups: int) -> None:
    """Pin the deal-out counts for every bucket size, not just the tiny ones.

    The v0.7 manifest has 26-322 groups per level, so this is the path that
    actually runs in production; the small-bucket guard must not perturb it.
    """
    assignment = stratified_group_split(
        _records([4] * n_groups), val_fraction=0.2, test_fraction=0.2, seed=4
    )
    counts = {
        split: sum(1 for value in assignment.values() if value == split)
        for split in ("train", "val", "test")
    }
    assert sum(counts.values()) == n_groups
    assert counts["train"] >= 1, n_groups
    assert counts["val"] >= 1, n_groups
    if n_groups >= 3:
        assert counts["test"] >= 1, n_groups


def test_buckets_with_three_or_more_groups_keep_one_in_train() -> None:
    """The small-bucket guard must not disturb the normal large-bucket path."""
    records = _records([4] * 10)
    assignment = stratified_group_split(
        records, val_fraction=0.2, test_fraction=0.2, seed=3
    )
    counts = {split: sum(1 for v in assignment.values() if v == split) for split in ("train", "val", "test")}
    assert counts["train"] >= 1
    assert counts["val"] >= 1
    assert counts["test"] >= 1


def test_stratified_group_split_is_deterministic_and_covers_every_record() -> None:
    records = _records([3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
    first = stratified_group_split(records, val_fraction=0.2, test_fraction=0.2, seed=9)
    second = stratified_group_split(records, val_fraction=0.2, test_fraction=0.2, seed=9)
    other = stratified_group_split(records, val_fraction=0.2, test_fraction=0.2, seed=10)
    assert first == second
    assert first != other
    assert set(first) == set(range(len(records)))
    assert set(first.values()) == {"train", "val", "test"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"val_fraction": 0.0, "test_fraction": 0.2},
        {"val_fraction": 0.2, "test_fraction": 1.0},
        {"val_fraction": 0.6, "test_fraction": 0.6},
    ],
)
def test_stratified_group_split_rejects_impossible_fractions(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        stratified_group_split(_records([4, 5, 6, 7]), seed=1, **kwargs)


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------


def _build(tmp_path: Path, **kwargs: object) -> tuple[list[dict], dict]:
    dataset = _write_dataset(tmp_path / "ds", SPEC)
    defaults: dict[str, object] = dict(
        capture_role="plate_photo",
        drop_rules=set(),
        drop_at_most=False,
        dedup_phash=True,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=17,
    )
    defaults.update(kwargs)
    return build_rows([dataset], **defaults)  # type: ignore[arg-type]


def test_build_rows_keeps_only_trainable_l3_l7_events(tmp_path: Path) -> None:
    rows, stats = _build(tmp_path)
    # No filters are applied by default and every phash is distinct here.
    assert stats["kept_total"] == len(rows) == len(SPEC)
    assert stats["dropped_duplicate_phash"] == 0
    assert {row["level"] for row in rows} <= set(LEVELS)
    assert set(stats["split_counts"]) == {"train", "val", "test"}


def test_drop_sink_rule_and_at_most_and_dedup_are_all_reported(tmp_path: Path) -> None:
    rows, stats = _build(
        tmp_path,
        drop_rules={SINK_RULE},
        drop_at_most=True,
        dedup_phash=False,
    )
    # One sink-rule row and one at_most row are dropped; no phash dedup here.
    assert stats["kept_total"] == len(rows) == len(SPEC) - 2
    assert stats["dropped_by_rule"] == {SINK_RULE: 1}
    assert stats["dropped_at_most"] == 1
    assert stats["dropped_duplicate_phash"] == 0
    # Nothing is dropped silently: every written row is accounted for.
    accounted = (
        stats["kept_total"]
        + stats["dropped_out_of_range_level"]
        + sum(stats["dropped_by_rule"].values())
        + stats["dropped_at_most"]
        + stats["dropped_duplicate_phash"]
        + stats["dropped_duplicate_event_id"]
    )
    assert accounted == len(SPEC)


def test_phash_dedup_keeps_the_first_occurrence(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "ds", SPEC, duplicate_phash=3)
    rows, stats = build_rows(
        [dataset],
        capture_role="plate_photo",
        drop_rules=set(),
        drop_at_most=False,
        dedup_phash=True,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=17,
    )
    assert stats["dropped_duplicate_phash"] == 3
    kept_ids = [row["event_id"] for row in rows]
    assert f"E{len(SPEC) - 3}" not in kept_ids  # the later duplicate is the one dropped
    assert "E0" in kept_ids


def test_out_of_range_levels_are_dropped_and_counted(tmp_path: Path) -> None:
    spec = [(2, "some_rule", "exact"), (4, "puree_consistency", "exact"),
            (5, "puree_consistency", "exact"), (6, "soft_and_bite_sized", "exact")]
    dataset = _write_dataset(tmp_path / "ds", spec)
    rows, stats = build_rows(
        [dataset],
        capture_role="plate_photo",
        drop_rules=set(),
        drop_at_most=False,
        dedup_phash=True,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=2,
    )
    assert stats["dropped_out_of_range_level"] == 1
    assert all(row["level"] in LEVELS for row in rows)


def test_build_rows_raises_when_everything_is_filtered(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "ds", SPEC)
    with pytest.raises(ValueError, match="every event was filtered out"):
        build_rows(
            [dataset],
            capture_role="plate_photo",
            drop_rules={SINK_RULE, "visible_liquid_flow", "puree_consistency",
                        "minced_and_moist", "soft_and_bite_sized", "regular_texture"},
            drop_at_most=False,
            dedup_phash=False,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=1,
        )


def test_build_rows_skips_a_dataset_with_no_labelled_events(tmp_path: Path) -> None:
    """A mid-bootstrap `deferred` dataset is normal, not a build failure."""
    empty = tmp_path / "empty"
    (empty / "media").mkdir(parents=True)
    (empty / "events.jsonl").write_text(
        json.dumps(
            {
                "event_id": "D0", "recipe_id": "r", "batch_id": "b", "kitchen_id": "k",
                "source": "real-weak", "split": "train", "level_weak": None,
                "media_files": [{"path": "media/x.jpg", "media_type": "image",
                                 "capture_role": "plate_photo"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = _write_dataset(tmp_path / "ds", SPEC)
    rows, stats = build_rows(
        [empty, dataset],
        capture_role="plate_photo",
        drop_rules=set(),
        drop_at_most=False,
        dedup_phash=True,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=17,
    )
    assert stats["labelled_events_per_dir"][str(empty)] == 0
    assert stats["kept_total"] == len(rows) > 0


def test_no_group_straddles_two_splits(tmp_path: Path) -> None:
    rows, _ = _build(tmp_path)
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------


def test_main_writes_a_manifest_the_trainer_accepts(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "ds", SPEC)
    out = tmp_path / "manifest.jsonl"
    exit_code = main(
        [
            "--dataset", str(dataset),
            "--out", str(out),
            "--drop-sink",
            "--drop-at-most",
            "--seed", "17",
        ]
    )
    assert exit_code == 0
    assert out.is_file()
    rows = load_manifest(out)
    # This is the contract: the flat manifest must pass the trainer's own
    # validator unchanged, including the group-leakage and split-coverage rules.
    validate_manifest(rows)
    assert len(rows) == len(SPEC) - 2
    assert {row["split"] for row in rows} == {"train", "val", "test"}


def test_read_event_metadata_exposes_the_weak_label_fields(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "ds", SPEC)
    meta = read_event_metadata(dataset)
    assert set(meta) == {f"E{i}" for i in range(len(SPEC))}
    assert meta["E1"]["rule"] == SINK_RULE
    assert meta["E4"]["level_relation"] == "at_most"
    assert meta["E0"]["phash"] == "hash_0"
    assert meta["E0"]["confidence"] == 0.7


def test_load_events_agrees_with_the_manifest_loader(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "ds", SPEC)
    records = load_events(dataset)
    assert len(records) == len(SPEC)
    assert {record.source for record in records} == {"real-weak"}
