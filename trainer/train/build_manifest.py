"""Bridge dataschema event datasets into the flat manifest the trainer consumes.

Two loaders exist in this package and until now nothing joined them:

``realdata.py``
    Reads ``events.jsonl`` (dataschema format) into :class:`realdata.EventRecord`
    with ``(recipe_id, batch_id, kitchen_id)`` group keys and leak-checked splits.
``data.py``
    What :mod:`trainer.train.engine` actually uses. It wants a *flat* manifest:
    ``image_path``, ``level`` (3-7), ``split`` (train/val/test), ``group_id``.

This CLI reads N dataset directories through ``realdata.load_events``, drops rows
that are not trainable, assigns whole groups to train/val/test stratified by
level, and writes a manifest that ``data.validate_manifest`` accepts unchanged.

Trainability filters (all reported in the summary, none silent):

``--drop-rule RULE``
    Drop rows whose ``weak_label_record.rule`` matches. The important one is
    ``other_or_unmeasurable_l7_visual_bucket``: it is the ``else`` branch of
    ``harvest.label.apply_descriptor_rules``, i.e. "nothing measurable was
    observed", not positive evidence of regular-texture food. Training on it
    teaches the model to answer L7 whenever it is unsure, which is the
    dangerous direction for a dysphagia application.
``--drop-at-most``
    Drop rows whose ``level_relation`` is ``at_most``. A "visible liquid flow"
    observation means "L3 *or below*", but the loader resolves it to a hard
    ``level_weak=3``, silently conflating L0-L3 into L3.
``--dedup-phash``
    Keep the first occurrence of each perceptual hash. Group keys here are
    near-unique per image, so group-based split protection does not stop the
    same photo appearing in two splits; the phash does.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from .model import LEVELS
from .realdata import EventRecord, load_events

SINK_RULE = "other_or_unmeasurable_l7_visual_bucket"


def read_event_metadata(dataset_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Map ``event_id`` -> weak-label metadata that ``EventRecord`` does not carry."""
    path = Path(dataset_dir).resolve() / "events.jsonl"
    meta: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            record = event.get("weak_label_record") or {}
            meta[str(event.get("event_id", f"line_{line_no}"))] = {
                "rule": record.get("rule"),
                "level_relation": record.get("level_relation"),
                "confidence": record.get("confidence"),
                "phash": event.get("phash"),
            }
    return meta


def stratified_group_split(
    records: Sequence[EventRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[int, str]:
    """Assign whole group keys to train/val/test, stratified by level.

    Groups are bucketed by their majority level and dealt out per bucket, so a
    level with only a handful of groups still reaches val and test instead of
    landing entirely in train by luck of the shuffle.
    """
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must leave room for train")

    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.group_key, []).append(index)

    by_level: dict[int, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for key, indices in groups.items():
        levels = [records[i].label for i in indices if records[i].label is not None]
        majority = collections.Counter(levels).most_common(1)[0][0]
        by_level[majority].append(key)

    assignment: dict[int, str] = {}
    rng = random.Random(seed)
    for level in sorted(by_level):
        keys = sorted(by_level[level])
        rng.shuffle(keys)
        n = len(keys)
        n_test = round(n * test_fraction)
        n_val = round(n * val_fraction)
        # With very few groups, guarantee val/test get one before train does.
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 2))
            n_val = min(n_val, max(0, n - 1 - n_test))
        for position, key in enumerate(keys):
            if position < n_test:
                split = "test"
            elif position < n_test + n_val:
                split = "val"
            else:
                split = "train"
            for index in groups[key]:
                assignment[index] = split
    return assignment


def build_rows(
    dataset_dirs: Iterable[str | Path],
    *,
    capture_role: str,
    drop_rules: set[str],
    drop_at_most: bool,
    dedup_phash: bool,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[EventRecord] = []
    meta: dict[str, dict[str, Any]] = {}
    per_dir: dict[str, int] = {}
    for dataset_dir in dataset_dirs:
        try:
            loaded = load_events(dataset_dir, capture_role=capture_role, keep_unlabeled=False)
        except ValueError:
            # A dataset whose rows are all still `deferred` has nothing labelled
            # yet. That is a normal mid-bootstrap state, not a build failure.
            per_dir[str(dataset_dir)] = 0
            continue
        per_dir[str(dataset_dir)] = len(loaded)
        records.extend(loaded)
        meta.update(read_event_metadata(dataset_dir))

    if not records:
        raise ValueError(
            "no labelled events in any dataset; run `python3 -m harvest.relabel` first"
        )

    stats: dict[str, Any] = {
        "labelled_events_per_dir": per_dir,
        "labelled_total": len(records),
        "dropped_out_of_range_level": 0,
        "dropped_by_rule": collections.Counter(),
        "dropped_at_most": 0,
        "dropped_duplicate_phash": 0,
        "dropped_duplicate_event_id": 0,
    }

    kept: list[EventRecord] = []
    seen_phash: set[str] = set()
    seen_event_ids: set[str] = set()
    for record in records:
        info = meta.get(record.event_id, {})
        if record.label not in LEVELS:
            stats["dropped_out_of_range_level"] += 1
            continue
        rule = info.get("rule")
        if rule in drop_rules:
            stats["dropped_by_rule"][rule] += 1
            continue
        if drop_at_most and info.get("level_relation") == "at_most":
            stats["dropped_at_most"] += 1
            continue
        if record.event_id in seen_event_ids:
            stats["dropped_duplicate_event_id"] += 1
            continue
        phash = info.get("phash")
        if dedup_phash and phash and phash in seen_phash:
            stats["dropped_duplicate_phash"] += 1
            continue
        if phash:
            seen_phash.add(phash)
        seen_event_ids.add(record.event_id)
        kept.append(record)

    if not kept:
        raise ValueError("every event was filtered out; nothing to train on")

    assignment = stratified_group_split(
        kept, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(kept):
        info = meta.get(record.event_id, {})
        rows.append(
            {
                "event_id": record.event_id,
                "image_path": record.image_path,
                "level": int(record.label),
                "split": assignment[index],
                "group_id": "/".join(record.group_key),
                "source": record.source,
                "weak_rule": info.get("rule"),
                "weak_confidence": info.get("confidence"),
            }
        )

    stats["kept_total"] = len(rows)
    stats["dropped_by_rule"] = dict(stats["dropped_by_rule"])
    stats["split_counts"] = dict(collections.Counter(r["split"] for r in rows))
    stats["level_counts"] = dict(sorted(collections.Counter(r["level"] for r in rows).items()))
    stats["level_by_split"] = {
        split: dict(sorted(collections.Counter(r["level"] for r in rows if r["split"] == split).items()))
        for split in ("train", "val", "test")
    }
    stats["source_counts"] = dict(collections.Counter(r["source"] for r in rows))
    return rows, stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", required=True, help="dataset directory (repeatable)")
    parser.add_argument("--out", required=True, type=Path, help="manifest.jsonl to write")
    parser.add_argument("--capture-role", default="plate_photo")
    parser.add_argument("--drop-rule", action="append", default=[], help="weak_label_record.rule to drop (repeatable)")
    parser.add_argument("--drop-sink", action="store_true", help=f"shorthand for --drop-rule {SINK_RULE}")
    parser.add_argument("--drop-at-most", action="store_true", help="drop level_relation=at_most rows")
    parser.add_argument("--dedup-phash", action="store_true", default=True)
    parser.add_argument("--no-dedup-phash", dest="dedup_phash", action="store_false")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    drop_rules = set(args.drop_rule)
    if args.drop_sink:
        drop_rules.add(SINK_RULE)

    rows, stats = build_rows(
        args.dataset,
        capture_role=args.capture_role,
        drop_rules=drop_rules,
        drop_at_most=args.drop_at_most,
        dedup_phash=args.dedup_phash,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Fail loudly here rather than deep inside the training loop.
    from .data import load_manifest, validate_manifest

    validate_manifest(load_manifest(args.out))
    stats["manifest"] = str(args.out)
    stats["validated"] = True
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
