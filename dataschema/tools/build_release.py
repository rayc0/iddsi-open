#!/usr/bin/env python3
"""Assemble an IDDSI-Open release from mixed source directories.

Inputs are one or more ``--source tag=path`` dataset directories, each with its
own ``events.jsonl`` and ``media/`` tree (the shape produced by the capture
pipeline and by ``harvest/``). Tags are the four release source classes:
``synthetic``, ``real-weak``, ``real-crowd``, ``real-tested``.

The builder writes, into ``--out``:

* ``events.parquet``       — one row per event in the CC BY-NC-SA main
  release, plus ``release_source`` and ``licence_partition=main`` columns
  (pyarrow when installed, otherwise polars; the engine used is recorded in
  the stats files);
* ``media_manifest.jsonl`` — one line per media file: release source, event,
  path, sha256 (always recomputed from disk), byte count, role, and type;
* ``source_counts.json``   — per-source event counts;
* ``licence_manifest.json``— per-source licence/provider counts, the allowlist
  each source was checked against, and the full attribution records; and
* ``datasheet_stats.json`` — counts the dataset cards and datasheet cite
  (per-source, split, licence, provider, sample-kind, level, media, and group
  statistics).

Fail-closed rules; any violation refuses the release and nothing is written:

* every non-synthetic row must carry attribution (an ``attribution.jsonl``
  entry for its ``event_id`` in its source directory, or in-row
  ``provenance`` with the same fields) with non-empty attribution text and a
  licence name;
* CC BY-SA rows are excluded from this CC BY-NC-SA main release and recorded
  in ``licence_manifest.json`` and the stats files; this builder does not
  create the optional, separately licensed BY-SA package described by
  ``docs/DATA_GOVERNANCE.md``;
* rows whose ``verification_status`` is not explicitly ``verified`` (including
  rows whose record predates the field and leaves it absent) are excluded
  until their per-image licence has been verified at the recorded landing
  page;
* when a source directory ships ``source_licenses.json`` with
  ``allowed_licence_families``, every attributed row's licence must be in
  that allowlist;
* media files must exist inside their source directory, and recorded
  sha256/bytes values must match the files on disk;
* event ids must be unique across all sources;
* splits are reassigned group-wise (connected components over ``recipe_id``,
  ``batch_id``, and ``kitchen_id``) with a deterministic seed, so no group
  value spans train/val/external_test, and synthetic rows never enter
  external_test; and
* rows tagged ``synthetic`` are additionally validated against
  ``schema/event.schema.json`` with the existing validator.

Input ``split`` values are ignored: the builder owns split assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .validate_dataset import GROUP_FIELDS, _read_jsonl, validate_rows
else:  # executed as a script: python3 dataschema/tools/build_release.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.validate_dataset import GROUP_FIELDS, _read_jsonl, validate_rows

SCHEMA_VERSION = "1.0.0"
RELEASE_SOURCES = ("synthetic", "real-weak", "real-crowd", "real-tested")
SOURCE_ALIASES = {
    "synthetic": "synthetic",
    "synth": "synthetic",
    "real-weak": "real-weak",
    "realweak": "real-weak",
    "weak": "real-weak",
    "real-crowd": "real-crowd",
    "crowd": "real-crowd",
    "real-tested": "real-tested",
    "tested": "real-tested",
}
SPLITS = ("train", "val", "external_test")
LEVEL_FIELDS = ("level_declared", "level_tested_RD", "level_tested_SLP", "level_adjudicated")
REQUIRED_ATTRIBUTION_FIELDS = ("attribution", "license_name")
# harvest/licenses.py treats "CC0" and "CC0 1.0" as the same explicit-CC0
# family; mirror that so short-form licence names still pass the allowlist.
LICENCE_ALIASES = {"CC0": "CC0 1.0"}
MAIN_DATASET_LICENCE = "CC BY-NC-SA 4.0"
OUTPUT_FILES = (
    "events.parquet",
    "media_manifest.jsonl",
    "source_counts.json",
    "licence_manifest.json",
    "datasheet_stats.json",
)


class ReleaseError(Exception):
    """Raised when the release must be refused. ``errors`` lists every reason."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


# ---------------------------------------------------------------------------
# input parsing


def _parse_sources(values: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ReleaseError([f"--source expects tag=path, got {value!r}"])
        tag, _, raw_path = value.partition("=")
        if not raw_path:
            raise ReleaseError([f"--source expects tag=path, got {value!r}"])
        normalized = SOURCE_ALIASES.get(tag.strip().lower())
        if normalized is None:
            allowed = ", ".join(RELEASE_SOURCES)
            raise ReleaseError([f"unknown source tag {tag!r}; expected one of: {allowed}"])
        path = Path(raw_path).expanduser()
        if normalized in sources:
            raise ReleaseError([f"source tag {normalized!r} given more than once"])
        if not path.is_dir():
            raise ReleaseError([f"source directory does not exist: {path}"])
        sources[normalized] = path
    if not sources:
        raise ReleaseError(["at least one --source tag=path is required"])
    return sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_licence_policy(source_dir: Path) -> dict[str, Any]:
    policy_path = source_dir / "source_licenses.json"
    if not policy_path.is_file():
        return {}
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return policy if isinstance(policy, dict) else {}


# ---------------------------------------------------------------------------
# attribution


def _effective_attribution(
    row: dict[str, Any],
    attribution_by_event: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the attribution record for a row, preferring attribution.jsonl."""
    record = attribution_by_event.get(row.get("event_id"))
    if isinstance(record, dict):
        return record
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        return provenance
    return None


def _check_row_attribution(
    tag: str,
    row: dict[str, Any],
    row_number: int,
    source_dir: Path,
    attribution_by_event: dict[str, dict[str, Any]],
    allowed_families: Sequence[str],
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate one row's attribution; return the record for the manifest."""
    event_id = row.get("event_id", f"<row {row_number}>")
    where = f"{source_dir.name}/events.jsonl row {row_number} (event {event_id})"
    record = _effective_attribution(row, attribution_by_event)
    if tag == "synthetic":
        # In-house generated rows carry no third-party attribution; the
        # dataset-level licence in docs/LICENSE_PLAN.md governs them.
        return record if isinstance(record, dict) else None
    if record is None:
        errors.append(f"{where}: no attribution record (attribution.jsonl entry or provenance) — refusing release")
        return None
    for field in REQUIRED_ATTRIBUTION_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{where}: attribution record lacks non-empty {field!r} — refusing release")
    if allowed_families:
        licence = record.get("license_name")
        if isinstance(licence, str) and licence:
            normalized = LICENCE_ALIASES.get(licence, licence)
            if normalized not in allowed_families:
                errors.append(
                    f"{where}: licence {licence!r} is not in the allowlist for this source "
                    f"({', '.join(allowed_families)}) — refusing release"
                )
    return record


def _is_sharealike_licence(record: dict[str, Any] | None) -> bool:
    """Return whether an attribution record carries a CC BY-SA licence."""
    if not isinstance(record, dict):
        return False
    licence = record.get("license_name")
    if not isinstance(licence, str):
        return False
    normalized = " ".join(licence.upper().replace("_", "-").split())
    return normalized.startswith("CC BY-SA ") or normalized == "CC BY-SA"


def _is_unverified(record: dict[str, Any] | None) -> bool:
    """Return whether a discovery record is still awaiting per-image verification.

    Fail closed on field drift: the current harvest pipeline always records an
    explicit ``verification_status`` (``verified`` or ``unverified``) in both
    the event provenance and attribution.jsonl. A non-synthetic row whose
    record predates that convention has an unknown per-image verification
    state, so a missing or unrecognized value is treated as unverified rather
    than silently admitted to the release.
    """
    if not isinstance(record, dict):
        return False
    status = record.get("verification_status")
    if not isinstance(status, str):
        return True
    return status.strip().casefold() != "verified"


# ---------------------------------------------------------------------------
# media checks and manifest


def _check_row_media(
    row: dict[str, Any],
    row_number: int,
    source_dir: Path,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Verify media files and return manifest entries (sha256 recomputed)."""
    event_id = row.get("event_id", f"<row {row_number}>")
    media_files = row.get("media_files")
    entries: list[dict[str, Any]] = []
    if not isinstance(media_files, list) or not media_files:
        errors.append(
            f"{source_dir.name}/events.jsonl row {row_number} (event {event_id}): "
            "media_files must be a non-empty list — refusing release"
        )
        return entries
    root = source_dir.resolve()
    for index, media in enumerate(media_files):
        if not isinstance(media, dict) or not isinstance(media.get("path"), str):
            errors.append(
                f"{source_dir.name}/events.jsonl row {row_number} (event {event_id}): "
                f"media_files[{index}] must be an object with a path"
            )
            continue
        relative = media["path"]
        candidate = (source_dir / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(
                f"{source_dir.name}/events.jsonl row {row_number} (event {event_id}): "
                f"media_files[{index}].path escapes the source directory: {relative}"
            )
            continue
        if not candidate.is_file():
            errors.append(
                f"{source_dir.name}/events.jsonl row {row_number} (event {event_id}): "
                f"media file does not exist: {relative}"
            )
            continue
        computed_sha = _sha256(candidate)
        computed_bytes = candidate.stat().st_size
        recorded_sha = media.get("sha256")
        if isinstance(recorded_sha, str) and recorded_sha != computed_sha:
            errors.append(
                f"{source_dir.name}/events.jsonl row {row_number} (event {event_id}): "
                f"media_files[{index}].sha256 does not match the file: {relative}"
            )
        recorded_bytes = media.get("bytes")
        if isinstance(recorded_bytes, int) and recorded_bytes != computed_bytes:
            errors.append(
                f"{source_dir.name}/events.jsonl row {row_number} (event {event_id}): "
                f"media_files[{index}].bytes does not match the file: {relative}"
            )
        entries.append(
            {
                "event_id": event_id,
                "media_path": relative,
                "media_type": media.get("media_type"),
                "capture_role": media.get("capture_role"),
                "sha256": computed_sha,
                "bytes": computed_bytes,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# grouped split assignment


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        parent = self.parent
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _assign_grouped_splits(
    rows: list[dict[str, Any]],
    tags: list[str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
    errors: list[str],
) -> int:
    """Assign train/val/external_test per connected group; return group count.

    Groups are connected components over shared recipe_id/batch_id/kitchen_id
    values (the GROUP_FIELDS the validator enforces), so no value of any of
    those fields can span splits. Assignment is a deterministic hash of the
    seed and the component key, so the same inputs and seed always produce
    the same splits.
    """
    union_find = _UnionFind()
    row_nodes: list[list[str]] = []
    for row in rows:
        nodes = [f"{field}:{row[field]}" for field in GROUP_FIELDS if isinstance(row.get(field), str)]
        if not nodes:
            nodes = [f"event:{row.get('event_id', id(row))}"]
        for node in nodes[1:]:
            union_find.union(nodes[0], node)
        row_nodes.append(nodes)

    components: dict[str, list[int]] = defaultdict(list)
    for index, nodes in enumerate(row_nodes):
        components[union_find.find(nodes[0])].append(index)

    component_keys: dict[str, str] = {}
    for root, indices in components.items():
        component_keys[root] = min(
            (str(rows[index].get("event_id", index)) for index in indices), key=str
        )

    split_of: dict[str, str] = {}
    for root, indices in components.items():
        digest = hashlib.sha256(f"{seed}|{component_keys[root]}".encode("utf-8")).hexdigest()
        u = int(digest[:16], 16) / float(1 << 64)
        if u < test_fraction:
            split = "external_test"
        elif u < test_fraction + val_fraction:
            split = "val"
        else:
            split = "train"
        if split == "external_test" and any(tags[index] == "synthetic" for index in indices):
            # Synthetic events are development-only and can never enter
            # external_test; park such groups in val.
            split = "val"
        split_of[root] = split
        for index in indices:
            rows[index]["split"] = split

    # Post-condition: no group field value spans splits.
    seen: dict[str, dict[str, set[str]]] = {field: defaultdict(set) for field in GROUP_FIELDS}
    for row in rows:
        for field in GROUP_FIELDS:
            value = row.get(field)
            if isinstance(value, str):
                seen[field][value].add(row["split"])
    for field, groups in seen.items():
        for value, splits in sorted(groups.items()):
            if len(splits) > 1:
                errors.append(f"grouped split violated for {field}={value!r}: spans {sorted(splits)}")
    for row, tag in zip(rows, tags):
        if tag == "synthetic" and row["split"] == "external_test":
            errors.append(f"synthetic event {row.get('event_id')!r} landed in external_test")
    return len(components)


# ---------------------------------------------------------------------------
# parquet writing


def _parquet_safe(value: Any) -> Any:
    """Replace empty JSON objects with null.

    Real-weak rows carry fields such as ``weak_label_record.observations: {}``;
    an empty struct cannot be serialized to Parquet (polars refuses and Arrow
    has no zero-field struct), so empty objects become null in the parquet.
    The source JSONL and the JSON manifests keep the original empty object.
    """
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _parquet_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_parquet_safe(child) for child in value]
    return value


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> str:
    """Write rows to parquet; return the engine name used.

    pyarrow is the layout's reference engine; polars is the fallback actually
    installed on machines without pyarrow. Both write the same nested data.
    """
    rows = [_parquet_safe(row) for row in rows]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pass
    else:
        pq.write_table(pa.Table.from_pylist(rows), path)
        return "pyarrow"
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on target image
        raise RuntimeError(
            "parquet output requires pyarrow or polars; neither is installed"
        ) from exc
    # infer_schema_length=None scans every row: mixed sources put int levels
    # next to null levels, and sampling a prefix can infer the wrong type.
    frame = pl.DataFrame(rows, infer_schema_length=None)
    casts = [pl.col(field).cast(pl.Int8) for field in LEVEL_FIELDS if field in frame.columns]
    if "captured_at" in frame.columns:
        casts.append(pl.col("captured_at").cast(pl.String))
    if casts:
        frame = frame.with_columns(*casts)
    frame.write_parquet(path)
    return "polars"


# ---------------------------------------------------------------------------
# release assembly


def build_release(
    sources: dict[str, Path],
    out: Path,
    *,
    seed: int = 13,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    release_version: str = "unversioned",
    copy_media: bool = False,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the release; raise ReleaseError (writing nothing) on refusal."""
    errors: list[str] = []
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0 or val_fraction + test_fraction >= 1.0:
        raise ReleaseError(["val_fraction and test_fraction must be in [0,1) and sum below 1"])
    if out.exists() and any(out.iterdir()):
        raise ReleaseError([f"output directory is not empty: {out}"])

    schema = None
    if schema_path is None:
        schema_path = Path(__file__).resolve().parents[1] / "schema" / "event.schema.json"
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    tags: list[str] = []
    seen_event_ids: dict[str, str] = {}
    media_entries: list[dict[str, Any]] = []
    attribution_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    input_counts: Counter = Counter()
    licence_policy_by_source: dict[str, Any] = {}

    for tag, source_dir in sorted(sources.items()):
        events_path = source_dir / "events.jsonl"
        if not events_path.is_file():
            errors.append(f"source {tag!r}: no events.jsonl in {source_dir}")
            continue
        try:
            source_rows = _read_jsonl(events_path)
        except (OSError, ValueError) as exc:
            errors.append(f"source {tag!r}: cannot read {events_path}: {exc}")
            continue
        if not source_rows:
            errors.append(f"source {tag!r}: events.jsonl is empty")

        attribution_by_event: dict[str, dict[str, Any]] = {}
        attribution_path = source_dir / "attribution.jsonl"
        if attribution_path.is_file():
            try:
                for record in _read_jsonl(attribution_path):
                    if isinstance(record.get("event_id"), str):
                        attribution_by_event[record["event_id"]] = record
            except (OSError, ValueError) as exc:
                errors.append(f"source {tag!r}: cannot read {attribution_path}: {exc}")

        policy = _load_source_licence_policy(source_dir)
        allowed_families = policy.get("allowed_licence_families") or []
        licence_policy_by_source[tag] = policy

        for row_number, row in enumerate(source_rows, start=1):
            event_id = row.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                errors.append(f"{source_dir.name}/events.jsonl row {row_number}: missing event_id")
                continue
            if event_id in seen_event_ids:
                errors.append(
                    f"{source_dir.name}/events.jsonl row {row_number}: event_id {event_id!r} "
                    f"already used by source {seen_event_ids[event_id]!r}"
                )
                continue
            seen_event_ids[event_id] = tag
            input_counts[tag] += 1

            record = _check_row_attribution(
                tag, row, row_number, source_dir, attribution_by_event, allowed_families, errors
            )
            row_media_entries = _check_row_media(row, row_number, source_dir, errors)

            provenance = row.get("provenance")
            if tag == "synthetic":
                # In-house generated rows carry no third-party verification
                # state; the dataset-level licence governs them.
                unverified = False
            else:
                unverified = _is_unverified(record) or _is_unverified(
                    provenance if isinstance(provenance, dict) else None
                )
            if unverified or _is_sharealike_licence(record):
                excluded = dict(record)
                excluded.setdefault("event_id", event_id)
                excluded["release_source"] = tag
                if unverified:
                    excluded.setdefault("verification_status", "unverified")
                    excluded["reason"] = (
                        "per-image licence verification is unverified; verify it at the "
                        "recorded landing page before release"
                    )
                else:
                    excluded["reason"] = (
                        "incoming CC BY-SA licence is incompatible with the "
                        f"{MAIN_DATASET_LICENCE} main release"
                    )
                excluded_records.append(excluded)
                continue

            if isinstance(record, dict):
                stored = dict(record)
                stored.setdefault("event_id", event_id)
                stored["release_source"] = tag
                stored["licence_partition"] = "main"
                attribution_records.append(stored)

            for entry in row_media_entries:
                entry["release_source"] = tag
                entry["licence_partition"] = "main"
                entry["origin_dir"] = str(source_dir)
                media_entries.append(entry)

            rows.append(row)
            tags.append(tag)

    if not rows:
        errors.append("no events were assembled from any source")

    if errors:
        raise ReleaseError(errors)

    group_count = _assign_grouped_splits(rows, tags, seed, val_fraction, test_fraction, errors)

    # Validate synthetic rows against the frozen event schema with the
    # existing validator (real-* rows follow realweak-event.schema.json and
    # are checked through their attribution/provenance record instead).
    if schema is not None:
        for tag, source_dir in sorted(sources.items()):
            subset = [row for row, row_tag in zip(rows, tags) if row_tag == tag]
            if tag != "synthetic" or not subset:
                continue
            schema_errors = validate_rows(subset, schema, source_dir)
            for error in schema_errors:
                errors.append(f"source {tag!r} failed event.schema.json validation: {error}")
    if errors:
        raise ReleaseError(errors)

    # Tag rows with their release source only after schema validation, so the
    # frozen event.schema.json (additionalProperties: false) still sees the
    # original row shape for synthetic sources.
    for row, tag in zip(rows, tags):
        row["release_source"] = tag
        row["licence_partition"] = "main"

    # From here on the release is committed: assemble stats, then write.
    per_source_counts = Counter(tags)
    excluded_by_source = Counter(record["release_source"] for record in excluded_records)
    excluded_by_licence = Counter(
        record["license_name"]
        for record in excluded_records
        if isinstance(record.get("license_name"), str) and record["license_name"]
    )
    split_counts = Counter(row["split"] for row in rows)
    split_by_source: dict[str, Counter] = {tag: Counter() for tag in per_source_counts}
    for row in rows:
        split_by_source[row["release_source"]][row["split"]] += 1
    sample_kind_counts = Counter(str(row.get("sample_kind")) for row in rows)

    def level_counts(field: str) -> dict[str, int]:
        counter: Counter = Counter(
            "null" if row.get(field) is None else str(row[field]) for row in rows
        )
        return dict(sorted(counter.items()))

    licence_counter: Counter = Counter()
    provider_counter: Counter = Counter()
    per_source_licence: dict[str, Counter] = defaultdict(Counter)
    per_source_provider: dict[str, Counter] = defaultdict(Counter)
    for record in attribution_records:
        tag = record["release_source"]
        licence = record.get("license_name")
        provider = record.get("provider")
        if isinstance(licence, str) and licence:
            licence_counter[licence] += 1
            per_source_licence[tag][licence] += 1
        if isinstance(provider, str) and provider:
            provider_counter[provider] += 1
            per_source_provider[tag][provider] += 1

    media_type_counts = Counter(str(entry.get("media_type")) for entry in media_entries)
    unique_group_values = {
        field: len({row.get(field) for row in rows if isinstance(row.get(field), str)})
        for field in GROUP_FIELDS
    }

    if copy_media:
        for row in rows:
            tag = row["release_source"]
            for media in row.get("media_files", []):
                if isinstance(media, dict) and isinstance(media.get("path"), str):
                    media["path"] = f"media/{tag}/{media['path'].lstrip('/')}"
        for entry in media_entries:
            entry["release_path"] = f"media/{entry['release_source']}/{entry['media_path'].lstrip('/')}"

    out.mkdir(parents=True, exist_ok=True)
    engine = _write_parquet(rows, out / "events.parquet")

    with (out / "media_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for entry in media_entries:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")

    source_counts = {
        "total_events": len(rows),
        "per_source": dict(sorted(per_source_counts.items())),
        "input_events": sum(input_counts.values()),
        "per_source_input": dict(sorted(input_counts.items())),
        "excluded_from_main": {
            "total_events": len(excluded_records),
            "per_source": dict(sorted(excluded_by_source.items())),
        },
        "per_source_split": {
            tag: dict(sorted(counter.items())) for tag, counter in sorted(split_by_source.items())
        },
    }
    (out / "source_counts.json").write_text(
        json.dumps(source_counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    licence_manifest = {
        "release_version": release_version,
        "dataset_licence": MAIN_DATASET_LICENCE,
        "policy": (
            "Fail closed. Only rows with complete attribution and, where the source "
            "directory ships source_licenses.json, an allowlisted redistribution "
            "licence enter the release. Rows without an explicit "
            "verification_status=verified record and "
            "CC BY-SA rows are excluded from this "
            "CC BY-NC-SA main release and must only be distributed in a separately "
            "licensed package."
        ),
        "per_source": {
            tag: {
                "events": per_source_counts[tag],
                "attributed_events": sum(per_source_licence[tag].values()),
                "licence_counts": dict(sorted(per_source_licence[tag].items())),
                "provider_counts": dict(sorted(per_source_provider[tag].items())),
                "allowed_licence_families": (
                    licence_policy_by_source.get(tag, {}).get("allowed_licence_families")
                    if licence_policy_by_source.get(tag)
                    else None
                ),
                "note": (
                    "in-house generated rows; dataset-level licence applies"
                    if tag == "synthetic" and not per_source_licence[tag]
                    else None
                ),
            }
            for tag in sorted(per_source_counts)
        },
        "licence_counts_total": dict(sorted(licence_counter.items())),
        "provider_counts_total": dict(sorted(provider_counter.items())),
        "excluded_from_main": {
            "total_events": len(excluded_records),
            "licence_counts": dict(sorted(excluded_by_licence.items())),
            "records": sorted(
                excluded_records,
                key=lambda record: (record["release_source"], str(record.get("event_id"))),
            ),
        },
        "attribution_records": sorted(
            attribution_records, key=lambda record: (record["release_source"], str(record.get("event_id")))
        ),
    }
    (out / "licence_manifest.json").write_text(
        json.dumps(licence_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    datasheet_stats = {
        "release_version": release_version,
        "schema_version": SCHEMA_VERSION,
        "total_events": len(rows),
        "input_events": sum(input_counts.values()),
        "per_source_counts": dict(sorted(per_source_counts.items())),
        "excluded_from_main": {
            "total_events": len(excluded_records),
            "per_source": dict(sorted(excluded_by_source.items())),
            "licence_counts": dict(sorted(excluded_by_licence.items())),
        },
        "split_counts": {split: split_counts.get(split, 0) for split in SPLITS},
        "split_counts_by_source": {
            tag: {split: split_by_source[tag].get(split, 0) for split in SPLITS}
            for tag in sorted(per_source_counts)
        },
        "sample_kind_counts": dict(sorted(sample_kind_counts.items())),
        "level_declared_counts": level_counts("level_declared"),
        "level_adjudicated_counts": level_counts("level_adjudicated"),
        "licence_counts": dict(sorted(licence_counter.items())),
        "provider_counts": dict(sorted(provider_counter.items())),
        "media_counts": {
            "total": len(media_entries),
            "by_type": dict(sorted(media_type_counts.items())),
        },
        "group_counts": {
            "connected_components": group_count,
            **{field: unique_group_values[field] for field in GROUP_FIELDS},
        },
        "grouped_split_fields": list(GROUP_FIELDS),
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "parquet_engine": engine,
        "synthetic_events_in_external_test": sum(
            1 for row in rows if row["release_source"] == "synthetic" and row["split"] == "external_test"
        ),
    }
    (out / "datasheet_stats.json").write_text(
        json.dumps(datasheet_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if copy_media:
        for entry in media_entries:
            destination = out / entry["release_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_file = Path(entry["origin_dir"]) / entry["media_path"]
            destination.write_bytes(source_file.read_bytes())

    return {
        "total_events": len(rows),
        "per_source_counts": dict(sorted(per_source_counts.items())),
        "split_counts": {split: split_counts.get(split, 0) for split in SPLITS},
        "licence_counts": dict(sorted(licence_counter.items())),
        "engine": engine,
        "group_count": group_count,
        "excluded_from_main": len(excluded_records),
        "out": out,
    }


def _parser() -> argparse.ArgumentParser:
    default_schema = Path(__file__).resolve().parents[1] / "schema" / "event.schema.json"
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Licence partition policy: this command builds only the CC BY-NC-SA 4.0 "
            "main release. Rows without an explicit verification_status=verified "
            "record are excluded. "
            "Any row attributed under CC BY-SA is excluded and recorded "
            "in licence_manifest.json; build a separately licensed BY-SA package if "
            "those rows are to be distributed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="TAG=PATH",
        dest="sources",
        help=(
            "input dataset directory with events.jsonl and media/, tagged "
            "synthetic / real-weak / real-crowd / real-tested (aliases: synth, "
            "realweak, crowd, tested); repeatable"
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory (created; must be empty)")
    parser.add_argument("--seed", type=int, default=13, help="deterministic seed for grouped split assignment")
    parser.add_argument(
        "--val-fraction", type=float, default=0.15, help="fraction of groups assigned to val (default 0.15)"
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="fraction of groups assigned to external_test (default 0.15)",
    )
    parser.add_argument("--release-version", default="unversioned", help="version string recorded in the manifests")
    parser.add_argument(
        "--copy-media",
        action="store_true",
        help="copy media into out/media/<tag>/... and rewrite the parquet paths accordingly",
    )
    parser.add_argument("--schema", type=Path, default=default_schema, help="event JSON Schema for synthetic rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        sources = _parse_sources(args.sources)
        summary = build_release(
            sources,
            args.out,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            release_version=args.release_version,
            copy_media=args.copy_media,
            schema_path=args.schema,
        )
    except ReleaseError as exc:
        for error in exc.errors:
            print(f"REFUSED: {error}", file=sys.stderr)
        print(f"REFUSED: release not written to {args.out}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sources_line = ", ".join(f"{tag}={count}" for tag, count in summary["per_source_counts"].items())
    splits_line = ", ".join(f"{split}={count}" for split, count in summary["split_counts"].items())
    licences_line = ", ".join(f"{name}={count}" for name, count in summary["licence_counts"].items())
    print(
        f"OK: {summary['total_events']} event(s) in {summary['group_count']} group(s) "
        f"-> {args.out} (parquet engine: {summary['engine']})"
    )
    print(f"sources: {sources_line}")
    print(f"splits: {splits_line}")
    if summary["excluded_from_main"]:
        print(f"excluded from CC BY-NC-SA main release: {summary['excluded_from_main']}")
    if licences_line:
        print(f"licences: {licences_line}")
    print(f"outputs: {', '.join(OUTPUT_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
