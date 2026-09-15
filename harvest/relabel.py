"""Relabel deferred real-weak events with a VLM (GPU pass).

Reads a dataset directory whose ``events.jsonl`` rows were written with
``weak_label_record.rule == "deferred"`` (see :class:`harvest.label.DeferLabeler`),
runs a labeler over each referenced media file, and rewrites ``events.jsonl``
in place. The rewrite is atomic (temp file + ``os.replace``) and a timestamped
backup of the original manifest is kept next to it. ``attribution.jsonl`` and
all media files are left untouched.

Safety rules:
- only rows whose ``weak_label_record.rule`` is ``deferred`` are modified;
- updated rows are re-validated against the real-weak schema extension before
  anything is written; any validation failure aborts the whole run without
  touching the original manifest;
- if the VLM flags a face or legible text in a previously accepted image, the
  row is left deferred and counted as ``vlm_flagged`` (never silently kept,
  never deleted here — takedown is a separate audited process).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .image_ops import decode_image
from .label import Qwen3VLLabeler
from .manifest import read_jsonl, utc_now, validate_realweak_event


DEFERRED_RULE = "deferred"


@dataclass(frozen=True)
class RelabelSummary:
    total: int
    deferred: int
    relabelled: int
    vlm_flagged: int
    already_labelled: int
    backup_path: str
    events_path: str


def is_deferred(event: dict[str, Any]) -> bool:
    record = event.get("weak_label_record")
    return isinstance(record, dict) and record.get("rule") == DEFERRED_RULE


def apply_weak_label(event: dict[str, Any], weak: Any) -> dict[str, Any]:
    """Return a copy of ``event`` with weak-label fields replaced by ``weak``.

    Mirrors the field wiring in :func:`harvest.manifest.build_event` so a
    relabelled row is indistinguishable from a row labelled at harvest time.
    """
    updated = dict(event)
    updated["level_weak"] = weak.level_weak
    updated["level_declared"] = weak.level_weak  # extension mirror, not a cook declaration
    updated["weak_confidence"] = weak.confidence
    updated["sample_kind"] = "liquid" if weak.level_relation == "at_most" else "food"
    updated["weak_label_record"] = {
        "model_id": weak.model_id,
        "prompt_version": weak.prompt_version,
        "rule": weak.rule,
        "level_relation": weak.level_relation,
        "observations": weak.observations,
        "rationale": weak.rationale,
        "is_ground_truth": False,
    }
    privacy = dict(event.get("privacy_screening") or {})
    privacy["vlm_face_check"] = bool(weak.contains_face)
    privacy["vlm_text_check"] = bool(weak.contains_text)
    updated["privacy_screening"] = privacy
    return updated


def _resolve_media(dataset: Path, event: dict[str, Any]) -> Path:
    media_files = event.get("media_files") or []
    if not media_files or not isinstance(media_files[0], dict):
        raise ValueError(f"event {event.get('event_id')} has no media_files entry")
    relative = media_files[0].get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"event {event.get('event_id')} has no media path")
    media_path = (dataset / relative).resolve()
    try:
        media_path.relative_to(dataset.resolve())
    except ValueError as exc:
        raise ValueError(f"media path escapes dataset root: {relative}") from exc
    if not media_path.is_file():
        raise ValueError(f"manifest media is missing: {relative}")
    return media_path


def relabel_dataset(dataset: Path, labeler: Any) -> RelabelSummary:
    dataset = dataset.resolve()
    events_path = dataset / "events.jsonl"
    if not events_path.is_file():
        raise ValueError(f"dataset has no events.jsonl: {dataset}")
    base_schema = Path(__file__).resolve().parent.parent / "dataschema" / "schema" / "event.schema.json"

    rows = read_jsonl(events_path)
    deferred_count = sum(1 for row in rows if is_deferred(row))

    relabelled = 0
    vlm_flagged = 0
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        if not is_deferred(row):
            new_rows.append(row)
            continue
        media_path = _resolve_media(dataset, row)
        image = decode_image(media_path.read_bytes())
        weak = labeler.label(image)
        if weak.contains_face or weak.contains_text:
            # Fail closed: keep the row deferred, surface the count. Media and
            # attribution stay untouched; removal is a separate audited step.
            vlm_flagged += 1
            new_rows.append(row)
            continue
        updated = apply_weak_label(row, weak)
        errors = validate_realweak_event(updated, base_schema)
        if errors:
            raise ValueError(
                f"relabelled row {row.get('event_id')} failed validation: " + "; ".join(errors)
            )
        relabelled += 1
        new_rows.append(updated)

    if relabelled == 0:
        return RelabelSummary(
            total=len(rows),
            deferred=deferred_count,
            relabelled=0,
            vlm_flagged=vlm_flagged,
            already_labelled=len(rows) - deferred_count,
            backup_path="",
            events_path=str(events_path),
        )

    timestamp = utc_now().replace("-", "").replace(":", "").replace(".", "").rstrip("Z") + "Z"
    backup_path = events_path.with_name(f"{events_path.name}.bak-{timestamp}")
    shutil.copyfile(events_path, backup_path)

    temporary = events_path.with_name(f"{events_path.name}.relab.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, events_path)

    return RelabelSummary(
        total=len(rows),
        deferred=deferred_count,
        relabelled=relabelled,
        vlm_flagged=vlm_flagged,
        already_labelled=len(rows) - deferred_count,
        backup_path=str(backup_path),
        events_path=str(events_path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relabel deferred real-weak events with the Qwen3-VL weak labeler."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory with events.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    labeler = Qwen3VLLabeler(args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    try:
        summary = relabel_dataset(args.dataset, labeler)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
