"""Human inter-rater baseline on a dataschema manifest.

Reads two rater label columns from an IDDSI-Open dataschema manifest
(JSONL, or Parquet when pyarrow is installed) and reports quadratic-weighted
Cohen's kappa, raw agreement, and the full rater-vs-rater confusion matrix.

Rater columns come from ``dataschema/schema/event.schema.json``:
``level_tested_RD``, ``level_tested_SLP``, and ``level_adjudicated`` are all
``nullable_level`` fields (integer 0..7 or null). A null rater value means the
rater abstained / did not rate that event; such rows are counted and reported
separately and are excluded from the kappa/agreement computation.

CLI::

    python -m trainer.eval.inter_rater <manifest> --rater-a level_tested_RD \
        --rater-b level_tested_SLP

Run from the repo root (``trainer`` resolves as a namespace package) or from
``trainer/`` with ``python -m eval.inter_rater``.

This is an inter-rater agreement baseline only. It is not model accuracy and
not a safety benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

#: Full IDDSI level range allowed by the schema (``nullable_level``: 0..7).
ALL_LEVELS = list(range(0, 8))


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a dataschema manifest into a list of row dicts.

    Supports the JSONL manifest (``events.jsonl``, dependency-free) and
    Parquet (a single file or a partitioned dataset directory) when pyarrow
    is importable, mirroring ``dataschema/tools/validate_dataset.py``.
    """

    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if manifest.suffix == ".jsonl" or manifest.is_file() and manifest.suffix == "":
        rows: list[dict[str, Any]] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{manifest}:{line_number}: invalid JSON line: {error}") from error
        if not rows:
            raise ValueError(f"{manifest}: manifest contains no rows")
        return rows
    try:
        import pyarrow.dataset as ds  # noqa: PLC0415 - optional dependency
    except ImportError as error:
        raise RuntimeError(
            "Parquet input requires pyarrow; use the JSONL manifest or pip install pyarrow"
        ) from error
    dataset = ds.dataset(str(manifest), format="parquet", partitioning="hive" if manifest.is_dir() else None)
    return list(dataset.to_table().to_pylist())


def _rater_values(rows: list[dict[str, Any]], column: str) -> list[int | None]:
    if not rows:
        raise ValueError("manifest contains no rows")
    if column not in rows[0]:
        raise ValueError(
            f"rater column {column!r} not found; schema rater columns are "
            "'level_tested_RD', 'level_tested_SLP', 'level_adjudicated'"
        )
    values: list[int | None] = []
    for index, row in enumerate(rows):
        raw = row.get(column)
        if raw is None:
            values.append(None)
            continue
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"row {index}: {column!r} must be an integer level 0..7 or null, got {raw!r}")
        if not 0 <= raw <= 7:
            raise ValueError(f"row {index}: {column!r} must be an integer level 0..7 or null, got {raw!r}")
        values.append(raw)
    return values


def inter_rater_baseline(
    manifest: str | Path,
    rater_a: str = "level_tested_RD",
    rater_b: str = "level_tested_SLP",
) -> dict[str, Any]:
    """Compute the inter-rater baseline between two manifest label columns."""

    rows = load_manifest_rows(manifest)
    if not rows[0].get("event_id"):
        raise ValueError("manifest rows must carry an 'event_id' field (dataschema event schema)")
    first = _rater_values(rows, rater_a)
    second = _rater_values(rows, rater_b)

    both = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    a_abstained = [i for i, (a, b) in enumerate(zip(first, second)) if a is None and b is not None]
    b_abstained = [i for i, (a, b) in enumerate(zip(first, second)) if a is not None and b is None]
    both_abstained = [i for i, (a, b) in enumerate(zip(first, second)) if a is None and b is None]

    result: dict[str, Any] = {
        "manifest": str(manifest),
        "rater_a_column": rater_a,
        "rater_b_column": rater_b,
        "n_rows": len(rows),
        "n_both_rated": len(both),
        "abstentions": {
            "rater_a_abstained": len(a_abstained),
            "rater_b_abstained": len(b_abstained),
            "both_abstained": len(both_abstained),
            "either_abstained": len(a_abstained) + len(b_abstained) + len(both_abstained),
            "rater_a_abstained_event_ids": [rows[i]["event_id"] for i in a_abstained],
            "rater_b_abstained_event_ids": [rows[i]["event_id"] for i in b_abstained],
            "both_abstained_event_ids": [rows[i]["event_id"] for i in both_abstained],
        },
        "note": "Rows where either rater abstained (null) are excluded from kappa/agreement and reported above.",
    }

    if not both:
        result["weighted_kappa"] = None
        result["raw_agreement"] = None
        result["confusion_matrix"] = None
        result["confusion_labels"] = ALL_LEVELS
        result["warning"] = "no events rated by both raters; kappa and agreement are undefined"
        return result

    a_arr = np.asarray([a for a, _ in both], dtype=np.int64)
    b_arr = np.asarray([b for _, b in both], dtype=np.int64)
    result["raw_agreement"] = float((a_arr == b_arr).mean())
    result["weighted_kappa"] = float(cohen_kappa_score(a_arr, b_arr, weights="quadratic"))
    labels = np.asarray(ALL_LEVELS, dtype=np.int64)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a_val, b_val in zip(a_arr, b_arr):
        matrix[int(np.searchsorted(labels, a_val)), int(np.searchsorted(labels, b_val))] += 1
    result["confusion_matrix"] = matrix.tolist()
    result["confusion_labels"] = ALL_LEVELS
    result["confusion_orientation"] = "rows = rater_a level, columns = rater_b level"
    result["rater_a_marginal_counts"] = {str(level): int((a_arr == level).sum()) for level in ALL_LEVELS}
    result["rater_b_marginal_counts"] = {str(level): int((b_arr == level).sum()) for level in ALL_LEVELS}
    result["agreement_by_rater_a_level"] = {
        str(level): {
            "n": int((a_arr == level).sum()),
            "agree": int(((a_arr == level) & (a_arr == b_arr)).sum()),
        }
        for level in ALL_LEVELS
        if (a_arr == level).any()
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human inter-rater baseline (quadratic-weighted kappa + confusion) on a dataschema manifest"
    )
    parser.add_argument("manifest", type=Path, help="dataschema manifest (events.jsonl or Parquet file/dir)")
    parser.add_argument("--rater-a", default="level_tested_RD", help="first rater column (default: level_tested_RD)")
    parser.add_argument("--rater-b", default="level_tested_SLP", help="second rater column (default: level_tested_SLP)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = inter_rater_baseline(args.manifest, args.rater_a, args.rater_b)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
