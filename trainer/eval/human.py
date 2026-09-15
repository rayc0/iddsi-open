from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score


def load_human_ratings(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"event_id", "rater_a_level", "rater_b_level"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"human ratings CSV needs columns {sorted(required)}")
    return rows


def human_inter_rater_baseline(
    rows: list[dict[str, str]],
    event_ids: list[str] | np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    first = np.asarray([int(row["rater_a_level"]) for row in rows])
    second = np.asarray([int(row["rater_b_level"]) for row in rows])
    if not np.isin(first, range(3, 8)).all() or not np.isin(second, range(3, 8)).all():
        raise ValueError("human rating levels must be in L3-L7")
    result: dict[str, Any] = {
        "n": len(rows),
        "raw_agreement": float((first == second).mean()),
        "weighted_kappa": float(cohen_kappa_score(first, second, weights="quadratic")),
        "note": "Human baseline is inter-rater agreement, not model accuracy and not a safety benchmark.",
    }
    if event_ids is not None and reference_labels is not None:
        references = {
            str(event_id): int(label) + 3 for event_id, label in zip(event_ids, np.asarray(reference_labels))
        }
        matched = [index for index, row in enumerate(rows) if row["event_id"] in references]
        if not matched:
            raise ValueError("no human rating event_id matches the prediction file")
        truth = np.asarray([references[rows[index]["event_id"]] for index in matched])
        result["matched_to_physical_test"] = len(matched)
        result["rater_vs_physical_test"] = {}
        for name, ratings in (("rater_a", first[matched]), ("rater_b", second[matched])):
            result["rater_vs_physical_test"][name] = {
                "raw_agreement": float((ratings == truth).mean()),
                "weighted_kappa": float(cohen_kappa_score(truth, ratings, weights="quadratic")),
                "dangerous_underclassification_fnr": float((ratings < truth).mean()),
            }
        result["note"] = (
            "Human photo-only ratings are matched by event_id to physical-test labels; this is not a safety benchmark."
        )
    return result
