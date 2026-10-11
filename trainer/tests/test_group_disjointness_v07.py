"""H-GROUP verification against the SHIPPED v0.7 manifest, not a fixture.

The H-GROUP hazard (`hf/HAZARD_LOG.md`) is that the configured leak control
(`data.group_field: group_id`) may not separate anything. These tests turn the
claim into a gate that runs on the real manifest:

  1. no `group_id` appears in more than one split  (the hazard's own claim,
     checked twice: through the shipped `validate_manifest` gate and independently)
  2. no near-duplicate IMAGE pair (dHash<=6 AND pHash<=8, recomputed from the
     JPEGs on disk) straddles a split boundary
  3. no dish identity and no source page straddles a split boundary -- the two
     coarser units H-GROUP says a group should have been built from
  4. `event_id` is unique, so it can anchor a key
  5. the `group_size` field's disagreement with the manifest is PINNED, so the
     known staleness stays visible instead of being silently forgotten

Everything skips (never fails) when the manifest or the corpus event files are
not mounted, so this runs in environments that ship code without data.

Measurements behind these tests: lane_runs/an04_groups/report.json and
lane_runs/an04_groups/report_followups.json.
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import re
from pathlib import Path

import pytest

from train.data import load_manifest, validate_manifest

_LANE_ROOT = Path(__file__).resolve().parents[2]  # .../lanes/an04_groups
_DATA_ROOT = _LANE_ROOT.parent.parent / "data"    # .../iddsi/data
MANIFEST_CANDIDATES = [
    os.environ.get("IDDSI_V07_MANIFEST"),
    "/work/data/manifest_v07_weighted.jsonl",
    str(_DATA_ROOT / "manifest_v07_weighted.jsonl"),
]
CORPUS_EVENTS = [
    "/work/data/harvest_d1/events.jsonl",
    "/work/data/harvest_d1b/events.jsonl",
    "/work/data/realweak_pilot/events.jsonl",
]
# v0.3's dedup criterion, reused verbatim so the test measures the same thing
# the split was built with (regroup_dedup.py: dHash<=6 AND pHash<=8).
DHASH_MAX = 6
PHASH_MAX = 8


def _manifest_path() -> Path:
    for candidate in MANIFEST_CANDIDATES:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    pytest.skip("v0.7 manifest not mounted; no data-side claim is made here")


def _rows() -> list[dict]:
    return load_manifest(_manifest_path())


def _data_root() -> Path:
    return _DATA_ROOT


def _host(path: str) -> str:
    """Manifest paths are container paths (/work/...); use them as-is when they
    exist (container), otherwise map /work to this checkout's data root."""
    if Path(path).exists():
        return path
    return path.replace("/work/", str(_data_root()) + "/", 1)


def _meta() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for candidate in CORPUS_EVENTS:
        p = Path(candidate) if Path(candidate).exists() else Path(_host(candidate))
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                out[event["event_id"]] = event
    return out


def _dhash(path: str, size: int = 8) -> int:
    import numpy as np
    from PIL import Image

    grey = Image.open(_host(path)).convert("L").resize((size + 1, size), Image.LANCZOS)
    array = np.asarray(grey, dtype="int16")
    return int("".join("1" if b else "0" for b in (array[:, :-1] > array[:, 1:]).flatten()), 2)


def _phash(path: str, size: int = 8) -> int:
    import numpy as np
    from PIL import Image

    array = np.asarray(
        Image.open(_host(path)).convert("L").resize((size, size), Image.LANCZOS), dtype=float
    )
    n = np.arange(size)
    basis = np.cos(np.pi * (2 * n + 1) * n.reshape(-1, 1) / (2 * size)) * np.sqrt(2.0 / size)
    basis[0] /= np.sqrt(2.0)
    dct = basis @ array @ basis.T
    return int(
        "".join("1" if b else "0" for b in (dct > np.median(dct.ravel()[1:])).flatten()), 2
    )


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def test_v07_group_disjoint_through_shipped_gate():
    """validate_manifest raises on group leakage; it must not raise here."""
    rows = _rows()
    validate_manifest(rows)  # raises ValueError("group leakage ...") on a violation


def test_v07_no_group_id_spans_two_splits():
    rows = _rows()
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        groups[str(row["group_id"])].add(str(row["split"]))
    spanning = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    assert not spanning, f"group_id appears in more than one split: {spaning}"
    # The claim is only meaningful if a group can hold more than one row.
    assert len(groups) > 0


def test_v07_no_near_duplicate_image_pair_across_splits():
    """Recompute dHash/pHash from the JPEGs and assert no near-duplicate pair
    (the v0.3 criterion) straddles a split boundary."""
    rows = _rows()
    split_of = {r["event_id"]: r["split"] for r in rows}
    hashes = {r["event_id"]: (_dhash(r["image_path"]), _phash(r["image_path"])) for r in rows}
    cross = []
    for a, b in itertools.combinations(list(hashes), 2):
        if split_of[a] == split_of[b]:
            continue
        if _hamming(hashes[a][0], hashes[b][0]) <= DHASH_MAX and (
            _hamming(hashes[a][1], hashes[b][1]) <= PHASH_MAX
        ):
            cross.append((a, b))
    assert not cross, f"near-duplicate images across splits: {cross[:10]}"


def test_v07_no_dish_identity_or_source_page_across_splits():
    """H-GROUP's fix asks groups to be dish identity / source page. Assert that
    neither of those units straddles a split, which is the part the group key
    is supposed to guarantee but does not."""
    rows = _rows()
    meta = _meta()
    if not meta:
        pytest.skip("corpus events.jsonl not mounted")
    split_of = {r["event_id"]: r["split"] for r in rows}
    dish: dict[str, list[str]] = collections.defaultdict(list)
    page: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        event = meta.get(row["event_id"])
        if event is None:
            continue
        prov = event.get("provenance") or {}
        for field in ("source_id", "download_url"):
            found = re.search(r"dish_(\d+)", prov.get(field) or "")
            if found:
                dish["dish_" + found.group(1)].append(row["event_id"])
                break
        url = (prov.get("download_url") or "").split("?")[0].strip().rstrip("/")
        if url:
            page[url].append(row["event_id"])

    def spanning(mapping):
        return {k: sorted(v) for k, v in mapping.items()
                if len({split_of[e] for e in v}) > 1}

    assert not spanning(dish), f"dish identity spans splits: {spaning(dish)}"
    assert not spanning(page), f"source page spans splits: {spaning(page)}"


def test_v07_event_id_is_unique():
    rows = _rows()
    counts = collections.Counter(r["event_id"] for r in rows)
    duplicated = {k: v for k, v in counts.items() if v > 1}
    assert not duplicated, f"event_id is not a unique key: {duplicated}"


def test_v07_group_size_field_mismatch_is_pinned_not_silenced():
    """KNOWN DEFECT, pinned so it cannot drift unnoticed: `group_size` still
    describes the pre-v0.5 population, so it over-states the group for rows
    whose group lost members when the manifest was cut from 684 to 532 rows.
    When the field is corrected this test must be updated in the same commit;
    until then the count is asserted so the staleness stays on the record.
    No split-disjointness consequence: every dropped member came from the same
    split its group still sits in."""
    rows = _rows()
    per_group: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        per_group[str(row["group_id"])].append(row)
    mismatched = {
        g: (r[0]["group_size"], len(r))
        for g, r in per_group.items()
        if r[0].get("group_size") != len(r)
    }
    assert len(mismatched) == 3, f"expected the 3 known stale groups, got {mismatched}"
    affected = sum(len(r) for g, r in per_group.items() if g in mismatched)
    assert affected == 18, f"expected 18 rows in those 3 groups, got {affected}"
