"""def04_manifest — the three manifest defects, reproduced and fixed.

Companion to `test_group_disjointness_v07.py` (which pins the SHIPPED
manifest's defects). This file does two things the existing gate does not:

  1. REPRODUCES each of the three defects an04_groups recorded against the
     shipped files, so none is silenced:
       D1  group_size stale for 3 groups / 18 rows
       D2  realweak_pilot/events.jsonl has 12 duplicated event rows, one
           conflicting on level_weak (5 vs 6)
       D3  the recorded phash — measured BOTH ways. Under the algorithm that
           PRODUCED it (harvest.image_ops.perceptual_hash) it reproduces for
           ~513/532; under the 8x8 DCT an04 recomputed with, only ~4/532. The
           an04 "not reproducible" finding was an algorithm-mismatch artifact.
  2. VALIDATES the derived artifacts under lane_runs/def04_manifest/ fix them
     (D1 corrected, D2 deduplicated + conflict annotated, D3 carries a
     validated phash).

Every test skips (never fails) when its inputs are not mounted, so a code-only
checkout stays green. Reproduction: tools/def04_reproduce.py, tools/def04_fix.py,
tools/def04_phash_check.py, tools/def04_verify.py; numbers in
lane_runs/def04_manifest/{reproduce_report.json,def04_fix_report.json}.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import pytest

sys_path = os.environ.get(
    "IDDSI_LANE_PATH", str(Path(__file__).resolve().parents[2])
)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from harvest.image_ops import perceptual_hash, hamming_distance  # noqa: E402

_LANE_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _LANE_ROOT.parent.parent / "data"
_DERIVED = Path(os.environ.get("IDDSI_RUNS_PATH", "."))
if not _DERIVED.exists():
    _DERIVED = _LANE_ROOT.parent.parent / "lane_runs" / "def04_manifest"

CONFLICT_EVENT = "RW_nutrition5k_8e68a80fd452f62af95f"


def _ship(path: str) -> Path:
    p = Path(path)
    return p if p.exists() else Path(str(_DATA_ROOT / Path(path).name))


def _shipped_manifest() -> Path:
    for c in [os.environ.get("IDDSI_V07_MANIFEST"),
              str(_data_root() / "data/manifest_v07_weighted.jsonl"),
              _DATA_ROOT / "manifest_v07_weighted.jsonl"]:
        if c and Path(c).exists():
            return Path(c)
    pytest.skip("v0.7 manifest not mounted")


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _realweak_events() -> Path:
    for c in [str(_data_root() / "data/realweak_pilot/events.jsonl"),
              _DATA_ROOT / "realweak_pilot" / "events.jsonl"]:
        if Path(c).exists():
            return Path(c)
    pytest.skip("realweak_pilot/events.jsonl not mounted")


def _host(p: str) -> str:
    return p if Path(p).exists() else p.replace("/work/", str(_data_root()) + "/", 1)


def _data_root() -> Path:
    return _DATA_ROOT


def _phash_8x8(path: str) -> int:
    """The non-standard 8x8 DCT an04 recomputed with (verify_group_disjointness.py)."""
    import numpy as np
    from PIL import Image
    size = 8
    arr = np.asarray(Image.open(_host(path)).convert("L").resize((size, size), Image.LANCZOS),
                     dtype=float)
    n = np.arange(size)
    basis = np.cos(np.pi * (2 * n + 1) * n.reshape(-1, 1) / (2 * size)) * np.sqrt(2.0 / size)
    basis[0] /= np.sqrt(2.0)
    dct = basis @ arr @ basis.T
    return int("".join("1" if b else "0" for b in (dct > np.median(dct.ravel()[1:])).flatten()), 2)


# ---------------------------------------------------------------------------
# D1 — group_size stale (reproduce) + derived fix
# ---------------------------------------------------------------------------
def test_def04_d1_shipped_group_size_stale_is_reproduced():
    rows = _rows(_shipped_manifest())
    per_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        per_group[str(r["group_id"])].append(r)
    mismatched = {g for g, grp in per_group.items() if grp[0].get("group_size") != len(grp)}
    affected = sum(len(per_group[g]) for g in mismatched)
    # the defect an04_groups recorded, kept on the record (also pinned by
    # test_group_disjointness_v07.py::test_v07_group_size_field_mismatch_is_pinned_not_silenced)
    assert len(mismatched) == 3 and affected == 18, (mismatched, affected)


def test_def04_d1_derived_group_size_is_consistent():
    dm = _DERIVED / "manifest_v07_weighted.def04.jsonl"
    if not dm.exists():
        pytest.skip("derived manifest not present")
    rows = _rows(dm)
    cnt = collections.Counter(str(r["group_id"]) for r in rows)
    bad = [r["event_id"] for r in rows if r["group_size"] != cnt[str(r["group_id"])]]
    assert not bad, f"derived group_size still stale for {bad}"


# ---------------------------------------------------------------------------
# D2 — duplicated event rows (reproduce) + derived dedup
# ---------------------------------------------------------------------------
def test_def04_d2_shipped_realweak_has_12_dups_one_conflict():
    recs = _rows(_realweak_events())
    by = collections.Counter(str(r["event_id"]) for r in recs)
    dup_rows = sum(n - 1 for n in by.values() if n > 1)
    conflicts = []
    for eid, n in by.items():
        if n > 1:
            copies = [r for r in recs if str(r["event_id"]) == eid]
            if len({c.get("level_weak") for c in copies}) > 1:
                conflicts.append(eid)
    assert len(recs) == 129 and len(by) == 117
    assert dup_rows == 12
    assert conflicts == [CONFLICT_EVENT]


def test_def04_d2_derived_events_deduped_and_conflict_annotated():
    ev = _DERIVED / "realweak_pilot.events.def04.jsonl"
    if not ev.exists():
        pytest.skip("derived events not present")
    recs = _rows(ev)
    by = collections.Counter(str(r["event_id"]) for r in recs)
    assert len(recs) == 117 and all(n == 1 for n in by.values())
    conf = [r for r in recs if str(r["event_id"]) == CONFLICT_EVENT][0]
    assert conf["level_weak"] == 6  # first occurrence, matches manifest level_tier1=6
    assert "def04_source_event_conflict" in conf  # annotated, not silenced


# ---------------------------------------------------------------------------
# D3 — recorded phash: reproducible under the producing algorithm
# ---------------------------------------------------------------------------
def test_def04_d3_recorded_phash_reproduces_under_producing_algorithm():
    rows = _rows(_shipped_manifest())
    # recorded phash (first occurrence) from the corpus events files
    recorded: dict[str, str] = {}
    for name in ("realweak_pilot", "harvest_d1", "harvest_d1b"):
        p = _DATA_ROOT / name / "events.jsonl"
        if not p.exists():
            pytest.skip("corpus events not mounted")
        for r in _rows(p):
            recorded.setdefault(str(r["event_id"]), str(r.get("phash")))

    harvest_exact = an04_exact = 0
    checked = 0
    from PIL import Image
    for r in rows:
        eid = str(r["event_id"])
        img = _host(str(r["image_path"]))
        if not Path(img).exists():
            continue
        rec = recorded.get(eid)
        if rec is None:
            continue
        checked += 1
        if hamming_distance(rec, perceptual_hash(Image.open(img))) == 0:
            harvest_exact += 1
        if hamming_distance(rec, f"{_phash_8x8(img):016x}") == 0:
            an04_exact += 1
    if checked == 0:
        pytest.skip("no images mounted")
    # Under the algorithm that produced the phash, it reproduces (measured 513/532).
    assert harvest_exact >= 500, (harvest_exact, checked)
    # Under the 8x8 DCT an04 recomputed with, it does not (measured 4/532) — that
    # mismatch is the algorithm, not corruption of the recorded field.
    assert an04_exact <= 10, an04_exact


def test_def04_d3_derived_manifest_carries_validated_phash():
    dm = _DERIVED / "manifest_v07_weighted.def04.jsonl"
    if not dm.exists():
        pytest.skip("derived manifest not present")
    rows = _rows(dm)
    reproduces = sum(1 for r in rows if r.get("phash_reproduces") is True)
    assert reproduces >= 500, reproduces
