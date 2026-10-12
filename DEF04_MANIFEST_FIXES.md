# def04_manifest — the three manifest defects reproduced and resolved

Lane `def04_manifest`, 2026-10-12, nonce `6cd83dc32b53`. Works alone.

Task: fix the three manifest defects `an04_groups` recorded (`hf/HAZARD_LOG.md`
H-GROUP measured status; `HGROUP_VERIFICATION.md` §D;
`lane_runs/an04_groups/report*.json`), none of which may be silenced. For each:
reproduce it, state the correct resolution, and either fix it in a NEW derived
manifest (never edit the shipped one in place) or explain why it cannot be fixed
without Raymond. **The shipped files are byte-identical before and after this
run** — nothing shipped was edited.

Shipped files touched read-only (md5 verified unchanged by `tools/def04_fix.py`):

| file | md5 |
|---|---|
| `data/manifest_v07_weighted.jsonl` | `520d2591bd0f8a2489bf3c0a3255dcd5` |
| `data/realweak_pilot/events.jsonl` | `d7e850c334bd3afab81795401f71538a` |
| `data/harvest_d1/events.jsonl` | `dd184ec647f9607a1a3e0ef90f96df92` |
| `data/harvest_d1b/events.jsonl` | `46f8247c0686c63da835dd3e26696bbd` |

## Reproduce

```
python3 -I tools/def04_reproduce.py  --manifest .../manifest_v07_weighted.jsonl \
    --events .../realweak_pilot/events.jsonl .../harvest_d1/events.jsonl \
             .../harvest_d1b/events.jsonl \
    --outdir /work/lane_runs/def04_manifest         # -> reproduce_report.json
python3 -I tools/def04_phash_check.py               # D3, both algorithms, host+container
```

All numbers below are from `lane_runs/def04_manifest/reproduce_report.json` unless
noted. No GPU was used; every check is CPU-only. No sealed file was read.

## D1 — `group_size` is stale for 3 groups / 18 rows — FIXED

**Reproduced** (`reproduce_report.json:D1_group_size_stale`): 501 groups over 532
rows, histogram `{1:495, 2:1, 4:2, 8:1, 9:1, 10:1}`; **3 groups disagree** with
their `group_size` field, **18 rows** affected, all in `train`:

| group | declared `group_size` | observed rows | split |
|---|---|---|---|
| g0010 | 9 | 8 | train |
| g0020 | 2 | 1 | train |
| g0060 | 10 | 9 | train |

Root cause: the field describes the 684-row pre-v0.5 population; each of these
three groups lost one member when the corpus was cut 684→532 (the member was one
of the 152 unusable events, H-UNUSABLE).

**Correct resolution.** `group_size` should equal the within-manifest cardinality
— the count of rows sharing that `group_id`. Deterministic; no adjudication. It
is consumed by **no** trainer code (`trainer/train/data.py:validate_manifest`
reads only `group_id`; `group_size` appears only in the an04 pin test), so the
correction has **no** training-semantics or split effect.

**Fixed** in the derived manifest `manifest_v07_weighted.def04.jsonl`
(`lane_runs/def04_manifest/`): `group_size` recomputed to observed cardinality
(0 mismatches); the 18 changed rows carry `group_size_def04_corrected=true`. The
shipped manifest still carries the defect and its an04 pin
(`test_group_disjointness_v07.py`) is left intact and passing.

## D2 — `realweak_pilot/events.jsonl` carries 12 duplicated event rows — STRUCTURALLY FIXED; one label conflict flagged for Raymond

**Reproduced** (`reproduce_report.json:D2_duplicate_events`): 129 rows, 117
distinct `event_id`, **12 duplicate rows**.

**Correction to the earlier reading.** an04 recorded "11 byte-identical
duplicates; one conflicts on `level_weak`". The 12 are **not** 11 byte-identical:
every duplicate differs in raw bytes on harvest metadata (`captured_at`,
`privacy_screening` engine, `provenance` timestamps, `weak_confidence`). But on
the **substantive** fields (`level_weak`, `phash`, `download_url`, `recipe_id`,
`media` path/sha/bytes) **11 agree** (pure-metadata duplicates) and **exactly 1
conflicts**. The substantive conclusion an04 reached is unchanged; the
"byte-identical" description is corrected.

The one conflict: `RW_nutrition5k_8e68a80fd452f62af95f` (lines 58 and 123) —
`level_weak` **5 vs 6**, with identical `phash` (`c4db337668581879`), identical
`weak_confidence` (0.75), same model (Qwen3-VL-2B). Line 58: L6 candidate,
~15 mm pieces; line 123: L5 candidate, ~4 mm particles. Two harvest runs of the
same image disagreeing on an L5/L6 interior case — genuine tier-1 instability,
the H-NOISE regime.

**Correct resolution.**
- Deduplicate to one row per `event_id` (129→117), first-occurrence rule — the
  rule `trainer/train/build_manifest.py` already uses (`if record.event_id in
  seen_event_ids: continue`). For the 11 pure-metadata duplicates the label
  fields are identical, so the choice is immaterial to any label.
- The single `level_weak` 5-vs-6 conflict has **no data-driven tie-break**
  (equal confidence, same model, same image). Authoritative adjudication
  **requires Raymond** (or a physical re-test / higher-tier re-label). It is
  **non-blocking**: the shipped manifest's training label for this event is the
  consensus `level=7` (`label_source=consensus`, `votes={step5:7}`), which does
  not depend on `level_weak`; `level_tier1=6` (the field the conflict lives in)
  is tier-1 provenance, never eval ground truth, and read by no gate.

**Fixed** in the derived events file `realweak_pilot.events.def04.jsonl`
(`lane_runs/def04_manifest/`): 117 rows, 0 duplicates; the conflict is kept at
`level_weak=6` (first occurrence, consistent with the manifest's
`level_tier1=6`) and **annotated** with `def04_source_event_conflict`
(`values [5,6]`, kept 6, resolution note) — surfaced, not silenced. The derived
manifest carries the same annotation on that one row.

## D3 — recorded `phash` "not reproducible" — CORRECTED; the field is KEPT

**Reproduced both ways** (`tools/def04_phash_check.py`, identical on host PIL
10.2.0 and container PIL 11.1.0):

| recompute algorithm | exact / 532 | mean Hamming (all) | range when ≠ |
|---|---|---|---|
| an04's 8×8 hand-rolled DCT (`verify_group_disjointness.py:_phash`) | **4** | 4.37 | 1–11 |
| `harvest/image_ops.py:perceptual_hash` (32×32 imagehash pHash, **the producing code**) | **513** | 0.07 | 1–2 |

**The recorded phash is not corrupt.** It reproduces **513/532 exactly** under
`harvest/image_ops.py:perceptual_hash` — the conventional 32×32 imagehash pHash
that `harvest/pipeline.py` and `harvest/manifest.py` actually used to produce the
stored values — with the other 19 off by only 1–2 of 64 bits (18 by 2, 1 by 1),
all in the open-images/nutrition5k/openverse corpora. an04's "4 of 532 exact,
mean 4.4" was measured by recomputing with a **different, non-standard 8×8 DCT**,
so it compared the recorded phash against a different quantity. The
irreproducibility was an **algorithm-mismatch artifact**, not data corruption.
The result is PIL-version-robust (identical host 10.2.0 / container 11.1.0).

**Correct resolution.** Keep the recorded phash; standardise any
recorded-vs-recomputed comparison on the producing algorithm
(`harvest/image_ops.py:perceptual_hash`). The 19 borderline cases are at the
pHash noise floor and immaterial at the ≤8 near-duplicate threshold. The
near-duplicate **gate conclusion is unchanged and holds under both algorithms**:
**0 cross-split near-duplicate pairs either way** (an04's 8×8 finds 4
within-split pairs at pHash=8; the harvest pHash finds 0 — a stricter
confirmation — but no pair crosses a split under either). No figure changes sign,
no gate is weakened.

**Fixed** in the derived manifest `manifest_v07_weighted.def04.jsonl`: carries
`phash` (recorded, validated), `phash_recomputed` (harvest algorithm),
`phash_reproduces` (513 True / 19 False). A hazard-log addendum records the
correction (see `hf/HAZARD_LOG.md`).

## Derived artifacts (all under `lane_runs/def04_manifest/`)

- `manifest_v07_weighted.def04.jsonl` — 532 rows; D1 fixed, D3 validated, D2
  annotated. Passes `train.data.validate_manifest`. Only `group_size` (18 rows)
  changed among original fields; new columns appended.
- `realweak_pilot.events.def04.jsonl` — 117 rows; D2 deduplicated + annotated.
- `reproduce_report.json`, `def04_fix_report.json` — measurements + resolutions.

## Gate added

`trainer/tests/test_def04_manifest_fixes.py` (6 tests; trainer suite 183 → 189,
all green). It reproduces all three defects against the **shipped** files (so
none is silenced), validates the derived artifacts fix them, and re-measures D3
under both algorithms. It skips (never fails) when inputs are not mounted. The
existing an04 pins in `test_group_disjointness_v07.py` are untouched.

## Verify

```
docker exec -w /work/lanes2/def04_manifest/trainer iddsi-train \
  python -I /work/lanes2/def04_manifest/tools/def04_verify.py
docker exec -w /work/lanes2/def04_manifest/trainer iddsi-train \
  python -m pytest -q -p no:cacheprovider            # 189 passed
```
