# H-GROUP verification — v0.7 manifest, measured 2026-10-12

Question asked: from `/work/data/manifest_v07_weighted.jsonl`, assert that no
`group_id` appears in more than one split, report groups and rows per split, and
check whether any group is a near-duplicate of a group in another split (same
recipe name, same food, differing only by an id suffix).

Reproduce with:

```
python3 -I lane_runs/an04_groups/verify_group_disjointness.py     # A*, B*, C
python3 -I lane_runs/an04_groups/verify_followups.py              # D1-D7
docker exec -w /work/lanes/an04_groups/trainer iddsi-train \
  python -m pytest -q -p no:cacheprovider tests/test_group_disjointness_v07.py
```

Raw output: `lane_runs/an04_groups/report.json`, `report.txt`,
`report_followups.json`, `report_followups.txt`. Every number below is from
those files. No GPU was used; all checks are CPU-only.

## Verdict

**The literal claim holds: no `group_id` appears in more than one split.**
501 groups over 532 rows; the shipped gate `validate_manifest`
(`trainer/train/data.py:35`) enforces exactly this and does not raise on this
manifest.

**The claim is still nearly vacuous as a leakage control.** 495 of 501 groups
(98.8%) contain exactly one row, so for those rows the control separates
nothing by construction. Only 6 groups have more than one row.

**No near-duplicate of any kind straddles a split** at the criterion the split
was built with (dHash ≤ 6 **and** pHash ≤ 8, recomputed from the JPEGs on disk).
**But same-dish-different-photograph does straddle splits**: 6 food names appear
on both sides of the boundary, 6 of those cross-split pairs carry the *same*
level on both sides. That is not image leakage; it is dish-identity leakage,
which the group key was supposed to prevent and does not.

## A. Structural assertions (manifest only)

| check | result |
|---|---|
| A1 `group_id` → split | **PASS**. 501 groups, 0 span more than one split |
| A2 rows per split | train 314 (59.0%), val 110 (20.7%), test 108 (20.3%) |
| A2 groups per split | train 284, val 109, test 108 |
| A3 group cardinality | histogram `{1: 495, 2: 1, 4: 2, 8: 1, 9: 1, 10: 1}`; max size 10 |
| A4 `group_size` field vs observed | **18 rows in 3 groups disagree** (see D2) |
| A5 `event_id` uniqueness | **PASS**. 532 distinct over 532 rows |
| A6 group spanning >1 `source_corpus` | 0 |
| A7 group spanning >1 level | 3 of 501 (g0060 L5/L6/L7, g0021 L6/L7, g0026 L6/L7 — a dish photographed at several levels; not a leak) |

All 6 multi-row groups are `realweak_pilot` / Nutrition5k: g0014 (10 rows,
train), g0060 (9, train), g0010 (8, train), g0021 (4, train), g0026 (4, train),
g0045 (2, val). **No multi-row group sits in `test`.**

## B. Near-duplicate across splits

Denominators: 532 rows, 141,246 unordered image pairs, 532 `events.jsonl`
records joined (0 missing).

| probe | result |
|---|---|
| B1a `event_id` sharing first 12 chars | 532 ids in 4 clusters (the 4 corpus/source prefixes). Coarse bucket only, not dish identity — reported for completeness, carries no signal |
| B1b ids differing only by a trailing `_1`/`_2`/`-copy`/`_v2` | 0 bases, 0 cross-split |
| B1c `recipe_id` shared across splits | 532 distinct recipe_ids over 532 rows, 0 cross-split |
| B2 same food name across splits | **6 of 515 distinct names** |
| B3 same source page (`download_url` minus query) across splits | **0 of 532 pages** |
| B4 md5 over the 532 files | 532 distinct, 0 collision clusters, 0 cross-split |
| B4b recorded `media_files.sha256` | 0 clusters shared by >1 event, 0 cross-split |
| B5 recomputed dHash/pHash, dHash ≤ 6 **and** pHash ≤ 8 | 4 near-dup pairs total, **0 cross-split**, 4 inside one split |
| B6 recorded `events.jsonl` phash vs recomputed | only 4 of 532 identical, mean 4.4 of 64 bits differ |

The 4 near-dup pairs (all Nutrition5k, all inside one split):
`…ae25b1be…/…3bac101a…` (train, dh=6 ph=8), `…db1859e7…/…2c317e16…` (train,
dh=2 ph=8), `…fd30cbad…/…c95fe69b…` (val, dh=6 ph=8), `…c239fdc4…/…0e8b618e…`
(train, dh=3 ph=8). Three of the four are inside the same `group_id`, so the
grouping caught the near-duplicates it found.

### B2 detail — same food name, both sides of the split

Cross-split pairs and their perceptual distance (min over the cluster's
cross-split pairs):

| food name | events | splits | levels | cross-split pairs | of which same level | min dHash / pHash hamming |
|---|---|---|---|---|---|---|
| `congee` | 5 | train, val | 3, 4, 5 | 6 | 2 | 24 / 28 |
| `douhua` | 5 | train, val, test | 4, 6, 7 | 7 | 2 | 22 / 24 |
| `untitled` | 3 | train, test | 3, 7 | 2 | 0 | 29 / 26 |
| `korean rice porridge [juk]` | 2 | train, val | 4, 5 | 1 | 0 | 22 / 20 |
| `elaine's kitchen` | 2 | train, test | 6 | 1 | 1 | 29 / 32 |
| `roasted squashes, chanterelle puree, pumpkin seed emulsion` | 2 | train, test | 7 | 1 | 1 | 16 / 14 |

So the answer to "same recipe name, same food, differing only by an id suffix"
is: **yes for the name, no for the image.** Every one of these pairs comes from
a different source page, and none is a near-duplicate — the closest pair
(`roasted squashes…`, test L7 vs train L7) is still 16/64 dHash bits and 14/64
pHash bits apart, far outside the v0.3 near-duplicate criterion. Six cross-split
pairs carry the same level on both sides, so a model that memorises the dish can
score them without learning texture.

### B5 at a looser threshold (bound on the residual)

dHash ≤ 10 **and** pHash ≤ 12: 75 pairs, **23 cross-split**. All 23 are
Nutrition5k, all between *different* dish ids, so they are similar-looking
different dishes rather than the same dish. The strict control therefore holds
at the asserted threshold and does not hold at a looser one.

## C. Provenance of the split

- `manifest_v03.jsonl`, `manifest_v04_consensus.jsonl`, `manifest_v05_clean.jsonl`:
  **0 disagreements** with v0.7 on all 532 shared events and all 501 shared
  groups. The config comment "Split is inherited verbatim from v0.3" is true.
- `manifest_v02.jsonl` (the contaminated row-wise split): 281 of 532 events sit
  in a different split.
- **Correction to an earlier reading of mine:** the first pass of this check
  compared v0.7 against `runs/v03/grouped_manifest.jsonl` and reported 281
  disagreements. That file is the *pre-split* intermediate (its `split` column is
  the placeholder `train`); the shipped v0.3 split is `data/manifest_v03.jsonl`,
  which agrees exactly. The 281 number was an artifact of the wrong comparison
  file and is retracted.
- The v0.7 config comment `group_field: group_id  # 650 groups over 684 rows,
  asserted group-disjoint` is **stale**: those are v0.3's counts. The v0.7
  manifest has 501 groups over 532 rows.

## D. Defects found that are not leakage

1. **`group_size` is stale for 3 groups / 18 rows.** g0010 declares 9 and holds
   8 (missing `RW_nutrition5k_0e309bfdd06ac5ce9606`), g0020 declares 2 and holds
   1 (missing `RW_nutrition5k_9ca9564df5b1480977ca`), g0060 declares 10 and
   holds 9 (missing `RW_nutrition5k_d37a410296d9d73bb875`). All three missing
   rows were `train` in v0.3 and their groups are still `train`, so no
   split-disjointness consequence — but the field describes the 684-row
   population, not this 532-row manifest.
2. **`realweak_pilot/events.jsonl` has 12 duplicated event rows** (129 rows, 117
   distinct ids). 11 are byte-identical duplicates; one,
   `RW_nutrition5k_8e68a80fd452f62af95f`, **conflicts on `level_weak`** (5 vs 6)
   with identical phash, download_url and recipe_id. The manifest carries level
   6 for it. `harvest_d1` and `harvest_d1b` have no duplicated event rows.
3. **The recorded `phash` is not reproducible** from the stored JPEGs: 4 of 532
   match exactly, the rest differ by 1–11 of 64 bits (mean 4.4). Recorded phash
   cannot be used as near-duplicate evidence without recalibration; the v0.3
   dedup was right to recompute from the images.
4. **Thin val cells** (context for metric variance, not leakage): level × split
   is L3 28/14/10, L4 37/8/11, L5 13/6/7, L6 35/9/10, L7 201/73/70
   (train/val/test). L4 has 8 val rows and L5 has 6.

## Gate added

`trainer/tests/test_group_disjointness_v07.py` (6 tests, all passing in the
`iddsi-train` container alongside the 37 pre-existing tests, 43 total):

- group-disjointness through the shipped `validate_manifest` gate
- group-disjointness computed independently
- no near-duplicate image pair across splits (dHash/pHash recomputed)
- no dish identity and no source page across splits
- `event_id` uniqueness
- the `group_size` staleness **pinned** at 3 groups / 18 rows so the defect
  cannot drift unnoticed

The tests skip (never fail) when the manifest or the corpus event files are not
mounted, so the code-only checkout stays green.

## What this does and does not settle

Settled: the split is group-disjoint on the group key as configured, and no
image-level near-duplicate, byte duplicate, source page, dish id or recipe id
crosses the boundary.

Not settled: whether the group key is the right unit. 98.8% of groups are
singletons, so the key still separates almost nothing, and 6 same-dish
cross-split pairs (2 of them at the same level in `test` vs `train`) show the
unit is finer than the dish. A group built from dish identity — the Nutrition5k
`dish_<id>` and the source page — would have caught those. That is a re-split,
and a re-split re-randomises the eval, so no published figure carries over.
