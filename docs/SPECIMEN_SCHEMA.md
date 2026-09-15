# Specimen Event Schema — Desk Specimens (IDDSI-Open)

> Status: research preview. One event row per specimen (one base food x one
> level, including FAILs). The event is the unit of the `real-tested` lane:
> no log row + no test film = no specimen. Labels from these events may be
> called "physically tested" and MUST be described as "single-operator,
> protocol-following, test-filmed" — never implying clinical or multi-rater
> validation. Research and culinary-education use only. Not IDDSI-endorsed.

## 1. Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `event_id` | string | yes | Unique event key, `EV-` + zero-padded serial (e.g. `EV-0042`). Never reused. |
| `base_food_id` | string | yes | `BF01`–`BF60`, per `docs/BASE_FOOD_LIST.md`. |
| `level` | integer | yes | Target level: 3, 4, 5, 6 or 7. |
| `prep.blend_seconds` | integer | yes | Blender run time in seconds; 0 where no blending (L6/L7 as-is). |
| `prep.sieve_mm` | number or null | for L4/L5 | Sieve gauge used (4 where the 4 mm sieve gauged L5 or strained L4); null where no sieving. |
| `prep.thickener_g` | number or null | for L3 | Weighed thickener in grams (0.1 g scale); null for non-L3. |
| `prep.liquid_ml` | number or null | for L3/L4/L5 | Added liquid (water, broth, milk) in ml; null where none added. |
| `prep.notes` | string | yes | Free text: method actually used, deviations, seeds/skins sieved, allergen extras. Empty string only if nothing to note. |
| `tests.fork_pressure` | string or null | where applicable | `pass` / `fail`, per Section 5.1 of `docs/CAPTURE_PROTOCOL.md`; null where the test does not apply (L3). |
| `tests.spoon_tilt` | string or null | where applicable | `pass` / `fail`, per Section 5.2; null where it does not apply (L6/L7). |
| `tests.fork_drip` | string or null | where applicable | `pass` / `fail`, per Section 5.3; null except L3/L4. |
| `tests.flow_ml_remaining` | number or null | L3 only | Millilitres left in the 10 ml syringe at t=10 s; L3 window is 8–10. Null for non-L3. |
| `tests.test_film_file` | string | yes | C2 test-film filename (the `*_testfilm.mp4` file). |
| `tests.pass` | boolean | yes | True only if every applicable test passed on the filmed evidence. False = FAIL row (kept, never silently kept as a pass, never renamed to another level). |
| `media.photo_45` | string | yes | S1 hero still filename (`*_45.jpg`, canonical take). |
| `media.photo_top` | string | yes | S2 top-down still filename (`*_top.jpg`, canonical take). |
| `media.photo_fork` | string | yes | S3 fork-press still filename (`*_fork.jpg`, canonical take). |
| `media.clip_file` | string | yes | C1 action-clip filename (`*_press.mp4`, canonical take). |
| `operator_id` | string | yes | Pseudonymous operator key (e.g. `OP-01`). No personal names anywhere. |
| `captured_at` | string | yes | ISO-8601 local timestamp of the filmed test. |
| `batch_id` | string | yes | Capture-day batch, `B-YYYYMMDD-NN` (e.g. `B-2026-09-18-01`). |
| `retest_of` | string or null | yes (nullable) | Null for first makes; the first-make `event_id` for a re-test second make. |
| `licence` | string | yes | `internal` for all desk-capture events (release licensing is decided at manifest build, never per row here). |
| `grouping_key` | string | derived | `base_food_id` x `batch_id` (e.g. `BF10|B-2026-09-18-01`). Computed, stored, and used as the leakage-safe split unit (Section 3). |

`pass` / `fail` values above describe only whether the specimen matched the
level descriptor in that filmed test. They are not suitability verdicts for
any person, and nothing in this schema decides whether food is suitable to
consume.

## 2. Example events (complete)

One L4 specimen (steamed pumpkin, first make, pass):

```json
{
  "event_id": "EV-0147",
  "base_food_id": "BF10",
  "level": 4,
  "prep": {
    "blend_seconds": 45,
    "sieve_mm": 4,
    "thickener_g": null,
    "liquid_ml": 20,
    "notes": "Steamed pumpkin blended 45 s with 20 ml water; passed through 4 mm sieve to remove skin fibres."
  },
  "tests": {
    "fork_pressure": "pass",
    "spoon_tilt": "pass",
    "fork_drip": "pass",
    "flow_ml_remaining": null,
    "test_film_file": "bf10-pumpkin_l4_take1_testfilm.mp4",
    "pass": true
  },
  "media": {
    "photo_45": "bf10-pumpkin_l4_take1_45.jpg",
    "photo_top": "bf10-pumpkin_l4_take1_top.jpg",
    "photo_fork": "bf10-pumpkin_l4_take1_fork.jpg",
    "clip_file": "bf10-pumpkin_l4_take1_press.mp4"
  },
  "operator_id": "OP-01",
  "captured_at": "2026-09-18T10:24:00+08:00",
  "batch_id": "B-2026-09-18-01",
  "retest_of": null,
  "licence": "internal",
  "grouping_key": "BF10|B-2026-09-18-01"
}
```

One L3 specimen (watermelon thickened to the flow window, first make, pass):

```json
{
  "event_id": "EV-0203",
  "base_food_id": "BF38",
  "level": 3,
  "prep": {
    "blend_seconds": 30,
    "sieve_mm": null,
    "thickener_g": 1.5,
    "liquid_ml": 0,
    "notes": "De-seeded watermelon blended 30 s; 1.5 g gum-based thickener sifted into 100 ml juice, rested 60 s, re-stirred. Second flow run filmed."
  },
  "tests": {
    "fork_pressure": null,
    "spoon_tilt": "pass",
    "fork_drip": "pass",
    "flow_ml_remaining": 9.0,
    "test_film_file": "bf38-watermelon_l3_take1_testfilm.mp4",
    "pass": true
  },
  "media": {
    "photo_45": "bf38-watermelon_l3_take1_45.jpg",
    "photo_top": "bf38-watermelon_l3_take1_top.jpg",
    "photo_fork": "bf38-watermelon_l3_take1_fork.jpg",
    "clip_file": "bf38-watermelon_l3_take1_press.mp4"
  },
  "operator_id": "OP-01",
  "captured_at": "2026-09-18T14:02:00+08:00",
  "batch_id": "B-2026-09-18-01",
  "retest_of": null,
  "licence": "internal",
  "grouping_key": "BF38|B-2026-09-18-01"
}
```

A re-test second make repeats the same shape with `retest_of` set to the
first-make `event_id` (e.g. `"retest_of": "EV-0147"`), a new `event_id`, its
own media files (stored under the dated re-test folder per
`docs/SHOT_LIST.md`), and a `captured_at` at least 24 h later.

## 3. Mapping to the trainer manifest

The existing trainer consumes dataset manifests with grouped splits. Desk
events map as follows:

1. **Split unit is `grouping_key`.** All levels of one base food from one
   batch (`BF10|B-2026-09-18-01`) stay in the same split. Splitting by
   individual event would leak near-identical tray, light and hand features
   across train and validation.
2. **Re-test rows NEVER enter train.** Any event with `retest_of` non-null is
   excluded from all training splits. Re-test rows exist only for the
   intra-rater reliability analysis (Cohen's kappa, per
   `docs/CAPTURE_PROTOCOL.md` Section 9) and as held-out reference.
3. **FAIL rows NEVER enter train.** Events with `tests.pass = false` are kept
   for audit and for the reliability denominator where applicable, but are
   excluded from training and validation.
4. **Eligible training rows** are first-make events (`retest_of` null) with
   `tests.pass = true`, grouped by `grouping_key`, stratified by `level`
   across splits so each of L3–L7 is represented.
5. **Media pointers resolve through the manifest only** (`media.*` and
   `tests.test_film_file` filenames joined to the release media root), so
   unreferenced spare takes on disk can never leak into a release.
6. Provenance travels with every row: the manifest records each training
   event as single-operator, protocol-following, test-filmed with its
   `event_id`, `operator_id`, `batch_id` and test-film pointer intact.
