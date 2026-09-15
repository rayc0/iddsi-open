# Desk-Specimen Capture Protocol — IDDSI-Open

> Status: research preview. Research and culinary-education artifact only.
>
> Labels produced under this protocol are "physically tested" and MUST be
> described as **"single-operator, protocol-following, test-filmed"** — never
> implying clinical or multi-rater validation.
>
> This protocol records the published IDDSI test methods on camera. It does not
> assess swallowing, determine suitability for any person, or decide whether any
> food is suitable to consume. Not IDDSI-endorsed. Never claim
> "clinically validated". No safe/unsuitable verdicts are produced at any step:
> a test "pass" below means only that the specimen matched the level descriptor
> in that filmed test.

## 1. Purpose and scope

Web-harvested labels hit a measured data-quality ceiling: two VLM judges agree
on only 22.6% of images (chance for 5 classes). This protocol replaces harvested
opinion with physically tested specimens made at a desk: one person with a
blender, a fork, a sieve and thickener turns one base food into all five levels
(L3–L7) in about 10 minutes. 60 base foods x 5 levels = 300 specimens in about
10 person-hours. No kitchen, no production staff, no clinicians.

The output is the `real-tested` data lane: every label is backed by a filmed
IDDSI test of that exact specimen, plus a filmed re-test subsample that measures
intra-rater reliability (Section 9). The test film is the evidence; the photo is
the training input.

This document pairs with:

- `docs/SHOT_LIST.md` — exact media per specimen and file naming.
- `docs/BASE_FOOD_LIST.md` — the 60 base foods and per-level prep notes.
- `docs/SPECIMEN_SCHEMA.md` — the specimen event record and trainer mapping.
- `docs/capture_kit_A4.pdf` — the 15 mm checkerboard + 100 mm ruler fiducial
  that MUST appear in every photo.
- `docs/REGULATORY_POSTURE.md` — the single source of truth for external
  claims wording (nothing here overrides it).

## 2. The five levels (desk working definitions)

| Level | Name | Desk meaning | Evidence test |
|---|---|---|---|
| L7 | Whole / regular | Food as bought and normally served | Served as-is; fork-pressure check on camera |
| L6 | Soft, bite-size | Soft pieces, each <= 15 mm | 100 mm ruler in frame; fork-pressure test |
| L5 | Minced and moist | Soft particles <= 4 mm, moist throughout | 4 mm sieve as gauge; spoon-tilt test |
| L4 | Pureed | Blended smooth; holds shape on a spoon; does NOT flow through fork tines; no separate thin liquid | Spoon-tilt + fork-drip tests |
| L3 | Liquidised / moderately thick | Thinned so it pours; drips slowly through fork tines; flow test leaves 8–10 ml after 10 s | Fork-drip + 10 ml / 10 s syringe flow test |

Some foods cannot reach some levels honestly (a thin soy milk has no L7 form; a
sesame paste has no L6 pieces). `docs/BASE_FOOD_LIST.md` marks those cells N/A
with the reason. Never force a level: an N/A cell is skipped, never faked.

## 3. Equipment list (required)

1. Consumer blender (jug type, with lid vent for hot liquids).
2. Standard dinner fork (metal; used for the fork-pressure and fork-drip tests).
3. 4 mm sieve / metal strainer (particle gauge for L5; fibre/skin removal for L4).
4. Thickener (any commercial food thickener; brand-agnostic starting quantities
   in `docs/BASE_FOOD_LIST.md`; the working quantity is fixed by calibration on
   the first 5 foods and recorded per specimen).
5. Digital scale (0.1 g resolution; weighs thickener and aliquots).
6. Smartphone (rear camera; stills at highest resolution, clips at 1080p or
   better, 30 fps).
7. A4 capture-kit printout (`docs/capture_kit_A4.pdf`: 15 mm checkerboard +
   100 mm ruler), printed at 100% scale, never cropped out of frame.
8. Plain matte tray (light neutral colour; background for every shot).
9. Chopping board + small knife (for L6 sizing; kept clean between foods).
10. 10 ml slip-tip syringe without needle (L3 flow test; one per session day,
    rinsed between specimens).
11. Teaspoon / tablespoon, small bowls for the 5 aliquots, kitchen timer,
    paper log sheet (backup to the digital specimen log).

Recommended but optional: a mini tripod for the 8–12 s clips, a white napkin
for wiping the tray edge, spare printed capture kits (they get splashed).

## 4. Station setup

1. Work near a window in natural daylight. No flash, no filters, no styling, no
   garnish. The tray is the whole set.
2. Lay the A4 capture kit flat on the tray, checkerboard and ruler fully
   visible. The specimen sits beside it, never covering it.
3. Fix the phone position before starting a food: one operator can hand-hold
   the 45-degree hero still, but the 8–12 s clips SHOULD be tripod-held or
   phone-braced so the fork action stays in frame.
4. Keep the scale, sieve, fork, syringe and timer inside arm's reach. Nothing
   may be fetched mid-food: the 10-minute budget assumes zero searching.
5. Wipe the tray and fork between levels so the previous level never
   contaminates the next test film.

## 5. IDDSI test methods (performed on camera for every specimen)

These are the published IDDSI test methods, applied exactly as written. The
camera MUST show the food, the utensil action, and the result in one
unbroken take per test.

### 5.1 Fork-pressure test (L4, L6, L7; supporting check for L5)

- Place the specimen on the tray. Press the bowl of the fork down onto it with
  the thumb until the thumbnail blanches white.
- Record: the food squashes readily under that pressure and does not recover
  its shape when the fork is lifted.
- L6 sizing check in the same take: hold two or three pieces against the
  100 mm ruler; every piece is <= 15 mm.

### 5.2 Spoon-tilt test (L3, L4, L5)

- Scoop a full spoon of the specimen, hold it level, then tilt it sideways.
- L4: the sample holds its shape on the spoon and slides off as one cohesive
  mass with little residue left behind; no thin liquid separates out.
- L5: soft moist particles hold a loose mound; the spoon is not swimming in
  separate liquid.
- L3: the sample pours off the tilted spoon in a slow continuous stream.
- Record on camera: no separate thin liquid pooling on the spoon or tray. Any
  pooling means the specimen fails its level.

### 5.3 Fork-drip test (L3, L4)

- Scoop with the fork and hold it over the tray.
- L4: the sample sits in a mound above the fork tines and does NOT flow
  through them.
- L3: the sample drips slowly through the fork tines in a continuous flow,
  leaving a coating on the fork.
- Record the drip (or the mound) in close-up; this take doubles as the
  fork-press close-up still if the frame is sharp (see `docs/SHOT_LIST.md`).

### 5.4 Syringe flow test (every L3 specimen; required)

- Fill a 10 ml slip-tip syringe to the 10 ml mark with the specimen, release
  with the nozzle open, and start the timer.
- At 10 s, read the millilitres remaining in the syringe.
- L3 (liquidised / moderately thick): 8–10 ml remaining.
- Film the whole 10 s: fill level at t=0 and the reading at t=10 must both be
  legible in the test film. A test film where either reading is illegible is a
  failed capture, not a passed test — re-film or re-make.

## 6. The 10-minute per-food workflow

One base food yields up to five specimens (fewer where the food list marks N/A
cells). Total portion: enough for five small aliquots (about 60–80 g each for
solids; 100–150 ml each for L3/L4 liquids). Work L7 first, L3 last: dry to wet
keeps the station clean and the blender load logical.

| Step | Clock | Action |
|---|---|---|
| 1. Portion | 0:00–1:00 | Divide the bought food into 5 labelled aliquots. Check the food-list row for N/A cells and set those aliquots aside (they are skipped, not substituted). Photograph the bought form once per food for the log. |
| 2. L7 | 1:00–2:00 | Plate as-is. Run the fork-pressure check on camera. Take the 45-degree hero + top-down stills. Film the 8–12 s clip. Log the event. |
| 3. L6 | 2:00–4:00 | Cut / break / crumble to <= 15 mm pieces (ruler in frame). Fork-pressure test on camera. Stills + clip. Log. |
| 4. L5 | 4:00–6:00 | Chop / mash to <= 4 mm particles (sieve as gauge), moisten with the food's own liquor, broth or water. Spoon-tilt test on camera. Stills + clip. Log. |
| 5. L4 | 6:00–8:00 | Blend to a smooth puree; sieve out fibres, skins or seeds where the food-list row says so. Spoon-tilt + fork-drip tests on camera. Stills + clip. Log. |
| 6. L3 | 8:00–10:00 | Thin the L4 puree (or blend fresh + liquid), add weighed thickener to reach flow, stir, rest 60 s, re-stir. Fork-drip + full 10 ml / 10 s syringe flow test on camera. Stills + clip. Log. |

Per-level media (3 stills + 2 clips) and file naming are fixed in
`docs/SHOT_LIST.md`. Per-level fields are fixed in `docs/SPECIMEN_SCHEMA.md`.
Log each level before starting the next: a specimen without a log row and a
test film does not exist.

## 7. QC rules

1. Any specimen failing its level test is re-made once from a reserve aliquot
   or marked FAIL in the log (`tests.pass = false`, with the reason in
   `prep.notes`). A failed specimen is NEVER silently kept as a pass, and its
   media is NEVER renamed into another level.
2. Any test film with an illegible reading, a cut take, or the fiducial out of
   frame is a failed capture: re-film within the level's timebox or mark FAIL.
3. Any photo without the checkerboard + ruler in frame is retaken on the spot.
4. Separation check: if an L4 or L5 specimen shows separate thin liquid at any
   point before its clip ends, it fails that level (re-make or FAIL — the
   operator's choice, recorded).
5. Thickener discipline: every thickener addition is weighed (0.1 g) and logged
   in `prep.thickener_g`. Unweighed additions are not repeatable and the
   specimen fails QC.
6. End-of-food review (inside the 10 minutes): 5 (or fewer) log rows, 15 (or
   fewer) stills, 10 (or fewer) clips, filenames matching the convention. Gaps
   are fixed before the next food starts.

## 8. Hygiene and equipment-safety notes

- Wash hands before each session; wash the blender jug, sieve, board, knife,
  fork and bowls between base foods. Wipe down between levels.
- All animal proteins are bought cooked or cooked through (steamed) before
  portioning. Work with ready-to-eat food only.
- Allergen control: foods carrying peanut, sesame, crustacean, fish, egg,
  milk, soy or gluten (tagged per row in `docs/BASE_FOOD_LIST.md`) are followed
  by a full wash-down before the next food. Shared sieves and blender jugs are
  the main carry-over risk.
- Blender safety: sharp blades — disassemble and wash with a brush, never
  fingers; when blending hot liquids, vent the lid and start on low speed.
- Knife and fork handling is ordinary kitchen care; cut away from the body,
  fork tines down in the wash bowl.
- Specimens are research records, not servings: tray portions are discarded at
  the end of the session. Nothing returns to storage or to any plate.
- The operator works under a pseudonymous operator ID (e.g. OP-01). No personal
  names appear in logs, filenames or releases.

## 9. Re-test subsample (intra-rater reliability; required)

Label quality is measured, not asserted. A filmed re-test subsample is
mandatory:

1. Coverage: at least 10% of specimens (>= 30 of the 300), drawn by a fixed
   rule (every 10th logged specimen) plus a seeded random draw to reach 30.
2. Re-made and re-tested >= 24 h after the first make, by the same operator.
3. Blinded to the first result: the operator receives only the base food and
   the target level. First-make readings, notes and media stay sealed until
   the second make is logged under a new event with `retest_of` pointing at
   the first event.
4. Full protocol both times: same prep method, same filmed tests, same media
   set. Shortcuts invalidate the pair.
5. Analysis: report Cohen's kappa on level agreement (first-make level vs
   second-make level, 5 x 5 table) over the subsample. Target: kappa >= 0.6.
   Publish the value whatever it is; a value below target is reported as a
   limitation of the release, and no labels are rewritten to raise it.
6. Re-test rows (every event with `retest_of` set) NEVER enter training splits
   (see `docs/SPECIMEN_SCHEMA.md`). They exist only for the reliability
   analysis and as held-out reference.

## 10. Timing budget

- Per food: 10 minutes all-in (portion, process, test on camera, photograph,
  film, log, wipe-down).
- 60 foods: about 10 person-hours of capture, plus station setup (~30 min) and
  the re-test subsample (~30 specimens x ~10 min, scheduled >= 1 day later).
- If a food consistently overruns (fibrous vegetables, bony fish), the prep
  notes in `docs/BASE_FOOD_LIST.md` say so — read the row before starting the
  clock, not during it.

## 11. What the labels may honestly be called

Labels from this protocol may be called "physically tested". They MUST be
described as "single-operator, protocol-following, test-filmed". They MUST NOT
be described with clinical or multi-rater language, certified or endorsed
language, or any wording reserved by `docs/REGULATORY_POSTURE.md`. The dataset
card carries the frozen disclaimer; this protocol does not restate it, does
not replace it, and does not grant any new claim.
