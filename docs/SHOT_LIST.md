# Shot List — Desk-Specimen Capture (IDDSI-Open)

> Status: research preview. Companion to `docs/CAPTURE_PROTOCOL.md` (the
> procedure), `docs/BASE_FOOD_LIST.md` (the foods) and `docs/SPECIMEN_SCHEMA.md`
> (the log record). Every specimen gets exactly the media below: 3 stills +
> 2 clips. A specimen missing any item is incomplete until the item is captured
> or the specimen is marked FAIL.

## 1. Media per specimen (3 stills + 2 clips)

All five shots share the station: plain matte tray, A4 capture kit flat beside
the food (15 mm checkerboard + 100 mm ruler fully visible), natural daylight,
no flash, no filters, no styling.

### Stills

| # | Shot | Framing | What must be legible |
|---|---|---|---|
| S1 | 45-degree hero | Phone at ~45 degrees to the tray, food centred, kit alongside | Food texture and surface; checkerboard squares; ruler markings; level aliquot fills a normal spoon/bowl portion, not a tasting dot |
| S2 | Top-down with checkerboard | Phone directly above, food on or beside the checkerboard area | True plan view of particle/piece spread; checkerboard grid for scale; for L6, several pieces against the ruler |
| S3 | Fork-press close-up, mid-press | Tight crop on the fork bowl pressing the food, thumbnail visible | Thumbnail blanched white under pressure; food squashing (L4/L6/L7) or mound/particles (L3/L5); fork tines and food contact sharp, not blurred |

S3 is taken from the test-film take where possible (a sharp frame of the press
itself). If no sharp frame exists, stage the press again — the food state, not
the camera, is what matters, so re-press the same aliquot rather than opening
a new one.

### Clips (each 8–12 s, 1080p or better, 30 fps, one unbroken take)

| # | Clip | Content | Pass bar |
|---|---|---|---|
| C1 | Fork-press / spoon-tilt clip | The level's action test performed once, cleanly: fork-press (L4/L6/L7), spoon-tilt (L3/L4/L5), or fork-drip (L3/L4) — whichever is the primary evidence test for the level | Utensil, food and result visible throughout; for spoon-tilt, the slide/pour and the clean (or not-clean) spoon; for fork-press, the squash and the non-recovery |
| C2 | IDDSI test film | The full level test per `docs/CAPTURE_PROTOCOL.md` Section 5: for L3 this MUST include the complete 10 ml / 10 s syringe run (fill mark at t=0 and residual reading at t=10 legible); for other levels the filmed fork-pressure / spoon-tilt / fork-drip test | No cuts; readings legible; fiducial in frame at the start of the take; timer visible or audible for the L3 flow run |

For L3, C1 and C2 may be filmed back-to-back but are stored as two files. For
L4, the spoon-tilt take serves as C1 and the fork-drip take as C2. For L5–L7,
C2 is the fuller test record (sizing against the ruler included) while C1 is
the single clean action.

## 2. File naming convention

Pattern (all lowercase, ASCII only):

```
<basefood>_<level>_take<n>_<shot>.<ext>
```

- `<basefood>`: the food ID + short slug, e.g. `bf01-rice`, `bf38-watermelon`.
- `<level>`: `l3`, `l4`, `l5`, `l6`, `l7`.
- `<n>`: take number from 1 within the specimen and shot type.
- `<shot>`: `45` (S1 hero), `top` (S2), `fork` (S3), `press` (C1 clip),
  `testfilm` (C2 clip).
- `<ext>`: `jpg` for stills, `mp4` for clips.

Examples for one L4 pumpkin specimen, first takes:

```
bf10-pumpkin_l4_take1_45.jpg
bf10-pumpkin_l4_take1_top.jpg
bf10-pumpkin_l4_take1_fork.jpg
bf10-pumpkin_l4_take1_press.mp4
bf10-pumpkin_l4_take1_testfilm.mp4
```

A second take of the hero still (first was blurred) is
`bf10-pumpkin_l4_take2_45.jpg`; the log records which take is canonical in the
`media` fields. Retakes never overwrite: keep every take on disk, reference
the best in the log.

Re-test subsample media (see `docs/CAPTURE_PROTOCOL.md` Section 9) uses the
same pattern inside a dated re-test folder (e.g. `retest-2026-09-20/`), so a
re-made specimen never collides with its first-make filenames.

## 3. Minimum technical requirements

- Stills: rear camera, highest resolution available (at least 12 MP
  recommended), JPEG, auto-exposure allowed, HDR off (it smears texture),
  tap-to-focus on the food.
- Clips: at least 1080p, 30 fps, H.264/H.265 MP4, 8–12 s each. Shorter than
  8 s misses the action; longer than 12 s wastes review time — stop the take.
- Light: natural daylight near a window, diffused (sheer curtain if harsh).
  No mixed tungsten/daylight, no overhead spotlight glare on wet foods, no
  phone shadow across the tray. If the operator's shadow falls on the tray,
  rotate the station, not the white balance.
- Background: the plain matte tray only. No tablecloth patterns, no hands
  except the testing hand, no packaging, no other foods in frame.
- Fiducial: checkerboard + ruler in frame in every still and at the start of
  every clip. The 15 mm checkerboard is the scale reference for the model; the
  100 mm ruler is the sizing reference for L6 pieces and L5 particles.
- Sound: irrelevant; clips may be muted in review. Do not narrate the level
  verdict on the audio track — the log carries the result, and blinded
  re-tests must not overhear it.

## 4. Per-specimen checklist (print one copy per specimen)

Copy this block per specimen, tick each box before moving to the next level:

```
Specimen: [BF__-slug]  Level: [L_]  Operator: [OP-__]  Date: [____-__-__]
Batch: [B-________-__]  Event: [EV-____]

Setup
[ ] Tray clean, capture kit flat, checkerboard + ruler visible
[ ] Daylight steady, no phone shadow, HDR off, 1080p/30fps clip mode checked
[ ] Aliquot portioned, N/A-cells skipped per the food-list row

Stills (jpg, fiducial in frame)
[ ] S1 45-degree hero: <basefood>_<level>_take<n>_45.jpg
[ ] S2 top-down:       <basefood>_<level>_take<n>_top.jpg
[ ] S3 fork-press mid-press: <basefood>_<level>_take<n>_fork.jpg

Clips (mp4, 8-12 s, unbroken take)
[ ] C1 action clip:    <basefood>_<level>_take<n>_press.mp4
[ ] C2 test film:      <basefood>_<level>_take<n>_testfilm.mp4
[ ] L3 only: syringe fill mark at t=0 AND residual reading at t=10 legible

Test result (matches the filmed test, not memory)
[ ] Fork-pressure / spoon-tilt / fork-drip / flow outcome recorded
[ ] L6: pieces <= 15 mm against ruler  |  L5: particles <= 4 mm via sieve
[ ] L4: holds shape, no flow through tines, no separate thin liquid
[ ] L3: drips slowly through tines; flow test 8-10 ml remaining
[ ] PASS or FAIL written in the log row now (never later, never blank)

Close-out
[ ] Filenames match the convention; canonical takes referenced in the log
[ ] Tray + fork wiped for the next level
```

End-of-food review (inside the 10-minute budget): count files against ticked
boxes. An unticked box is a re-capture or a FAIL row — never an empty promise
that the file "will be added later".
