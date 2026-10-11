**BF13 · steamed chicken thigh 蒸雞腿 · target L6 (soft, bite-size: pieces 8–15 mm, soft, cut)**
**Bought form:** fresh chilled thigh, steamed. **Allergens:** none — confirm against the pack label at portioning and log extras in `prep.notes`.
**Tests that apply at L6:** fork pressure (5.1) only.

1. PREP
- Steam one fresh chilled chicken thigh (220–260 g raw) in a steamer basket over 1 L briskly boiling water for 18–22 min, until the core reaches 75 °C on the probe and the juices run clear with no pink. Rest 2 min.
- Strip off and discard the skin, the bone, and every piece of gristle and visible fat. Weigh the cooked trimmed meat: target 150 g (record actual).
- Dice **by hand** with a knife on the board into cubes whose longest dimension measures **8–15 mm** — no piece under 8 mm, none over 15 mm. Check every cube against the **100 mm ruler** (mm minor ticks, cm majors): the longest dimension must read between the 8 mm and 15 mm marks. Do **not** size against the checkerboard. The printed 15 mm fiducial card is 15 mm across its outer black frame and its checkerboard cells are **3.75 mm** (`tools/capture_kit.py`: 15 mm card, 4x4 grid), so "one full cell" is 3.75 mm, not 15 mm, and "half a cell" is 1.9 mm, not 8 mm. An operator who sizes by cells cuts 2–4 mm pieces — L5 material — and labels them L6.
- Pick out by hand any fragment below 8 mm; that is L5 material and must not enter an L6 specimen.
- **No blender. No sieve** — a sieve cannot enforce a 15 mm limit and must never be used to size an L6 specimen. **No thickener, no added liquid, no oil.**
- Plate **80 g** of diced meat in a single layer on the tray (record actual mass). Test at **20–40 °C**.

2. IDDSI TEST
- Fork pressure (5.1 — the only test that applies at L6): place **three** cubes on the tray — one from the top, one from the middle and one from the bottom of the plated portion. Press the flat of the fork bowl down onto each in turn with the thumb until the thumbnail blanches white, then lift. **Pass** = **every** cube tested compresses/crushes under that pressure and does not recover its shape when the fork is lifted; each stays one piece that the fork lifts whole. One cube in three that springs back fails the specimen — a single filmed cube can pass on the softest piece in a mixed batch.
- Sizing check, in the same unbroken take: hold two or three cubes against the 100 mm ruler. Every piece must measure **8–15 mm** on its longest dimension. One piece over 15 mm fails the specimen.
- Spoon tilt: **skip** — 5.2 covers L3/L4/L5. L6 is cut pieces, not a spoonable mass, so this test is not run and is not recorded as pass/fail.
- Fork drip: **skip** — 5.3 covers L3/L4. L6 pieces have no flow through the tines to observe, so this test is not run and is not recorded as pass/fail.
- Syringe flow: **skip** — 5.4 is required on every L3 only; an L6 solid is not a 10 ml flow sample.

3. VISUAL QC
- Cubes are matte pale-beige with clean cut faces — no glistening skin, no jelly sheen, no yellow fat.
- Every cube measures **8–15 mm on its longest dimension against the 100 mm ruler**; no fragment under 8 mm anywhere in the plated portion. The 15 mm fiducial card is the photo scale reference only — its checkerboard cells are 3.75 mm and must never be used as the L6 sizing gauge.
- Cubes sit separately, not clumped or stuck, and hold their cut geometry on the tray.
- No free liquid pools on the tray or runs from a cube when it is lifted.
- The fork leaves a compression mark that stays.

4. SHOTS
- `bf13-chicken_l6_take1_45.jpg` — 45° hero, diced pile on the tray, 100 mm ruler and 15 mm checkerboard in frame (`media.photo_45`).
- `bf13-chicken_l6_take1_top.jpg` — top-down, checkerboard centred under the pile (`media.photo_top`).
- `bf13-chicken_l6_take1_fork.jpg` — fork-press close-up, tines mid-press into one cube, blanched thumbnail visible (`media.photo_fork`).
- `bf13-chicken_l6_take1_press.mp4` — 8–12 s continuous fork-press, then the ruler sizing check in the same take (`media.clip_file`).
- `bf13-chicken_l6_take1_testfilm.mp4` — full IDDSI test film: fork pressure plus sizing check, one unbroken take (`tests.test_film_file`).

5. FAIL MODES
- A cube will not crush under blanching pressure and springs back (**too firm — above L6**): re-steam the whole batch **2 min** over briskly boiling water and re-test; if it still will not crush, reject the specimen. Never blend or sieve to "soften" it — blending makes paste (L4/L5) and a sieve cannot size a 15 mm limit.
- Cubes mash to paste under light finger pressure (**too soft — below L6, L5 material**): discard and re-make from a fresh thigh steamed to a 75 °C core, dicing by hand only. Do not add thickener or liquid to recover the level; neither moves a specimen toward L6.
- Any piece over 15 mm: re-dice against the **100 mm ruler** and re-check the whole plated portion before filming. A batch diced "to the checkerboard" (3.75 mm cells) is L5 material, not L6 — discard it and start again from a fresh thigh, sizing against the ruler.
- Any fragment under 8 mm: pick it out by hand or re-dice it. A batch with more than about 1 piece in 10 under 8 mm is too fine for L6 — start again from a fresh thigh and dice coarser.
- Gristle, skin or fat found in QC: discard those pieces and re-trim the remainder before plating.
- Free liquid pooling on the tray: the thigh is over-steamed. Wipe the tray edge, re-shoot within 60 s; if liquid keeps coming out of the meat, re-make with the 18 min minimum steam.
- One re-make from a reserve aliquot is allowed; a second failure on the same criterion is logged as a FAIL row (`tests.pass = false`), never relabelled.
