# IDDSI-Open phone capture protocol

Version 1.0 — 31 August 2026

## Purpose and boundary

This protocol creates research/food-service QA records that pair guided smartphone media with contemporaneous physical-test labels. A photograph is not an official physical test and cannot establish flow, hardness, adhesiveness, cohesiveness, swallowing suitability, or safety.

Each event is one independently prepared recipe batch or one freshly prepared liquid test—not a burst frame, crop, or duplicate angle. Kitchen staff or interns capture the media. An RD and an SLP independently perform and record the applicable physical tests; a third qualified reviewer adjudicates disagreements.

Before collection, use the current official IDDSI testing instructions and locally approved food-safety SOP. If this capture brief and the official instructions differ, stop and escalate to the study lead; do not improvise.

## Kit and setup

- A charged phone with its native camera, enough storage, and a clean lens. Disable beauty filters, portrait blur, auto-enhancement when possible, and cloud accounts that expose staff names.
- A standard metal fork and a non-food-contact 15 mm reference card. The card must show a verified 15 mm feature and a unique event ID; do not use a face, staff name, patient name, or other personal information.
- A plain plate/bowl selected from the assigned plate rotation, spoon, timer, and the physical-test utensils required by the study SOP.
- For liquid events: the study-approved 10 mL syringe with 61.5 mm barrel length, stable stand/background, and 10-second timer.
- The blank ground-truth sheet at the end of this document.

Do not stage patient trays or medication. Keep the food at its recorded serving condition. Do not alter a sample merely to make a more attractive image.

## One 15-minute capture session

Plan one food event or one liquid event per session. When multiple pre-prepared samples are captured, give each its own event ID and full record.

| Time | Action |
|---|---|
| 00:00–02:00 | Confirm staff consent, no-face zone, clean lens, phone ID, kitchen ID, assigned split, free storage, and correct clock/time zone. |
| 02:00–04:00 | Assign `recipe_id`, `batch_id`, and `event_id`; place the 15 mm card; record pseudonymous operator ID, declared level, cuisine tag(s), lighting/plate categories, and serving temperature. |
| 04:00–06:00 | Food: take the 45° plate photo. Liquid: frame the syringe side view and run a rehearsal without the study sample if needed. |
| 06:00–11:00 | Record the applicable 8–12 s guided clip(s) from the shot list. Do not cut away during the test action. |
| 11:00–13:00 | Review only for focus, full utensil/test visibility, correct card, no faces, and complete action. Repeat a failed take; never delete the audit record merely because the result is unexpected. |
| 13:00–15:00 | Rename files, enter the manifest row, hand off the untouched sample and ground-truth sheet, and verify every referenced file opens. |

The RD and SLP assessment may occur immediately after capture or in an approved linked workflow, but it must concern the same batch at the same serving condition. A row is not release-ready until required ground truth and any adjudication are complete.

## Food shot list: declared L3–L7

For every food level, keep the entire utensil, sample, and 15 mm reference in frame. Use the rear camera at normal focal length (avoid digital zoom), hold the phone steady, and lock focus/exposure if the phone permits. Show the test continuously; extracted frames are derivatives and never independent events.

| Declared level | Required 45° still | Required guided clips (each 8–12 s) | What must remain visible |
|---|---|---|---|
| L3 | Plate/bowl with standard fork and 15 mm card | Spoon-tilt; fork-drip | Scoop, tilt/release, flow/drip behaviour, utensil surface after release |
| L4 | Plate/bowl with standard fork and 15 mm card | Spoon-tilt; fork-drip | Scoop, tilt/release, whether material sits above/below or through fork slots |
| L5 | Plate with standard fork and 15 mm card | Spoon-tilt; fork-press to thumbnail blanching | Representative piece before pressure, thumbnail blanching, deformation and any rebound |
| L6 | Plate with standard fork and 15 mm card | Fork-press to thumbnail blanching | Representative piece and scale, applied pressure, deformation and any rebound |
| L7 | Plate with standard fork and 15 mm card | Fork-press to thumbnail blanching | Representative piece and scale, applied pressure, deformation and any rebound; record the declared L7 variant in notes if applicable |

Photo framing: camera approximately 45° above the plate, whole plated sample visible, no portrait blur, no clipped card or fork, and enough resolution to read the reference marking. Mixed-component meals require a study decision before capture; do not silently label the whole plate from one component.

Clip framing: start recording before the utensil contacts the sample; end only after release, drip, deformation, or rebound can be seen. “Thumbnail blanching” is a visual endpoint for consistent fork-pressure capture, not an inferred force measurement.

## Liquid shot list: declared L0–L4

Record one uninterrupted side-view video of the study-approved syringe flow test for each event.

1. Mount or hold the phone perpendicular to the syringe. Show the full barrel, outlet, scale markings, and collection vessel against a contrasting background.
2. Confirm the approved syringe has a 61.5 mm barrel length and begin with exactly 10 mL at the recorded serving temperature, following the current official procedure.
3. Start recording before release. Make the release instant visible or audible, keep the syringe vertical, and run the timer continuously.
4. Keep the meniscus visible through the reading exactly 10 seconds after release, then retain at least one second of footage.
5. Repeat as a new take if there is occlusion, camera movement, bubbles/lumps that prevent a reading, wrong starting volume, uncertain release time, or incomplete 10-second view. Preserve the reason in notes.

The captured video supports a deterministic measurement workflow. It does not replace the physical test record. Boundary or technically inadequate observations must be recorded as unclear and repeated according to the approved SOP, never rounded toward a desired level.

## Required variety and split planning

The collection lead owns a prospective capture matrix; operators must follow the assigned cell rather than choosing the easiest setup. For every declared level and across both Chinese/Cantonese and other planned cuisines:

- cover multiple recipes, independent batches, operators, kitchens, and serving temperatures;
- rotate at least three documented lighting conditions across the collection (for example daylight, overhead kitchen light, and mixed/low light) without adding filters;
- rotate at least three plate/bowl appearances across the collection, including light, dark, and patterned/reflective backgrounds, while keeping the sample visible;
- use at least three documented phone models across the collection, with both iOS and Android when available; assign a pseudonymous `phone_id`, not an owner name;
- include ordinary capture imperfections as long as the physical action and scale remain readable; reject unusable media rather than secretly correcting it; and
- reserve external-test kitchens and their assigned events before capture. Never move a recipe, batch, or kitchen between splits to improve results.

These are acquisition requirements, not achieved dataset statistics. The released datasheet must report the observed counts and gaps; it must not imply that a target was met unless the manifest proves it.

## IDs, filenames, and folders

Use ASCII identifiers containing letters, digits, `.`, `_`, or `-`; never encode a person’s name or clinical information.

```text
event_id  = <kitchen_id>_<YYYYMMDD>T<HHMMSS>_<batch_id>_<sequence>
filename  = <event_id>__<capture_role>__t<take>.<extension>
```

Allowed capture roles are `plate_photo`, `spoon_tilt`, `fork_drip`, `fork_press`, and `syringe_flow`. Example:

```text
K03_20260831T142500_B017_01__plate_photo__t1.jpg
K03_20260831T142500_B017_01__fork_press__t1.mp4
```

Store paths relative to the dataset root, such as `media/K03/...`. Retakes share the event ID and increment `t1`, `t2`, etc.; the manifest may reference only quality-accepted takes, while the capture log records rejected takes and reasons. Never reuse an event ID.

## Consent, privacy, and no-face rule

- Obtain the study-approved consent from every identifiable person involved before recording and set `consent_recorded=true` for real events.
- Frame only food, utensils, hands when necessary, and the non-identifying reference card. Faces are prohibited. Any face makes the event invalid (`contains_face` must be `false`); quarantine it for privacy review and do not publish it.
- Remove names, badges, voices containing personal information, order slips, screens, reflections, geolocation metadata, and patient/resident identifiers. Avoid speech; if operational cues are required, use non-identifying fixed phrases.
- Consent to staff participation is separate from permission to release media under the dataset licence. The release manager must retain the approved consent/release record outside the public dataset.
- Do not collect patient data, prescribed texture levels, diagnoses, or swallowing outcomes in this dataset.

## Physical-test ground-truth sheet

Print one sheet per event. RD and SLP sections must be completed independently without seeing the other assessor’s level. Assessor codes and assessment IDs are pseudonymous; the secure linkage log is not published. Record observations, including failures and unclear tests, rather than forcing agreement.

### Event and serving condition

| Field | Entry |
|---|---|
| Event ID | |
| Recipe ID / batch ID | |
| Kitchen ID / phone ID | |
| Capture date-time and time zone | |
| Sample kind (`food` / `liquid`) | |
| Declared level | |
| Serving temperature (°C) | |
| Time from preparation to first test | |
| Required media filenames checked | |
| Deviations / repeat-take reasons | |

### Independent assessments

| Field | RD (complete privately) | SLP (complete privately) |
|---|---|---|
| Pseudonymous assessor code | | |
| Assessment ID | | |
| Assessment date-time | | |
| Same batch/serving condition confirmed? | Yes / No | Yes / No |
| Tests physically performed | Spoon-tilt / fork-drip / fork-pressure / flow / other: | Spoon-tilt / fork-drip / fork-pressure / flow / other: |
| For flow: start volume, release time, 10 s residual | | |
| For food: particle size / drip / tilt / pressure observations | | |
| Test invalid or unclear? Why? | | |
| Tested level (L0–L7 or unresolved) | | |
| Signature/date under study SOP | | |

### Adjudication

Complete after both sections are locked. A qualified third person must adjudicate disagreements; do not resolve them by majority voting, declared recipe level, or model output.

| Field | Entry |
|---|---|
| RD tested level | |
| SLP tested level | |
| Agreement? | Yes / No |
| Adjudication status | Not required / Adjudicated / Pending |
| Third adjudicator code (required for disagreement) | |
| Evidence reviewed / repeat physical test performed | |
| Final adjudicated level (L0–L7 or unresolved/excluded) | |
| Reasoning and protocol deviations | |
| Adjudicator signature/date | |

Only a final numeric `level_adjudicated` enters a physically tested release. Pending, unresolved, invalid, or protocol-deviating events remain quarantined or are explicitly excluded; they are never relabelled from visual appearance.
