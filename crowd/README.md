# IDDSI-Open CROWD-CAPTURE

This directory is a non-deployed skeleton for a future Hugging Face Space. It
collects voluntary smartphone media into a **quarantined staging area**. It
does not grade food, perform an official IDDSI test, or decide whether anything
is safe for a person to consume.

The app displays the frozen research-preview wording from `r5_reg.md`:

> Research demonstration for culinary education. Estimates visual similarity
> to IDDSI descriptors; does not perform official IDDSI tests, assess
> swallowing, determine suitability for any person, or decide whether food is
> safe to consume.

## Consent and privacy boundary

Nothing is saved unless both boxes are checked:

1. the uploader confirms that the files contain no faces, names, badges,
   documents, screens, patient information, location metadata, or identifying
   speech; and
2. the uploader explicitly selects **“Contribute this sample to the open IDDSI
   dataset (CC BY-NC-SA 4.0)”**.

The opt-in is a dataset contribution/release permission, not evidence that the
media is private or that its level is correct. The app calls a server-side
face/text screening integration stub, but the stub honestly reports
`stub_only_not_cleared`: it implements neither face detection nor OCR. Every
submission therefore has `release_eligible=false` and stays under `incoming/`.
Do not deploy this app or publish its uploads until real frame-sampled face,
OCR, metadata, and identifying-audio screening is implemented and evaluated.

## Guided capture

- Food (user-declared L3–L7): a 45° photo is required. One 8–12 second
  spoon-tilt, fork-drip, or fork-press clip is optional in this crowd bootstrap
  interface.
- Liquid (user-declared L0–L4): one uninterrupted side-view 10 mL / 10 second
  syringe-flow video is required. A reference photo is optional.
- The uploader is asked for no name, diagnosis, prescribed level, patient
  profile, or swallowing outcome.

The full acquisition rules remain in `../dataschema/docs/CAPTURE_PROTOCOL.md`.

## Staging manifest

Local storage defaults to `crowd/data/` (gitignored):

```text
data/
├── crowd_events.jsonl
└── incoming/
    └── CRD_<UTC timestamp>_<random id>/
        ├── manifest.json
        └── media/...
```

Each event is one JSON row. Important fields are:

- `source="real-crowd"`, never `real`;
- `level_user=<the uploader's stated recipe level>`;
- `level_tested_RD=null`, `level_tested_SLP=null`, and
  `level_adjudicated=null`;
- a UTC `captured_at` value, per-file byte count and SHA-256, plus a canonical
  `sample_sha256` for the row;
- explicit contribution/licence flags and an honest privacy-screening status;
- `release_eligible=false` while the screening stub is in use.

The frozen canonical event schema currently accepts only `source=real` or
`source=synthetic` and calls the recipe claim `level_declared`. For that reason,
this is explicitly a **pre-ingestion crowd staging schema**
(`crowd-staging-1.0.0`), not a conforming release row. A later reviewed schema
migration may map `level_user` to `level_declared`, but must preserve provenance
and must not map `real-crowd` to physically tested `real` merely to satisfy the
release schema.

## Why crowd labels are weak

`level_user` records what the contributor says the recipe is. A photo cannot
measure flow, hardness, adhesiveness, or cohesiveness, and an uploaded video is
not proof that the official procedure was followed. There is no independent RD
assessment, SLP assessment, same-batch confirmation, or adjudication.

Accordingly, crowd events can support pipeline development, robustness work,
or a separately reported weak-supervision experiment. They must not enter a
physically tested benchmark split or be cited as ground truth. Promotion to a
tested release requires the same-batch physical-test workflow and adjudication
specified by the frozen capture protocol. Until then, the source remains
`real-crowd` and all tested-level fields remain null.

## Run locally

Python 3.10+:

```bash
cd crowd
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

No GPU is used. The app does not call the local VLM endpoint; privacy screening
is intentionally an unimplemented server-side hook rather than a fabricated
model result.

### Hugging Face Space deployment constraint

The organisation Space is currently constrained to the static SDK. Gradio on
an organisation Space requires the paid organisation plan; otherwise this app
must be hosted under a personal namespace. This repository does not change the
Space SDK, plan, visibility, or namespace, and W19 sends nothing externally.

Configuration:

| Variable | Default | Meaning |
|---|---:|---|
| `CROWD_DATA_DIR` | `crowd/data` | Local quarantine root |
| `CROWD_MAX_IMAGE_BYTES` | 10 MiB | Per-image cap |
| `CROWD_MAX_VIDEO_BYTES` | 80 MiB | Per-video cap |
| `CROWD_MAX_TOTAL_BYTES` | 170 MiB | Per-submission cap |
| `CROWD_RATE_LIMIT_COUNT` | 3 | Submissions per process/client window |
| `CROWD_RATE_LIMIT_WINDOW_SECONDS` | 3600 | Limiter window |
| `CROWD_HF_UPLOAD_ENABLED` | `false` | Mirror quarantine to HF when explicitly true |
| `CROWD_HF_REPO_ID` | unset | Existing HF dataset repository |
| `HF_TOKEN` | unset | Hugging Face token used by `huggingface_hub` |

HF upload is **off by default**. When enabled, the app first verifies that the
configured dataset repository is private, then uploads to
`incoming/<event_id>/`. It refuses to send uncleared crowd media to a public
repository and never creates a repository automatically.

The limiter is process-local and is only a baseline for one Space process. A
production deployment needs a shared reverse-proxy or datastore-backed limiter.
Size caps are enforced server-side before copying files.

## Tests

The tests use tiny CPU-only byte fixtures and make no network calls:

From the repository root:

```bash
pytest -q crowd/tests
```
