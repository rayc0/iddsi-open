# IDDSI-Open real-weak harvester

This package autonomously collects licence-eligible third-party food images for pipeline development. Every accepted row is tagged `source=real-weak`. It never converts a visual guess into a physical-test label.

> Weak labels are **not ground truth**. A photograph cannot establish flow, softness, stickiness, cohesiveness, particle compliance, swallowing suitability, or safety. Real-tested release rows still require the frozen capture and independent RD/SLP/adjudication process in `../dataschema`.

## Run

The target environment is the NVIDIA PyTorch 25.04 container with Pillow, NumPy, OpenCV, Torch, and Transformers. The model is loaded lazily; CUDA is used only when available (or explicitly requested).

```bash
python -m harvest.run --max 2000 --out /work/data/realweak
```

Useful controls:

```bash
python -m harvest.run \
  --max 100 \
  --out /work/data/realweak \
  --sources wikimedia,openverse,open-images,nutrition5k \
  --provider-cap 0.4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --device auto
```

The OpenAI-compatible endpoint at `127.0.0.1:8090` is text-only and is deliberately not used for image labelling. Qwen3-VL-2B runs locally through Transformers. Network/source errors and model errors are recorded honestly as rejections; no result or count is fabricated. A nonzero exit status (`3`) means the candidate budget was exhausted before the requested accepted count.

Rerunning against an existing output directory resumes from its manifest, retains its pHash index, and treats `--max` as the maximum total accepted rows.
By default, no provider may contribute more than 40% of the requested accepted-row target; candidates beyond that provider limit are audited in `rejected.jsonl`. Set `--provider-cap` to a fraction greater than 0 and at most 1.

## Relabel deferred rows (GPU pass)

Harvesting with `--labeler defer` writes rows whose `weak_label_record.rule` is `"deferred"` (`model_id="deferred"`, `level_weak=null`). The `harvest.relabel` command fills those rows in later, on a machine with torch + the Qwen3-VL model:

```bash
python -m harvest.relabel --dataset /work/data/realweak_pilot
```

What it does:

- loads `events.jsonl`, selects only rows with `weak_label_record.rule == "deferred"` (already-labelled rows are passed through unchanged);
- decodes each referenced media file (path must stay inside the dataset root) and runs the labeler;
- rewrites the weak-label fields (`level_weak`, `level_declared` mirror, `weak_confidence`, `sample_kind`, `weak_label_record`, `privacy_screening.vlm_*_check`) exactly as `build_event` would have at harvest time;
- re-validates every rewritten row against the real-weak schema extension; any failure aborts the run before anything is written;
- writes the new manifest atomically (temp file + `os.replace`) after copying the original to `events.jsonl.bak-<UTC timestamp>` next to it;
- never touches `attribution.jsonl`, media files, or `rejected.jsonl`;
- if the VLM flags a face or legible text in a previously accepted image, that row is left deferred and counted as `vlm_flagged` — relabel never silently keeps a flagged image and never deletes anything (takedown is a separate audited process).

If no row is deferred, the command is a no-op: no backup, no rewrite. The summary JSON printed on stdout reports `total / deferred / relabelled / vlm_flagged / already_labelled / backup_path`.

GPU invocation, inside a CUDA container:

```bash
docker exec -e HTTP_PROXY= -e HTTPS_PROXY= -e http_proxy= -e https_proxy= -e NO_PROXY='*' \
  -e HF_HOME=/work/hf -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  iddsi-train python -m harvest.relabel --dataset /work/data/realweak_pilot
```

Add `--device cuda` to force CUDA (the default `auto` picks CUDA when available) and `--model` to override the default `Qwen/Qwen3-VL-2B-Instruct`.

## Output contract

```text
<out>/
├── events.jsonl                 # canonical real-weak event manifest
├── attribution.jsonl            # mandatory release attribution ledger
├── rejected.jsonl               # auditable exclusions/errors (no rejected media)
├── source_licenses.json         # checked source-level policy snapshot
├── realweak-event.schema.json   # generated frozen-schema extension
└── media/realweak/<provider>/...
```

`events.jsonl` retains every required field from `../dataschema/schema/event.schema.json`, keeps all physical-test fields null, uses only `split=train`, and adds provenance/privacy/pHash/weak-label fields. The frozen schema currently enumerates only `real` and `synthetic` and forbids extension fields, so changing it would violate this task's frozen-spec rule. Instead, the harvester generates `realweak-event.schema.json` by copying the frozen schema in memory, changing only `source` to the required `real-weak` constant, and adding the explicit extension fields. This makes the compatibility boundary machine-readable rather than pretending the unmodified schema accepts real-weak rows.

`level_declared` mirrors `level_weak` only to preserve the base layout; it is not a declaration by a cook and is never a tested label. `level_tested_RD`, `level_tested_SLP`, and `level_adjudicated` are always null. Downloaded stills use `plate_photo`; they do not claim that the frozen guided test shot list was performed.

## Sources and licences

The source decision ledger is [`source_licenses.json`](source_licenses.json). The runtime policy fails closed:

- Wikimedia Commons accepts only per-file CC0, CC BY, or CC BY-SA metadata with creator, licence URL, and landing page.
- Openverse searches the requested food terms with its `cc0,by` filter, but is discovery-only: the foreign landing page and returned licence fields are retained, and each row remains `verification_status=unverified` until a person or verification job checks that page.
- Open Images accepts only per-image metadata rows recorded as CC BY 2.0, preserves the original landing page, and marks each row `verification_status=unverified` because Open Images does not guarantee those rights per image.
- Nutrition5k is accepted under its dataset-level CC BY 4.0 grant and receives dataset attribution per row.
- Food-101 is disabled because ETH does not own the images and does not grant general redistribution.
- `sababishraq/foodsense-dataset` is disabled despite the HF card's CC BY 4.0 tag: the card says its images came from Yelp Open Dataset, and it does not establish authority to relicense those photos for onward redistribution.

### Release invariant

**`attribution.jsonl` is mandatory in every data release.** Do not publish `media/` or `events.jsonl` without the matching attribution rows, `source_licenses.json`, licence texts/links, and a final per-asset provenance audit. The release builder excludes every row marked `verification_status=unverified`. CC BY-SA derivatives must retain compatible share-alike treatment. Takedown or licence uncertainty requires removing the media and its event from the next release, while preserving a private audit record where lawful.

## Filtering and weak labels

Accepted images pass:

1. exact licence allowlist and required attribution;
2. safe source-host and bounded-download checks;
3. image decoding and metadata-stripping normalization;
4. exact SHA-256 plus DCT pHash near-duplicate rejection;
5. OpenCV face detection, optional Tesseract/QR text detection, and a second face/text check by Qwen;
6. Qwen visual observations followed by deterministic rules:
   - visible liquid flow → L3 with relation `at_most`;
   - uniform smooth → L4 candidate;
   - visible particles no larger than 4 mm → L5 candidate;
   - visible pieces around 15 mm (10–20 mm rule band) → L6 candidate;
   - otherwise → L7 visual bucket.

Confidence is the VLM's confidence in visible-form observations, capped at 0.95 to avoid presenting static-image weak labels as certainty. The complete model observations, rule, prompt version, and privacy decisions remain in each row.

## Tests

Tests use generated tiny images, mocked HTTP, and a fake labeler; they do not load a GPU or download a model:

```bash
python -m unittest discover -s harvest/tests -v
```
