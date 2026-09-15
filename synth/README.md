# IDDSI-Open synthetic image generator

This package generates **synthetic, development-only** food photographs whose visible appearance is prompted toward the official IDDSI L3–L7 descriptors. It is for pipeline development and representation-aware pretraining—not physical-test ground truth, clinical validation, swallowing assessment, or safety decisions.

## Non-negotiable limitation

Synthetic images cannot encode or prove rheology. A generated photograph cannot measure flow, hardness, adhesiveness, cohesiveness, particle softness, fork pressure, spoon tilt, drip behaviour, chewing effort, or suitability for any person. Therefore:

- every manifest row has `source="synthetic"`;
- `level_declared` is only the prompt target;
- `level_tested_RD`, `level_tested_SLP`, and `level_adjudicated` are always `null`;
- synthetic rows can only enter `train` or `val`, never `external_test`;
- a visible fork/spoon cue is staged imagery, never a performed test; and
- these rows must never be counted in a physically tested dataset, result, coverage statistic, validation set, or product claim.

The broader project's `real-weak` and `real-tested` provenance classes are outside this package. This generator emits only the honest `synthetic` source tag supported by the frozen data schema.

## Prompt bank

The bank crosses all five prompt levels with more than 60 Chinese/Cantonese dishes (粥/糊/蓉/羹/燉/蒸 soft-meal families: congee, dessert pastes, vegetable and fish mashes, thick soups, steamed custards) and more than 20 Western items. It varies plate or bowl, lighting, simulated phone profile, near-45-degree framing, and an optional staged still cue (spoon tilt, fork drip, or fork press). Negative prompts suppress text, labels, logos, watermarks, IDDSI marks, faces, clinical context, utensil-only shots with no food, and common generation defects. Dish base names are level-neutral: all level wording comes from the level guidance, so no prompt carries another level's descriptor vocabulary (`python -m synth.coverage` prints the level-by-descriptor coverage matrix; `tests/test_prompt_bank_qa.py` enforces it).

Descriptor wording is prompt-oriented and deliberately includes limitations. L7 combines Easy to Chew and Regular because the frozen task requests both; the image cannot establish chewing effort and Regular itself has no texture restriction.

## Install and run

The intended runtime is the NVIDIA PyTorch 25.04 container on the DGX Spark GB10. From this directory:

```bash
python -m pip install -e '.[models]'
python -m synth.generate --level 4 --n 500 --out /work/data/synth
```

The default generator is `black-forest-labs/FLUX.1-schnell`; a load failure triggers the explicit `stabilityai/sdxl-turbo` fallback. The default judge is the local `Qwen/Qwen3-VL-2B-Instruct` transformers path. Model loading is lazy, CUDA is required by default, and CPU execution of release models requires the deliberate `--allow-cpu` flag. Unit tests use a stub and never download a model.

Useful controls:

```bash
python -m synth.generate \
  --level 5 --n 100 --out /work/data/synth \
  --batch-size 4 --seed 20260831 \
  --cuisine cantonese --cue-probability 0.4 \
  --descriptor-threshold 0.70 --quality-threshold 0.55 --cue-threshold 0.55
```

An OpenAI-compatible vision endpoint can be selected:

```bash
python -m synth.generate \
  --level 6 --n 100 --out /work/data/synth \
  --judge-backend openai \
  --endpoint http://127.0.0.1:8090/v1 \
  --judge-model YOUR_VISION_CAPABLE_MODEL
```

The configured `deepseek-v4-flash` endpoint may be text-only. If it does not accept image content, the judge records an error and rejects the candidate; it does not invent a score. `--judge-backend none` is an explicit diagnostic escape hatch and records `status="not_run"` with null scores.

The VLM judge evaluates only visible descriptor compatibility, image quality, and cue consistency. Its scores are weak model judgements—not physical labels. Invalid JSON, endpoint errors, or model errors are rejections. By default, the judge must also return the declared level; thresholds and the unknown-level policy are configurable.

## Outputs and resumability

Each invocation appends a uniquely named run:

```text
OUT/
├── events.jsonl
├── generation_log.jsonl
├── runs.jsonl
├── media/synthetic/SYN_...__plate_photo__t1.jpg
└── rejected/RUN_ID/candidate_...jpg
```

`generation_log.jsonl` records every seed, full positive and negative prompt, selected variations, actual generator model, source tag, candidate disposition, raw judge response, scores, and decision. Rejected images are retained for audit and failure analysis. `events.jsonl` contains accepted images only. Run one writer per output directory; concurrent appenders are not supported.

## Frozen schema boundary

Rows conform structurally to `../dataschema/schema/event.schema.json` and contain one real generated `plate_photo` file. The stricter `../dataschema/tools/validate_dataset.py` additionally enforces the **real capture protocol's complete guided-video shot list** for every L3–L7 event. It will therefore reject these still-only synthetic rows. This is intentional and honest: the generator does not fabricate `.mp4` files or claim that a generated still depicts a continuous 8–12 second physical test. Synthetic output remains a pretraining/pipeline artifact and is not a release-ready capture dataset.

## Tests

```bash
python -m unittest discover -s tests -v
# or: pytest
```

Tests run on CPU with 32×32 fixtures and a stub image pipeline. They verify prompt coverage, deterministic prompt sampling, VLM-decision parsing, accepted/rejected logging, media hashes, null physical labels, and each manifest row against the frozen JSON Schema. No model is imported or downloaded.

## Interpretation boundary

Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume. This project is not the official IDDSI website and is not IDDSI-endorsed.
