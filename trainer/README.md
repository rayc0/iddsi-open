# IDDSI-Open training and evaluation harness

This directory contains reproducible research baselines for **visual compatibility with IDDSI L3–L7 descriptors**. It is a culinary-education and quality-assurance research tool, not a swallowing-safety classifier. Appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or suitability for a person; the relevant physical IDDSI test remains mandatory.

The primary model is `google/siglip2-base-patch16-224` with a monotone CORAL-style ordinal head. A `timm` ResNet-18 uses the same head as the small-CNN baseline. `tiny_cnn` exists only to test the plumbing on synthetic CPU fixtures.

## Install

Python 3.10 or newer is required. The DGX target is an NVIDIA PyTorch 25.04 container with one visible GPU.

```bash
cd trainer
python -m pip install -e '.[models,test]'
```

No model download or GPU is needed for the test suite.

## Data manifest

Use JSON Lines or CSV with one independent capture event per row. Required fields are:

| Field | Meaning |
|---|---|
| `event_id` | Stable event identifier |
| `image_path` | Absolute path, or path relative to the manifest |
| `level` | Physical-test label, integer L3 through L7 |
| `split` | `train`, `val`, or `test` |
| `group_id` | Recipe/batch/kitchen grouping key |

The loader rejects a `group_id` appearing in more than one split. The real manifest should also retain test method, raters/adjudication, recipe, batch, kitchen, phone, lighting, temperature, cuisine, and capture-protocol metadata. Labels must come from contemporaneous physical tests, not visual judgement. Keep the external test set locked; calibration uses validation data only.

## Train

Edit the manifest path in a config, then run:

```bash
python -m train.cli --config configs/siglip2.yaml
python -m train.cli --config configs/resnet18.yaml
```

Both release configs run multiple seeds sequentially on one GPU. Each run first trains the ordinal head with the encoder frozen, then unfreezes the configured final encoder blocks. Effective-number weights balance the training levels. The best validation checkpoint is temperature-calibrated, then a confidence threshold is selected on validation data subject to the configured coverage and dangerous under-classification target. Failure to meet that target is recorded explicitly; it is never converted into a passing result.

Every seed writes its resolved config, checkpoint, learning history, calibration decision, calibrated test predictions, abstention-aware structured outputs, JSON metrics, and Markdown report. The multi-seed JSON summary preserves separate seed outcomes instead of pooling test events.

## Evaluate saved predictions

`test_predictions.npz` contains `probabilities`, zero-based `labels`, and `event_ids`. Evaluate any equivalent file with a threshold fixed on separate validation data:

```bash
python -m eval.cli \
  --predictions runs/EXPERIMENT_SEED/test_predictions.npz \
  --threshold VALIDATION_CALIBRATED_THRESHOLD \
  --output-dir runs/EXPERIMENT_SEED/reeval
```

The report includes a per-level confusion matrix with an `unclear` column, abstention-aware macro-F1, quadratic weighted kappa on covered events, dangerous-direction under-classification FNR, ECE, coverage, a risk–coverage curve, and stratified bootstrap confidence intervals. Metric denominators are stated in the report.

For a human inter-rater comparison, pass a CSV with `event_id,rater_a_level,rater_b_level` using `--human-ratings`. Event IDs are matched to the prediction file's physical-test labels. The hook reports inter-rater agreement and each photo-only rater against the physical label separately; it does not relabel the model test set or portray agreement as safety.

## Synthetic CPU smoke run

The generator makes colored textures with fake labels solely to test file loading, optimization, calibration, evaluation, and reporting:

```bash
python -m fixtures.generate --output-dir artifacts/synthetic --per-level-per-split 2 --size 32
python -m train.cli --config configs/synthetic_cpu.yaml
```

Generated fixtures carry `physically_tested: false` and `release_eligible: false`, and their reports are watermarked:

> SYNTHETIC DATA — PIPELINE DEVELOPMENT ONLY — DO NOT RELEASE OR REPORT AS PERFORMANCE.

Never merge these fixtures into a physically tested dataset, model card, paper result, or public benchmark.

## Grounded multilingual explanations

The explanation layer never changes the structured classifier decision. It accepts one object from `test_structured_outputs.json`: `decision: level_compatibility` with an L3–L7 `level`, or `decision: unclear`. Optional observed cues and physical-test guidance may be supplied by a separate guided-test component. Patient profiles, diagnoses, safety verdicts, and aspiration-risk fields are rejected.

It can call a local Qwen3-VL-2B installation or an OpenAI-compatible chat endpoint:

```bash
python -m explain.cli \
  --config configs/explain_openai.yaml \
  --input runs/EXPERIMENT_SEED/test_structured_outputs.json \
  --event-id EVENT_ID \
  --language yue \
  --output explanation.txt
```

Languages are written Cantonese (`yue`), Traditional Chinese (`zh`), and English (`en`). A controlled disclaimer is appended in code after generation, so a model or endpoint cannot omit it.

## Interpretation boundary

Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume. Never replaces clinician-prescribed texture or physical IDDSI testing. This project is not the official IDDSI website and is not endorsed by IDDSI.
