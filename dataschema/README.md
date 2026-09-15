# IDDSI-Open data contract

This directory defines the capture, event, validation, documentation, and licensing contract for the open IDDSI-Open dataset. It contains no measured dataset or model-performance claims.

## Contents

```text
dataschema/
├── docs/
│   ├── CAPTURE_PROTOCOL.md
│   ├── DATASHEET_TEMPLATE.md
│   ├── DATASET_CARD_TEMPLATE.md
│   └── LICENSE_PLAN.md
├── schema/
│   ├── event.schema.json
│   ├── parquet_layout.json
│   └── PARQUET_LAYOUT.md
├── tests/
│   └── fixtures/valid_dataset/...
└── tools/
    └── validate_dataset.py
```

The canonical interchange is newline-delimited JSON at `events.jsonl`, one event per line. A conforming dataset has this shape:

```text
my_dataset/
├── events.jsonl                 # or events.parquet / partitioned events/
└── media/
    └── <kitchen_id>/...
```

One event is an independent prepared batch/test. Multiple stills, clips, retakes, or extracted frames from it remain media within that event and must not become independently split rows.

## Validation

Python 3.10 or newer is required. JSONL validation uses only the standard library:

```bash
python3 tools/validate_dataset.py /path/to/my_dataset
```

Strict validation checks:

- every row against `schema/event.schema.json`;
- unique event IDs, safe relative paths, existing media, optional byte counts/hashes, and role/type consistency;
- the required per-level food shot list or liquid syringe-flow capture;
- no faces and recorded consent for real captures;
- complete, independent RD/SLP labels and final adjudication for real release rows;
- no physical-test label fields or external-test placement for synthetic rows; and
- zero recipe, batch, or kitchen overlap among `train`, `val`, and `external_test`.

For a capture-stage audit before expert labels are complete:

```bash
python3 tools/validate_dataset.py /path/to/my_dataset --allow-incomplete-ground-truth
```

That flag does not make the records eligible for a physically-tested release. To verify recorded SHA-256 values or impose another grouped split (for example, phone):

```bash
python3 tools/validate_dataset.py /path/to/my_dataset --verify-hashes \
  --extra-group-field phone_id
```

Parquet input uses the same logical schema and requires `pyarrow`, available in the target NVIDIA training container. See `schema/PARQUET_LAYOUT.md`.

## Tests

Tests use tiny synthetic placeholder media and require no GPU or third-party package:

```bash
python3 -m unittest discover -s tests -v
```

The fixture is explicitly `source=synthetic`; it is pipeline data only and is not evidence of physical testing, accuracy, or achieved collection coverage.

## Release boundary

Real public-release events require contemporaneous physical tests independently performed by an RD and SLP, with third-person adjudication of disagreements. Declared recipe levels, visual labels, synthetic labels, and model predictions are not substitutes.

Required public disclaimer:

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

Data is intended for CC BY-NC-SA 4.0 release; code and legally eligible weights are intended for Apache-2.0 release. See `docs/LICENSE_PLAN.md` for the necessary provenance, consent, dependency, and licence checks before attaching those licences.
