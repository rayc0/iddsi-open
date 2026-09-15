# Parquet layout

`parquet_layout.json` is the machine-readable Arrow/Parquet contract. Each row is one independent preparation/test event, not a frame or extracted crop. The canonical logical record is defined by `event.schema.json`; JSONL and Parquet must contain the same fields.

Two storage forms are accepted:

```text
dataset/
├── events.parquet
└── media/...
```

or a partitioned dataset:

```text
dataset/
├── events/
│   ├── split=train/source=real/part-00000.parquet
│   ├── split=train/source=synthetic/part-00000.parquet
│   ├── split=val/source=real/part-00000.parquet
│   └── split=external_test/source=real/part-00000.parquet
└── media/...
```

Partitioning is physical only: `split` and `source` remain logical columns when materializing rows for validation. Dictionary encoding is recommended for identifiers and enums; Zstandard compression is recommended. Neither affects conformance.

`tools/validate_dataset.py` accepts a single Parquet file or a partitioned `events/` directory when `pyarrow` is installed. JSONL remains the zero-dependency interchange and test format.
