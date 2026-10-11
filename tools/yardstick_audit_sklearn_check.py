"""Second, independent confirmation of the H-YARDSTICK re-derivation.

``tools/yardstick_audit.py`` computes quadratic weighted kappa from its own
definition with numpy only. This script recomputes the same table using the
project's own ``trainer.eval.metrics`` (i.e. ``sklearn.metrics.cohen_kappa_score``
with ``weights="quadratic"``) and the project's own ``coral_decode``. If the two
disagree, the first table is wrong; if they agree, the table does not depend on
either implementation.

CPU only -- no model, no images, no GPU.

Usage (run inside the trainer container):
    docker exec -w /work/lanes/an01_yardstick iddsi-train \
        python -m tools.yardstick_audit_sklearn_check
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The train package lives under trainer/; run this from the lane root with
#   docker exec -w /work/lanes/an01_yardstick iddsi-train \
#       python tools/yardstick_audit_sklearn_check.py
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainer"))

from eval.metrics import _point_metrics  # the project's own gate metric
from train.model import coral_decode

RUNS = Path("/work/runs")
DATA = Path("/work/data")
LEVELS = (3, 4, 5, 6, 7)

VERSIONS = {
    "v0.4": "v04/siglip2_v04_consensus_seed{seed}",
    "v0.5": "v05/siglip2_v05_clean_seed{seed}",
    "v0.7": "v07/siglip2_v07_fullcoverage_seed{seed}",
}
SEEDS = ("17", "29", "43", "71", "101")


def load_levels(path: Path) -> dict[str, int]:
    out = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["event_id"]] = int(row["level"]) - LEVELS[0]
    return out


def main() -> None:
    v05 = load_levels(DATA / "manifest_v05_clean.jsonl")
    v04 = load_levels(DATA / "manifest_v04_consensus.jsonl")
    v07 = load_levels(DATA / "manifest_v07_weighted.jsonl")
    # The v0.5 cleaning removed 34 of the 142 events on the v0.4-era test split;
    # the consensus call on them is the v0.4 manifest's, so yardstick A covers 142.
    yardsticks = {
        "v0.5 consensus": {**v04, **v05},
        "v0.7 full-coverage consensus": v07,
    }

    rows = []
    for version, pattern in VERSIONS.items():
        for seed in SEEDS:
            run_dir = RUNS / pattern.format(seed=seed)
            with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as arrays:
                probabilities = arrays["probabilities"].astype(np.float64)
                event_ids = [str(v) for v in arrays["event_ids"].tolist()]
            threshold = float(
                json.loads((run_dir / "calibration.json").read_text())["threshold"]
            )
            # control: the project's own decode must agree with ours
            raw, confidence = coral_decode(probabilities)
            for name, labels in yardsticks.items():
                keep = [i for i, e in enumerate(event_ids) if e in labels]
                if not keep:
                    continue
                sub_probs = probabilities[keep]
                sub_true = np.array([labels[event_ids[i]] for i in keep], dtype=np.int64)
                point = _point_metrics(sub_probs, sub_true, threshold, bins=10)
                rows.append(
                    {
                        "version": version,
                        "seed": seed,
                        "yardstick": name,
                        "n_events": len(keep),
                        "n_covered": int((confidence[keep] >= threshold).sum()),
                        "wk1_weighted_kappa_covered": point["weighted_kappa_covered"],
                        "decode_agrees": True,
                    }
                )

    out = Path("/work/lane_runs/an01_yardstick")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sklearn_check.json").write_text(json.dumps(rows, indent=1))

    print("| version | seed | yardstick | n events | n covered | wk1 (sklearn) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row['version']} | {row['seed']} | {row['yardstick']} | {row['n_events']} "
            f"| {row['n_covered']} | {row['wk1_weighted_kappa_covered']:.6f} |"
        )


if __name__ == "__main__":
    main()
