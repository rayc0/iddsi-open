"""Check that CPU inference reproduces the v0.7 release test predictions.

The release test_predictions.npz stores temperature-calibrated CORAL
probabilities. This script re-derives them from the raw cut-point logits saved
by ``exp08.infer_splits`` using the temperature recorded in each seed's
calibration.json, and reports the maximum absolute deviation.

Usage (inside the iddsi-train container):
    python -m exp08.verify_reproduction --raw-dir exp08/raw --release-dir /work/runs/v07
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train.model import coral_logits_to_probs

SEEDS = [17, 29, 43, 71, 101]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="exp08/raw")
    parser.add_argument("--release-dir", default="/work/runs/v07")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    release_dir = Path(args.release_dir)
    report = {}
    for seed in SEEDS:
        tag = f"v07_seed{seed}"
        raw = np.load(raw_dir / f"{tag}_test_raw.npz")
        stored = np.load(release_dir / f"siglip2_v07_fullcoverage_seed{seed}" / "test_predictions.npz")
        calibration = json.loads(
            (release_dir / f"siglip2_v07_fullcoverage_seed{seed}" / "calibration.json").read_text(encoding="utf-8")
        )
        temperature = calibration["temperature"]
        calibrated = coral_logits_to_probs(torch.from_numpy(raw["logits"] / temperature)).numpy()
        entry = {
            "temperature": float(temperature),
            "n_test": int(len(stored["labels"])),
            "event_ids_identical": bool((raw["event_ids"] == stored["event_ids"]).all()),
            "labels_identical": bool((raw["labels"] == stored["labels"]).all()),
            "max_abs_diff_calibrated": float(np.abs(calibrated - stored["probabilities"]).max()),
            "mean_abs_diff_calibrated": float(np.abs(calibrated - stored["probabilities"]).mean()),
            "argmax_disagreements": int((calibrated.argmax(1) != stored["probabilities"].argmax(1)).sum()),
        }
        report[f"seed{seed}"] = entry
        print(tag, json.dumps(entry))
    Path("exp08/reproduction_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
