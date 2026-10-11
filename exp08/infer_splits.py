"""Inference for the exp08_ensemble seed-ensemble experiment.

Re-runs the trainer's own prediction path (``train.engine._predict``) over the
val and test splits of a saved seed checkpoint, on CPU, and writes the RAW
(uncalibrated) CORAL probabilities plus the cut-point logits. The logits are
kept because temperature must scale the CORAL cut-points, not the reconstructed
categorical (see ``train.eval.calibration.fit_coral_temperature``).

Why this exists: ``run_seed`` only persists *test* predictions and the scalar
calibration (temperature + abstention threshold). The ensemble needs
validation-split probabilities too, so the threshold can be fixed on
validation data exactly the way the single seeds do it.

Usage (inside the iddsi-train container, from the repo root):

    python -m exp08.infer_splits --run-dir <dir with best.pt + resolved_config.yaml> \
        --out-dir <dir> --tag <name> [--device cpu] [--threads N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "trainer"
for path in (str(REPO), str(TRAINER)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train.data import (  # noqa: E402
    ManifestImageDataset,
    build_transform,
    load_manifest,
    validate_manifest,
)
from train.engine import _loader, _predict  # noqa: E402
from train.model import OrdinalImageModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="seed run dir with best.pt + resolved_config.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True, help="prefix for the written npz files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    cfg["training"]["device"] = args.device
    cfg["data"]["num_workers"] = 0  # deterministic, no worker RNG

    device = torch.device(args.device)
    rows = load_manifest(cfg["data"]["manifest"])
    validate_manifest(rows, cfg["data"]["level_field"], cfg["data"]["split_field"], cfg["data"]["group_field"])
    model_cfg = cfg["model"]
    transform = build_transform(model_cfg["kind"], model_cfg["name"], int(cfg["data"]["image_size"]), False)
    dataset_args = {"level_field": cfg["data"]["level_field"], "split_field": cfg["data"]["split_field"]}

    model = OrdinalImageModel(
        model_cfg["kind"], model_cfg["name"], float(model_cfg["dropout"]), bool(model_cfg["pretrained"])
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"loaded {run_dir/'best.pt'} epoch={checkpoint.get('epoch')} val_nll={checkpoint.get('val_nll')}")

    for split in ("val", "test"):
        dataset = ManifestImageDataset(rows, split, transform, **dataset_args)
        loader = _loader(dataset, cfg, False, 0)
        probabilities, logits, labels, event_ids = _predict(model, loader, device)
        target = out_dir / f"{args.tag}_{split}_raw.npz"
        np.savez_compressed(
            target,
            probabilities=probabilities.astype(np.float64),
            logits=logits.astype(np.float64),
            labels=labels.astype(np.int64),
            event_ids=np.asarray(event_ids),
        )
        print(f"wrote {target} n={len(labels)} label_counts={np.bincount(labels, minlength=5).tolist()}")


if __name__ == "__main__":
    main()
