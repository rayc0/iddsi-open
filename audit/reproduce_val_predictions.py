"""Reproduce VALIDATION-split predictions for a saved v0.7 seed checkpoint.

The training engine (trainer/train/engine.py) fits temperature and selects the
abstention threshold on validation logits but only writes the TEST predictions
to disk.  This script re-runs the identical eval-time forward pass on the
validation split so the calibration decisions can be audited (ECE, risk
coverage, and bootstrap stability of the chosen threshold) without ever
selecting anything on the test split.

It imports the shipped trainer code read-only; nothing in trainer/ is modified.

Usage (inside the iddsi-train container):
    python audit/reproduce_val_predictions.py \
        --run-dir /work/runs/v07/siglip2_v07_fullcoverage_seed43 \
        --out /work/lane_runs/an03_calibration/val_predictions_seed43.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pin_trainer import pin as pin_trainer  # noqa: E402  (must precede train/eval imports)

CODE_PROVENANCE = pin_trainer()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train.data import ManifestImageDataset, build_transform, load_manifest  # noqa: E402
from train.engine import _predict  # noqa: E402
from train.model import OrdinalImageModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="seed output dir containing best.pt")
    parser.add_argument("--out", required=True, help="destination .npz path")
    parser.add_argument("--split", default="val", help="manifest split to predict (default val)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    device = torch.device("cpu")

    rows = load_manifest(cfg["data"]["manifest"])
    transform = build_transform(
        cfg["model"]["kind"], cfg["model"]["name"], int(cfg["data"]["image_size"]), False
    )
    dataset = ManifestImageDataset(
        rows,
        args.split,
        transform,
        level_field=cfg["data"]["level_field"],
        split_field=cfg["data"]["split_field"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = OrdinalImageModel(
        cfg["model"]["kind"], cfg["model"]["name"], float(cfg["model"]["dropout"]), bool(cfg["model"]["pretrained"])
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    probabilities, logits, labels, event_ids = _predict(model, loader, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        probabilities=probabilities,
        cutpoint_logits=logits,
        labels=labels,
        event_ids=np.asarray(event_ids),
        seed=np.asarray(int(checkpoint["seed"])),
        epoch=np.asarray(int(checkpoint["epoch"])),
        checkpoint_val_nll=np.asarray(float(checkpoint["val_nll"])),
        split=np.asarray(args.split),
    )
    counts = {int(level): int((labels == level).sum()) for level in range(5)}
    print(
        yaml.safe_dump(
            {
                "code_provenance": CODE_PROVENANCE,
                "run_dir": str(run_dir),
                "split": args.split,
                "n": int(len(labels)),
                "level_counts_zero_based": counts,
                "checkpoint_seed": int(checkpoint["seed"]),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "checkpoint_val_nll": float(checkpoint["val_nll"]),
                "out": str(out),
            },
            sort_keys=False,
        ).strip()
    )


if __name__ == "__main__":
    main()
