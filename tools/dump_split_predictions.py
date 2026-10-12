"""Dump val + test predictions for the five released v0.7 seeds.

The released run directory stores only ``test_predictions.npz``; the validation
probabilities the calibration was fit on are never written out, so a decode-rule
comparison that must be selected on validation cannot be run from the shipped
artifacts alone.  This script reloads each seed's ``best.pt`` and re-runs
inference over the val and test splits of the v0.7 manifest.

Device note: the ``iddsi-train`` container currently cannot see the GB10
(``cuInit`` -> ``CUDA_ERROR_NO_DEVICE``), so this runs on CPU.  Inference over
218 images is minutes on CPU and the arithmetic is deterministic either way, so
the numbers do not depend on the device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.data import ManifestImageDataset, build_transform, load_manifest, validate_manifest
from train.model import OrdinalImageModel, coral_logits_to_probs
from train.utils import seed_everything


def build_loaders(cfg: dict, device: torch.device):
    rows = load_manifest(cfg["data"]["manifest"])
    validate_manifest(
        rows, cfg["data"]["level_field"], cfg["data"]["split_field"], cfg["data"]["group_field"]
    )
    model_cfg = cfg["model"]
    eval_transform = build_transform(
        model_cfg["kind"], model_cfg["name"], int(cfg["data"]["image_size"]), False
    )
    dataset_args = {"level_field": cfg["data"]["level_field"], "split_field": cfg["data"]["split_field"]}
    loaders = {}
    for split in ("val", "test"):
        dataset = ManifestImageDataset(rows, split, eval_transform, **dataset_args)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(cfg["data"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
    return loaders


def predict(model, loader, device):
    model.eval()
    probabilities, logits, labels, event_ids = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
                if key in batch
            }
            out = model(**inputs)
            probabilities.append(coral_logits_to_probs(out).cpu().numpy())
            logits.append(out.float().cpu().numpy())
            labels.append(batch["label"].numpy())
            event_ids.extend(batch["event_id"])
    return (
        np.concatenate(probabilities),
        np.concatenate(logits),
        np.concatenate(labels),
        event_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="directory holding best.pt; defaults to <config output_dir>/<name>_seed<seed>",
    )
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(args.seed)
    device = torch.device("cpu")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(cfg["experiment"]["output_dir"]) / f"{cfg['experiment']['name']}_seed{args.seed}"
    )
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model = OrdinalImageModel(
        cfg["model"]["kind"], cfg["model"]["name"], float(cfg["model"]["dropout"]),
        bool(cfg["model"]["pretrained"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    loaders = build_loaders(cfg, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    temperature = float(
        _json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))["temperature"]
    )
    for split, loader in loaders.items():
        raw_probabilities, logits, labels, event_ids = predict(model, loader, device)
        calibrated = coral_logits_to_probs(torch.from_numpy(logits / temperature)).numpy()
        path = out_dir / f"{split}_predictions_seed{args.seed}.npz"
        np.savez_compressed(
            path,
            probabilities=calibrated,
            raw_probabilities=raw_probabilities,
            cutpoint_logits=logits,
            labels=labels,
            event_ids=np.asarray(event_ids),
            seed=args.seed,
            epoch=checkpoint["epoch"],
            checkpoint_val_nll=checkpoint["val_nll"],
            split=split,
            temperature=temperature,
        )
        print(f"{split}: n={len(labels)} T={temperature:.6f} -> {path}")


if __name__ == "__main__":
    main()
