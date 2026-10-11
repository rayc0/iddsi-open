"""Timing probe: how long would one v0.7 epoch cost on CPU?

The iddsi-train container currently has no CUDA device (NVML init failure after
a host driver reload), so this lane's inference ran on CPU. Before committing to
a full 2-seed CPU training run, measure one training epoch plus one validation
inference pass and extrapolate to the 75-epoch v0.7 schedule.

    docker exec -w /work/lanes/exp08_ensemble iddsi-train python -m exp08.probe_epoch_cost
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "trainer")):
    if path not in sys.path:
        sys.path.insert(0, path)

from train.data import ManifestImageDataset, build_transform, load_manifest, validate_manifest  # noqa: E402
from train.engine import _loader, _predict, _train_epoch  # noqa: E402
from train.losses import ClassBalancedCoralLoss, effective_number_weights  # noqa: E402
from train.model import OrdinalImageModel  # noqa: E402


def main() -> None:
    torch.set_num_threads(16)
    cfg = yaml.safe_load((REPO / "trainer" / "configs" / "exp08_ensemble.yaml").read_text(encoding="utf-8"))
    cfg["training"]["device"] = "cpu"
    cfg["data"]["num_workers"] = 4
    rows = load_manifest(cfg["data"]["manifest"])
    validate_manifest(rows, "level", "split", "group_id")
    model_cfg = cfg["model"]
    train_ds = ManifestImageDataset(
        rows, "train", build_transform(model_cfg["kind"], model_cfg["name"], 224, True),
        level_field="level", split_field="split",
    )
    val_ds = ManifestImageDataset(
        rows, "val", build_transform(model_cfg["kind"], model_cfg["name"], 224, False),
        level_field="level", split_field="split",
    )
    model = OrdinalImageModel(model_cfg["kind"], model_cfg["name"], 0.1, True)
    loss_fn = ClassBalancedCoralLoss(effective_number_weights(train_ds.labels, 0.999))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01
    )
    train_loader = _loader(train_ds, cfg, True, 17)
    val_loader = _loader(val_ds, cfg, False, 17)
    device = torch.device("cpu")

    start = time.time()
    loss = _train_epoch(model, train_loader, optimizer, loss_fn, device, False, 1.0)
    train_seconds = time.time() - start
    start = time.time()
    _predict(model, val_loader, device)
    val_seconds = time.time() - start
    per_seed_minutes = (train_seconds + val_seconds) * 75 / 60
    print(
        f"train_epoch_s={train_seconds:.1f} val_predict_s={val_seconds:.1f} loss={loss:.4f} "
        f"-> 75 epochs ~ {per_seed_minutes:.1f} min/seed, ~ {2 * per_seed_minutes:.1f} min for 2 seeds"
    )


if __name__ == "__main__":
    main()
