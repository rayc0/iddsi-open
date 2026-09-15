from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "iddsi_ordinal", "output_dir": "runs", "seeds": [17]},
    "data": {
        "manifest": "",
        "image_size": 224,
        "num_workers": 4,
        "batch_size": 16,
        "level_field": "level",
        "split_field": "split",
        "group_field": "group_id",
        "synthetic": False,
    },
    "model": {
        "kind": "siglip2",
        "name": "google/siglip2-base-patch16-224",
        "dropout": 0.1,
        "pretrained": True,
    },
    "training": {
        "weight_decay": 0.01,
        "class_balance_beta": 0.999,
        "gradient_clip_norm": 1.0,
        "amp": True,
        "device": "cuda",
        "phases": [
            {"name": "head", "epochs": 3, "lr": 3.0e-4, "unfreeze_last_blocks": 0},
            {"name": "last_blocks", "epochs": 5, "lr": 3.0e-5, "unfreeze_last_blocks": 2},
        ],
    },
    "calibration": {
        "target_under_fnr": 0.05,
        "max_abstention": 0.30,
        "temperature_min": 0.25,
        "temperature_max": 4.0,
        "temperature_steps": 81,
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = _merge(deepcopy(DEFAULTS), raw)
    cfg["_config_path"] = str(path)
    manifest = Path(cfg["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = (path.parent / manifest).resolve()
    cfg["data"]["manifest"] = str(manifest)
    output = Path(cfg["experiment"]["output_dir"])
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    cfg["experiment"]["output_dir"] = str(output)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg["model"]["kind"] not in {"siglip2", "resnet18", "tiny_cnn"}:
        raise ValueError("model.kind must be siglip2, resnet18, or tiny_cnn")
    seeds = cfg["experiment"].get("seeds", [])
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("experiment.seeds must contain at least one integer")
    if not cfg["training"].get("phases"):
        raise ValueError("training.phases must not be empty")
    for phase in cfg["training"]["phases"]:
        if int(phase["epochs"]) < 1 or float(phase["lr"]) <= 0:
            raise ValueError("every phase needs positive epochs and learning rate")
    if not 0 <= float(cfg["calibration"]["target_under_fnr"]) <= 1:
        raise ValueError("target_under_fnr must be in [0, 1]")
    if not 0 <= float(cfg["calibration"]["max_abstention"]) < 1:
        raise ValueError("max_abstention must be in [0, 1)")

