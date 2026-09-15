from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from eval.calibration import calibrate_abstention_threshold, fit_coral_temperature
from eval.metrics import evaluate_predictions
from eval.report import write_markdown_report

from .data import ManifestImageDataset, build_transform, load_manifest, validate_manifest
from .losses import ClassBalancedCoralLoss, effective_number_weights
from .model import OrdinalImageModel, coral_decode, coral_logits_to_probs
from .utils import resolve_device, seed_everything, write_json


def _forward(model: OrdinalImageModel, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    inputs = {
        key: batch[key].to(device, non_blocking=True)
        for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
        if key in batch
    }
    return model(**inputs)


def _loader(dataset: ManifestImageDataset, cfg: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=str(cfg["training"]["device"]).startswith("cuda"),
        generator=generator,
    )


def _predict(
    model: OrdinalImageModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(probabilities, cutpoint_logits, labels, event_ids)``.

    The raw logits are needed because temperature must scale the CORAL
    cut-points; rescaling the reconstructed categorical would change the
    cumulative sums the ordinal rank decode depends on.
    """
    model.eval()
    probabilities: list[np.ndarray] = []
    raw_logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    event_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            logits = _forward(model, batch, device)
            probabilities.append(coral_logits_to_probs(logits).cpu().numpy())
            raw_logits.append(logits.float().cpu().numpy())
            labels.append(batch["label"].numpy())
            event_ids.extend(batch["event_id"])
    return (
        np.concatenate(probabilities),
        np.concatenate(raw_logits),
        np.concatenate(labels),
        event_ids,
    )


def _validation_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12)).mean())


def _train_epoch(
    model: OrdinalImageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ClassBalancedCoralLoss,
    device: torch.device,
    amp: bool,
    gradient_clip_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            loss = loss_fn(_forward(model, batch, device), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
    return running_loss / max(seen, 1)


def run_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    seed_everything(seed)
    device = resolve_device(str(cfg["training"]["device"]))
    output_dir = Path(cfg["experiment"]["output_dir"]) / f"{cfg['experiment']['name']}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({key: value for key, value in cfg.items() if not key.startswith("_")}, sort_keys=False), encoding="utf-8"
    )

    rows = load_manifest(cfg["data"]["manifest"])
    validate_manifest(rows, cfg["data"]["level_field"], cfg["data"]["split_field"], cfg["data"]["group_field"])
    model_cfg = cfg["model"]
    train_transform = build_transform(model_cfg["kind"], model_cfg["name"], int(cfg["data"]["image_size"]), True)
    eval_transform = build_transform(model_cfg["kind"], model_cfg["name"], int(cfg["data"]["image_size"]), False)
    dataset_args = {"level_field": cfg["data"]["level_field"], "split_field": cfg["data"]["split_field"]}
    train_data = ManifestImageDataset(rows, "train", train_transform, **dataset_args)
    val_data = ManifestImageDataset(rows, "val", eval_transform, **dataset_args)
    test_data = ManifestImageDataset(rows, "test", eval_transform, **dataset_args)
    if min(len(train_data), len(val_data), len(test_data)) == 0:
        raise ValueError("train, val, and test datasets must each be non-empty")
    loaders = {
        "train": _loader(train_data, cfg, True, seed),
        "val": _loader(val_data, cfg, False, seed),
        "test": _loader(test_data, cfg, False, seed),
    }

    model = OrdinalImageModel(
        model_cfg["kind"], model_cfg["name"], float(model_cfg["dropout"]), bool(model_cfg["pretrained"])
    ).to(device)
    loss_fn = ClassBalancedCoralLoss(
        effective_number_weights(train_data.labels, float(cfg["training"]["class_balance_beta"]))
    ).to(device)
    amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    best_nll = math.inf
    best_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    epoch_number = 0
    for phase in cfg["training"]["phases"]:
        model.set_trainable_backbone(int(phase["unfreeze_last_blocks"]))
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=float(phase["lr"]), weight_decay=float(cfg["training"]["weight_decay"])
        )
        for _ in range(int(phase["epochs"])):
            epoch_number += 1
            train_loss = _train_epoch(
                model, loaders["train"], optimizer, loss_fn, device, amp,
                float(cfg["training"]["gradient_clip_norm"]),
            )
            val_probs, _, val_labels, _ = _predict(model, loaders["val"], device)
            val_nll = _validation_nll(val_probs, val_labels)
            record = {"epoch": epoch_number, "phase": phase["name"], "train_loss": train_loss, "val_nll": val_nll}
            history.append(record)
            print(yaml.safe_dump(record, default_flow_style=True).strip())
            if val_nll < best_nll:
                best_nll = val_nll
                torch.save(
                    {
                        "state_dict": model.state_dict(), "metadata": model.checkpoint_metadata(),
                        "seed": seed, "epoch": epoch_number, "val_nll": val_nll,
                    },
                    best_path,
                )
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_probs, val_logits, val_labels, _ = _predict(model, loaders["val"], device)
    calibration_cfg = cfg["calibration"]
    temperature, calibrated_val_nll = fit_coral_temperature(
        val_logits, val_labels,
        float(calibration_cfg["temperature_min"]), float(calibration_cfg["temperature_max"]),
        int(calibration_cfg["temperature_steps"]),
    )
    calibrated_val_probs = coral_logits_to_probs(
        torch.from_numpy(val_logits / temperature)
    ).numpy()
    threshold_data = calibrate_abstention_threshold(
        calibrated_val_probs, val_labels,
        float(calibration_cfg["target_under_fnr"]), float(calibration_cfg["max_abstention"]),
    )
    calibration = {"temperature": temperature, "validation_nll": calibrated_val_nll, **threshold_data}

    test_probs, test_logits, test_labels, test_event_ids = _predict(model, loaders["test"], device)
    calibrated_test_probs = coral_logits_to_probs(
        torch.from_numpy(test_logits / temperature)
    ).numpy()
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        probabilities=calibrated_test_probs,
        labels=test_labels,
        event_ids=np.asarray(test_event_ids),
    )
    threshold = float(threshold_data["threshold"])
    structured_outputs = []
    for event_id, probabilities in zip(test_event_ids, calibrated_test_probs):
        decoded, decoded_confidence = coral_decode(probabilities[None, :])
        prediction = int(decoded[0])
        confidence = float(decoded_confidence[0])
        output: dict[str, Any] = {
            "event_id": event_id,
            "decision": "level_compatibility" if confidence >= threshold else "unclear",
            "confidence": confidence,
            "calibrated_threshold": threshold,
            "level_probabilities": {
                f"L{level}": float(probability)
                for level, probability in zip(range(3, 8), probabilities)
            },
            "interpretation": "visual_compatibility_only",
            "synthetic": bool(cfg["data"]["synthetic"]),
        }
        if output["decision"] == "level_compatibility":
            output["level"] = prediction + 3
        structured_outputs.append(output)
    write_json(output_dir / "test_structured_outputs.json", structured_outputs)
    result = evaluate_predictions(
        calibrated_test_probs, test_labels, threshold,
        bootstrap_samples=int(cfg.get("evaluation", {}).get("bootstrap_samples", 1000)),
        bootstrap_seed=int(cfg.get("evaluation", {}).get("bootstrap_seed", 2026)),
    )
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "calibration.json", calibration)
    write_json(output_dir / "metrics.json", result)
    write_markdown_report(
        output_dir / "report.md", result, title=f"IDDSI-Open evaluation — seed {seed}",
        synthetic=bool(cfg["data"]["synthetic"]), calibration=calibration,
    )
    return {
        "seed": seed, "output_dir": str(output_dir), "best_epoch": int(checkpoint["epoch"]),
        "calibration_target_met": bool(calibration["target_met"]), "metrics": result["metrics"],
    }


def run_experiment(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = [run_seed(cfg, int(seed)) for seed in cfg["experiment"]["seeds"]]
    root = Path(cfg["experiment"]["output_dir"])
    write_json(root / f"{cfg['experiment']['name']}_multi_seed_summary.json", summaries)
    return summaries
