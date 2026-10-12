"""Schedule-length driver for lane exp02_schedule.

Why this file exists
--------------------
``train/engine.py`` has **no learning-rate scheduler at all**: every phase runs
its epochs at a constant LR (``AdamW(parameters, lr=phase_lr)``, never stepped by
anything but the optimizer). The lane task is to compare that shipped schedule
against 2x epochs with cosine decay and a linear warmup of 10% of steps, so a
scheduler has to live somewhere.

This module is that somewhere. It does **not** modify ``train/engine.py`` or any
other shared file: it imports the engine's own helpers and re-implements
``run_seed`` line-for-line, with exactly two additions --

  1. an optional per-phase LR schedule read from ``training.scheduler``, and
  2. a per-epoch ``schedule_trace.json`` recording the LR actually applied.

Everything else -- seeding order, dataset construction, model construction, the
class-balanced CORAL loss, the AMP grad-scaler being re-created every epoch,
best-checkpoint selection by validation NLL, temperature fitting, abstention
threshold calibration, bootstrap evaluation and every output artefact -- is
copied from ``train/engine.py:run_seed``. With ``training.scheduler.type:
constant`` this driver is numerically identical to the shipped engine; that
equivalence is asserted by ``tests/test_schedule_driver_matches_engine.py``
before any variant number is believed.
"""

from __future__ import annotations

import argparse
import json
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

from train.data import ManifestImageDataset, build_transform, load_manifest, validate_manifest
from train.engine import _forward, _loader, _predict, _validation_nll
from train.losses import ClassBalancedCoralLoss, effective_number_weights
from train.model import OrdinalImageModel, coral_decode, coral_logits_to_probs
from train.utils import resolve_device, seed_everything, write_json


def _train_epoch_scheduled(
    model: OrdinalImageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ClassBalancedCoralLoss,
    device: torch.device,
    amp: bool,
    gradient_clip_norm: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> float:
    """Byte-for-byte ``train.engine._train_epoch`` plus ``scheduler.step()``.

    The GradScaler is deliberately constructed inside the function, matching the
    engine: the shipped run re-initialises the loss scale every epoch and the
    control arm must keep that quirk for the comparison to be about the
    schedule only.
    """
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
        if scheduler is not None:
            scheduler.step()
        running_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
    return running_loss / max(seen, 1)


def build_phase_scheduler(
    optimizer: torch.optim.Optimizer,
    phase: dict[str, Any],
    steps_per_epoch: int,
    scheduler_cfg: dict[str, Any],
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, dict[str, Any]]:
    """Return ``(scheduler, description)`` for one training phase.

    ``constant`` -> LR fixed at the phase value (the shipped behaviour).
    ``cosine``   -> linear warmup over ``warmup_fraction`` of the phase's
                    optimizer steps, then cosine decay from the phase peak LR to
                    ``min_lr_fraction * peak`` at the phase's final step.
    """
    kind = str(scheduler_cfg.get("type", "constant"))
    peak_lr = float(phase["lr"])
    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0), {
            "type": "constant",
            "peak_lr": peak_lr,
            "steps_per_epoch": int(steps_per_epoch),
            "warmup_fraction": 0.0,
            "min_lr_fraction": 1.0,
        }
    if kind != "cosine":
        raise ValueError(f"training.scheduler.type must be 'constant' or 'cosine', got {kind!r}")

    total_steps = max(1, int(phase["epochs"]) * int(steps_per_epoch))
    warmup_fraction = float(scheduler_cfg.get("warmup_fraction", 0.1))
    min_lr_fraction = float(scheduler_cfg.get("min_lr_fraction", 0.0))
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("training.scheduler.warmup_fraction must be in [0, 1)")
    if not 0.0 <= min_lr_fraction <= 1.0:
        raise ValueError("training.scheduler.min_lr_fraction must be in [0, 1]")
    warmup_steps = max(1, int(round(warmup_fraction * total_steps)))
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_fraction + (1.0 - min_lr_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, {
        "type": "cosine",
        "peak_lr": peak_lr,
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_fraction": warmup_fraction,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "min_lr_fraction": min_lr_fraction,
    }


def run_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """``train.engine.run_seed`` with a per-phase LR schedule bolted on."""
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
    steps_per_epoch = len(loaders["train"])
    scheduler_cfg = dict(cfg["training"].get("scheduler") or {"type": "constant"})

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
    schedule_trace: list[dict[str, Any]] = []
    epoch_number = 0
    for phase in cfg["training"]["phases"]:
        model.set_trainable_backbone(int(phase["unfreeze_last_blocks"]))
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=float(phase["lr"]), weight_decay=float(cfg["training"]["weight_decay"])
        )
        scheduler, description = build_phase_scheduler(optimizer, phase, steps_per_epoch, scheduler_cfg)
        for _ in range(int(phase["epochs"])):
            epoch_number += 1
            # The LR in force at the epoch's first optimizer step. Read BEFORE
            # training, because scheduler.step() advances it every step and an
            # after-the-fact read would report the next epoch's rate.
            lr_at_epoch_start = float(optimizer.param_groups[0]["lr"])
            train_loss = _train_epoch_scheduled(
                model, loaders["train"], optimizer, loss_fn, device, amp,
                float(cfg["training"]["gradient_clip_norm"]), scheduler,
            )
            lr_at_epoch_end = float(optimizer.param_groups[0]["lr"])
            val_probs, _, val_labels, _ = _predict(model, loaders["val"], device)
            val_nll = _validation_nll(val_probs, val_labels)
            record = {"epoch": epoch_number, "phase": phase["name"], "train_loss": train_loss, "val_nll": val_nll}
            history.append(record)
            schedule_trace.append({
                "epoch": epoch_number, "phase": phase["name"],
                "lr_at_epoch_start": lr_at_epoch_start, "lr_at_epoch_end": lr_at_epoch_end,
                **description,
            })
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
    # Kept in lockstep with train/engine.py: this driver is a fork of the
    # shipped engine and the equivalence test compares its artefacts to the
    # engine's byte for byte, so any annotation the engine adds must appear
    # here too or the comparison fails for a reason that is not the schedule.
    result["split"] = "test"
    result["level_support"] = {
        f"L{int(level) + 3}": int((test_labels == level).sum()) for level in range(5)
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "schedule_trace.json", schedule_trace)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with a per-phase LR schedule (lane exp02_schedule)")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    from train.config import load_config

    summaries = run_experiment(load_config(args.config))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
