"""Re-calibrate and re-evaluate the SHIPPED v0.7 checkpoints with the CURRENT
trainer code, under BOTH decode rules.

Why: the v0.7 release run predates two things that are now in the repo --
the `decoded_level_counts_pre_abstention` / `interior_*` diagnostics, and the
clip in `calibrate_abstention_threshold` that stops a selected threshold from
sitting above the CORAL interior mass ceiling. The published v0.7 thresholds
sit above that ceiling in all five seeds (measured in
`analysis/baseline_v07_reproduction.json`), which is the H-CORAL symptom
arriving through the abstention gate instead of through argmax.

This script reloads each shipped checkpoint, predicts on VAL and TEST with the
v0.7 recipe's own eval transform, and then for each decode rule selects the
threshold on VALIDATION ONLY and evaluates TEST once with it. So the question
"was L5/L6 unreachable because of the decode, or because of the threshold?" is
answered on identical weights, with the threshold frozen on validation data.

No sealed file is opened. Nothing is trained here.

Output: /work/lane_runs/fix03_retrain/analysis/v07_checkpoint_recalibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.calibration import calibrate_abstention_threshold, fit_coral_temperature
from eval.metrics import evaluate_predictions
from train.config import load_config
from train.data import ManifestImageDataset, build_transform, load_manifest, validate_manifest
from train.engine import _loader, _predict
from train.model import (
    OrdinalImageModel,
    coral_decode,
    coral_interior_mass_ceiling,
    coral_logits_to_probs,
)

V07_CONFIG = Path("/work/lanes2/fix03_retrain/trainer/configs/siglip2_v07.yaml")
V07_RUNS = Path("/work/runs/v07")
OUT = Path("/work/lane_runs/fix03_retrain/analysis")
SEEDS = [17, 29, 43, 71, 101]


def main() -> None:
    cfg = load_config(V07_CONFIG)
    rows = load_manifest(cfg["data"]["manifest"])
    validate_manifest(rows, cfg["data"]["level_field"], cfg["data"]["split_field"], cfg["data"]["group_field"])
    eval_transform = build_transform(cfg["model"]["kind"], cfg["model"]["name"], int(cfg["data"]["image_size"]), False)
    dataset_args = {"level_field": cfg["data"]["level_field"], "split_field": cfg["data"]["split_field"]}
    val_data = ManifestImageDataset(rows, "val", eval_transform, **dataset_args)
    test_data = ManifestImageDataset(rows, "test", eval_transform, **dataset_args)
    device = torch.device("cuda")

    val_label_array = np.asarray(val_data.labels, dtype=np.int64)
    test_label_array = np.asarray(test_data.labels, dtype=np.int64)

    results: dict[str, object] = {
        "note": (
            "Shipped v0.7 checkpoints re-predicted with the v0.7 recipe's eval "
            "transform. Threshold selected on VALIDATION ONLY under the current "
            "calibrate_abstention_threshold (which clips candidates to the CORAL "
            "interior mass ceiling), then TEST evaluated once with that threshold."
        ),
        "val_level_support": {
            f"L{index + 3}": int((val_label_array == index).sum()) for index in range(5)
        },
        "test_level_support": {
            f"L{index + 3}": int((test_label_array == index).sum()) for index in range(5)
        },
        "n_val": len(val_data),
        "n_test": len(test_data),
        "seeds": {},
    }

    for seed in SEEDS:
        run_dir = V07_RUNS / f"siglip2_v07_fullcoverage_seed{seed}"
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model = OrdinalImageModel(
            cfg["model"]["kind"], cfg["model"]["name"],
            float(cfg["model"]["dropout"]), bool(cfg["model"]["pretrained"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        val_loader = _loader(val_data, cfg, False, seed)
        test_loader = _loader(test_data, cfg, False, seed)
        val_probs, val_logits, val_labels, _ = _predict(model, val_loader, device)
        test_probs, test_logits, test_labels, _ = _predict(model, test_loader, device)
        temperature, _ = fit_coral_temperature(
            val_logits, val_labels,
            float(cfg["calibration"]["temperature_min"]),
            float(cfg["calibration"]["temperature_max"]),
            int(cfg["calibration"]["temperature_steps"]),
        )
        calibrated_val = coral_logits_to_probs(torch.from_numpy(val_logits / temperature)).numpy()
        calibrated_test = coral_logits_to_probs(torch.from_numpy(test_logits / temperature)).numpy()
        published_calibration = json.loads((run_dir / "calibration.json").read_text())

        entry: dict[str, object] = {
            "published_threshold": float(published_calibration["threshold"]),
            "temperature": float(temperature),
            "rules": {},
        }
        for rule in ("rank", "argmax"):
            selection = calibrate_abstention_threshold(
                calibrated_val, val_labels,
                float(cfg["calibration"]["target_under_fnr"]),
                float(cfg["calibration"]["max_abstention"]),
                rule,
            )
            threshold = float(selection["threshold"])
            result = evaluate_predictions(
                calibrated_test, test_labels, threshold, bootstrap_samples=0, decode=rule
            )
            raw, confidence = coral_decode(calibrated_test, rule)
            accepted = confidence >= threshold
            predictions = np.where(accepted, raw, -1)
            entry["rules"][rule] = {
                "threshold": threshold,
                "val_selection": selection,
                "interior_mass_ceiling_test": coral_interior_mass_ceiling(calibrated_test),
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
                "decoded_level_counts_pre_abstention": result["decoded_level_counts_pre_abstention"],
                "interior_decodes_total": result["interior_decodes_total"],
                "interior_decodes_accepted": result["interior_decodes_accepted"],
                "interior_levels_reachable": result["interior_levels_reachable"],
                "emitted_level_counts": {
                    f"L{index + 3}": int((predictions == index).sum()) for index in range(5)
                },
                "per_level_recall_all_events": {
                    f"L{index + 3}": {
                        "support": int((test_labels == index).sum()),
                        "hits": int(((predictions == index) & (test_labels == index)).sum()),
                    }
                    for index in range(5)
                },
            }
        results["seeds"][str(seed)] = entry
        print(
            f"seed {seed}: published thr={entry['published_threshold']:.4f} | "
            + " | ".join(
                f"{rule} thr={entry['rules'][rule]['threshold']:.4f} "
                f"int {entry['rules'][rule]['interior_decodes_accepted']}/"
                f"{entry['rules'][rule]['interior_decodes_total']} "
                f"reach={entry['rules'][rule]['interior_levels_reachable']}"
                for rule in ("rank", "argmax")
            ),
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v07_checkpoint_recalibration.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'v07_checkpoint_recalibration.json'}")


if __name__ == "__main__":
    main()
