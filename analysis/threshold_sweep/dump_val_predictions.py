"""Dump the VALIDATION-split CORAL predictions for every v0.7 release seed.

The shipped trainer writes only ``test_predictions.npz``; the validation
probabilities that the temperature and the abstention threshold were both
selected on are never persisted.  This script rebuilds them from the shipped
checkpoints so a threshold sweep can be run on VALIDATION ONLY.

For each seed it:
  1. rebuilds the eval (no-augmentation) transform and the ``val`` loader
     exactly as ``train.engine.run_seed`` does,
  2. loads ``best.pt`` and runs the forward pass,
  3. re-fits the CORAL temperature with the shipped grid and checks it against
     the shipped ``calibration.json``,
  4. re-selects the abstention threshold with the shipped gate constants and
     checks it against the shipped ``calibration.json``,
  5. writes the CALIBRATED validation probabilities (logits / temperature,
     passed through ``coral_logits_to_probs``) plus the raw cut-point logits.

Nothing here reads, copies or trains on the test split.  The test split is
never loaded by this script.

Deterministic: eval transform, ``shuffle=False`` loader, ``torch.inference_mode``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# The container has an editable install of ANOTHER lane's trainer clone.  Force
# THIS lane's clone so every number comes from code we can name and sha256.
LANE_ROOT = Path(__file__).resolve().parents[2]
TRAINER = LANE_ROOT / "trainer"
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from eval.calibration import calibrate_abstention_threshold, fit_coral_temperature  # noqa: E402
from train.data import ManifestImageDataset, build_transform, load_manifest, validate_manifest  # noqa: E402
from train.engine import _loader, _predict  # noqa: E402
from train.model import OrdinalImageModel, coral_decode, coral_logits_to_probs  # noqa: E402
from train.utils import resolve_device, seed_everything, write_json  # noqa: E402

SEEDS = (17, 29, 43, 71, 101)
LEVELS = np.array([3, 4, 5, 6, 7], dtype=np.int64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_seed(
    seed: int,
    run_root: Path,
    manifest: str,
    out_dir: Path,
    device_override: str | None = None,
    amp_override: bool | None = None,
    suffix: str = "",
) -> dict:
    run_dir = run_root / f"siglip2_v07_fullcoverage_seed{seed}"
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    calibration = json.loads((run_dir / "calibration.json").read_text())
    # A device override is recorded, never silent: the shipped run is CUDA+AMP
    # fp16 and a CPU fp32 forward is a different numerical path.  The caller
    # cross-checks it against the GPU-produced reference before trusting it.
    resolved = json.loads(json.dumps(resolved))
    if device_override:
        resolved["training"]["device"] = device_override
    if amp_override is not None:
        resolved["training"]["amp"] = bool(amp_override)

    seed_everything(seed)
    device = resolve_device(str(resolved["training"]["device"]))
    rows = load_manifest(manifest)
    validate_manifest(
        rows,
        str(resolved["data"]["level_field"]),
        str(resolved["data"]["split_field"]),
        str(resolved["data"]["group_field"]),
    )
    model_cfg = resolved["model"]
    eval_transform = build_transform(
        model_cfg["kind"], model_cfg["name"], int(resolved["data"]["image_size"]), False
    )
    dataset_args = {
        "level_field": str(resolved["data"]["level_field"]),
        "split_field": str(resolved["data"]["split_field"]),
    }
    val_data = ManifestImageDataset(rows, "val", eval_transform, **dataset_args)
    loader = _loader(val_data, resolved, False, seed)

    model = OrdinalImageModel(
        model_cfg["kind"], model_cfg["name"], float(model_cfg["dropout"]), bool(model_cfg["pretrained"])
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    val_probs, val_logits, val_labels, val_event_ids = _predict(model, loader, device)

    # --- temperature: refit on val, compare with the shipped value -----------
    cal_cfg = resolved["calibration"]
    temperature, calibrated_val_nll = fit_coral_temperature(
        val_logits, val_labels,
        float(cal_cfg["temperature_min"]), float(cal_cfg["temperature_max"]),
        int(cal_cfg["temperature_steps"]),
    )
    calibrated_val_probs = coral_logits_to_probs(torch.from_numpy(val_logits / temperature)).numpy()

    # --- threshold: re-select on val, compare with the shipped value --------
    threshold_data = calibrate_abstention_threshold(
        calibrated_val_probs, val_labels,
        float(cal_cfg["target_under_fnr"]), float(cal_cfg["max_abstention"]),
    )

    decoded, confidence = coral_decode(calibrated_val_probs)
    payload = {
        "probabilities": calibrated_val_probs.astype(np.float64),
        "cutpoint_logits": val_logits.astype(np.float64),
        "labels": val_labels.astype(np.int64),
        "event_ids": np.asarray(val_event_ids),
        "seed": np.int64(seed),
        "epoch": np.int64(int(checkpoint["epoch"])),
        "temperature": np.float64(temperature),
        "shipped_temperature": np.float64(float(calibration["temperature"])),
        "shipped_threshold": np.float64(float(calibration["threshold"])),
        "reselect_threshold": np.float64(float(threshold_data["threshold"])),
        "val_nll_calibrated": np.float64(calibrated_val_nll),
        "split": np.asarray(["val"]),
    }
    np.savez_compressed(out_dir / f"val_predictions_seed{seed}{suffix}.npz", **payload)

    counts = {f"L{int(LEVELS[i])}": int((decoded == i).sum()) for i in range(5)}
    return {
        "seed": seed,
        "device_used": str(device),
        "amp_used": bool(resolved["training"]["amp"]),
        "run_dir": str(run_dir),
        "n_val": int(len(val_labels)),
        "val_level_support": {f"L{int(LEVELS[i])}": int((val_labels == i).sum()) for i in range(5)},
        "shipped_temperature": float(calibration["temperature"]),
        "refit_temperature": float(temperature),
        "temperature_abs_delta": abs(float(temperature) - float(calibration["temperature"])),
        "shipped_threshold": float(calibration["threshold"]),
        "reselect_threshold": float(threshold_data["threshold"]),
        "threshold_abs_delta": abs(float(threshold_data["threshold"]) - float(calibration["threshold"])),
        "shipped_target_met": bool(calibration["target_met"]),
        "reselect_target_met": bool(threshold_data["target_met"]),
        "calibrated_val_nll_refit": float(calibrated_val_nll),
        "calibrated_val_nll_shipped": (
            float(calibration["validation_nll"]) if np.isfinite(float(calibration["validation_nll"])) else None
        ),
        "decoded_level_counts_pre_abstention": counts,
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "npz": str(out_dir / f"val_predictions_seed{seed}{suffix}.npz"),
        "npz_sha256": sha256(out_dir / f"val_predictions_seed{seed}{suffix}.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="/work/runs/v07")
    parser.add_argument("--manifest", default="/work/data/manifest_v07_weighted.jsonl")
    parser.add_argument("--out-dir", default="/work/lane_runs/fix02_threshold")
    parser.add_argument(
        "--device",
        default=None,
        help="override training.device (e.g. 'cpu'). Recorded in the summary; the shipped "
             "run is cuda+amp, so a CPU override is a different numerical path and must be "
             "cross-checked before use.",
    )
    parser.add_argument("--no-amp", action="store_true", help="force amp off with a device override")
    parser.add_argument("--suffix", default="", help="suffix for the output npz name")
    args = parser.parse_args()

    import eval.metrics as _em
    import train.model as _tm
    provenance = {
        "lane": "fix02_threshold",
        "trainer_dir": str(TRAINER),
        "eval.metrics": f"{_em.__file__} sha256={sha256(Path(_em.__file__))}",
        "eval.calibration": f"{__import__('eval.calibration', fromlist=['x']).__file__} "
                            f"sha256={sha256(Path(__import__('eval.calibration', fromlist=['x']).__file__))}",
        "train.model": f"{_tm.__file__} sha256={sha256(Path(_tm.__file__))}",
        "train.engine": f"{__import__('train.engine', fromlist=['x']).__file__} "
                        f"sha256={sha256(Path(__import__('train.engine', fromlist=['x']).__file__))}",
        "note": "sys.path is pinned to this lane's trainer clone; the container's editable "
                "install of another lane's clone is shadowed.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        dump_seed(
            seed, Path(args.run_root), args.manifest, out_dir,
            device_override=args.device, amp_override=(False if args.no_amp else None),
            suffix=args.suffix,
        )
        for seed in SEEDS
    ]
    name = "val_dump_summary" + (f"_{args.suffix.strip('_')}" if args.suffix else "") + ".json"
    write_json(
        out_dir / name,
        {"provenance": provenance, "device_override": args.device, "seeds": results},
    )
    print(
        json.dumps(
            {"provenance": provenance, "device_override": args.device, "seeds": results},
            indent=2, default=str,
        )
    )


if __name__ == "__main__":
    main()
