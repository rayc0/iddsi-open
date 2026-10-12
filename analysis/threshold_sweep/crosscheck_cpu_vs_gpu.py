"""Cross-check the CPU forward pass against a GPU-produced reference.

Wave-1 lane ``an03_calibration`` dumped ``val_predictions_seed43.npz`` from the
same ``best.pt`` on the GPU with AMP fp16.  This container has lost CUDA, so the
sweep ran the forward pass on CPU in fp32.  This script measures how far apart
the two are, and — decisively for a threshold sweep — how far apart the SHIPPED
DECODE is: the rank rule flips only where ``P(y > k)`` is within the observed
perturbation of 0.5, so the minimum such margin is reported too.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LANE_ROOT = Path(__file__).resolve().parents[2]
TRAINER = LANE_ROOT / "trainer"
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from train.model import coral_decode, coral_logits_to_probs  # noqa: E402

REFERENCE = Path("/work/lane_runs/an03_calibration/val_predictions_seed43.npz")
MINE = Path("/work/lane_runs/fix02_threshold/val_predictions_seed43_cpu.npz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_margin(probabilities: np.ndarray) -> np.ndarray:
    """Distance of every P(y > k) from the 0.5 decision boundary."""
    probs = probabilities / probabilities.sum(axis=1, keepdims=True)
    survival = 1.0 - np.cumsum(probs, axis=1)[:, :-1]
    return np.abs(survival - 0.5)


def main() -> None:
    reference = np.load(REFERENCE, allow_pickle=True)
    mine = np.load(MINE, allow_pickle=True)

    ref_probs = np.asarray(reference["probabilities"], dtype=np.float64)
    my_probs = np.asarray(mine["probabilities"], dtype=np.float64)
    ref_logits = np.asarray(reference["cutpoint_logits"], dtype=np.float64)
    my_logits = np.asarray(mine["cutpoint_logits"], dtype=np.float64)

    ref_pred, ref_conf = coral_decode(ref_probs)
    my_pred, my_conf = coral_decode(my_probs)
    ref_labels = np.asarray(reference["labels"], dtype=np.int64)
    my_labels = np.asarray(mine["labels"], dtype=np.int64)

    # an03 stored the CALIBRATED probabilities; verify they are reproducible
    # from its own raw logits at the shipped temperature, so the comparison
    # below is apples to apples.
    ref_temperature = float(reference["temperature"]) if "temperature" in reference.files else None
    rebuilt = None
    if ref_temperature is not None:
        rebuilt = coral_logits_to_probs(
            __import__("torch").from_numpy(ref_logits / ref_temperature)
        ).numpy()

    result = {
        "reference_npz": str(REFERENCE),
        "reference_npz_sha256": sha256(REFERENCE),
        "mine_npz": str(MINE),
        "mine_npz_sha256": sha256(MINE),
        "reference_files": list(reference.files),
        "n": int(len(ref_labels)),
        "labels_identical": bool(np.array_equal(ref_labels, my_labels)),
        "event_ids_identical": bool(
            np.array_equal(np.asarray(reference["event_ids"]), np.asarray(mine["event_ids"]))
        ),
        "max_abs_prob_difference": float(np.abs(ref_probs - my_probs).max()),
        "mean_abs_prob_difference": float(np.abs(ref_probs - my_probs).mean()),
        "max_abs_logit_difference": float(np.abs(ref_logits - my_logits).max()),
        "max_abs_confidence_difference": float(np.abs(ref_conf - my_conf).max()),
        "n_decoded_level_disagreements": int((ref_pred != my_pred).sum()),
        "decoded_level_counts_reference": {
            f"L{3 + i}": int((ref_pred == i).sum()) for i in range(5)
        },
        "decoded_level_counts_mine": {f"L{3 + i}": int((my_pred == i).sum()) for i in range(5)},
        "min_decode_margin_reference": float(decode_margin(ref_probs).min()),
        "min_decode_margin_mine": float(decode_margin(my_probs).min()),
        "reference_temperature_field": ref_temperature,
    }
    if rebuilt is not None:
        result["reference_probs_rebuilt_from_its_logits_max_abs_diff"] = float(
            np.abs(ref_probs - rebuilt).max()
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
