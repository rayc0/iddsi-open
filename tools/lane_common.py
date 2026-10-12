"""Shared loader + decode-rule primitives for the fix01_decode lane.

Every number this lane reports comes through this module so the decode rules and
the metrics cannot drift between scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Must come first: the shared container installs `train`/`eval` editable against
# whichever lane last ran pip install -e, which is not this lane.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pin_trainer  # noqa: F401  (side effect: pins train/eval to this clone)

import numpy as np
import torch

from train.model import LEVELS, coral_decode, coral_interior_mass_ceiling, coral_logits_to_probs

SEEDS = [17, 29, 43, 71, 101]
PRED_DIR = Path("/work/lane_runs/fix01_decode/predictions")
RUN_DIR = Path("/work/runs/v07")
GATE = {"target_under_fnr": 0.05, "max_abstention": 0.3}

DISCLAIMER = (
    "Research demonstration for culinary education. Estimates visual similarity to IDDSI "
    "descriptors; does not perform official IDDSI tests, assess swallowing, determine "
    "suitability for any person, or decide whether food is safe to consume."
)


def load_seed(seed: int, split: str) -> dict:
    path = PRED_DIR / f"{split}_predictions_seed{seed}.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "probabilities": np.asarray(data["probabilities"], dtype=np.float64),
        "logits": np.asarray(data["cutpoint_logits"], dtype=np.float64),
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "event_ids": [str(v) for v in data["event_ids"]],
    }


def shipped_calibration(seed: int) -> dict:
    return json.loads((RUN_DIR / f"siglip2_v07_fullcoverage_seed{seed}" / "calibration.json").read_text())


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """CORAL reconstruction of the categorical at a given cut-point temperature."""
    return coral_logits_to_probs(torch.from_numpy(np.asarray(logits, dtype=np.float64) / temperature)).numpy()


def survival_from_probs(probabilities: np.ndarray) -> np.ndarray:
    """P(y > k) for k = 0..C-2, the exact inverse of coral_logits_to_probs."""
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    return 1.0 - np.cumsum(probs, axis=1)[:, :-1]


def level_counts(values: np.ndarray) -> dict[str, int]:
    return {f"L{int(LEVELS[i])}": int((np.asarray(values) == i).sum()) for i in range(len(LEVELS))}
