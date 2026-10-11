from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def ordinal_targets(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels[:, None] > thresholds[None, :]).to(torch.float32)


def smoothed_ordinal_targets(
    labels: torch.Tensor, num_classes: int = 5, smoothing: float = 0.0
) -> torch.Tensor:
    """Binary CORAL targets optionally softened toward the 0/1 midpoint.

    Each threshold target ``t`` becomes ``t * (1 - eps) + eps / 2``, the direct
    ordinal analogue of categorical label smoothing. ``smoothing=0`` returns the
    hard targets unchanged, so the v0.7 recipe is bit-identical by default.
    """
    targets = ordinal_targets(labels, num_classes)
    if smoothing <= 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def effective_number_weights(labels: list[int] | np.ndarray, beta: float = 0.999) -> torch.Tensor:
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=5).astype(np.float64)
    weights = np.zeros_like(counts)
    present = counts > 0
    if beta == 0:
        weights[present] = 1.0
    else:
        weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedCoralLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = smoothed_ordinal_targets(labels, logits.shape[1] + 1, self.label_smoothing)
        per_threshold = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        per_sample = per_threshold.mean(dim=1)
        sample_weights = self.class_weights[labels]
        return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
