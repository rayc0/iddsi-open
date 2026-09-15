from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def ordinal_targets(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels[:, None] > thresholds[None, :]).to(torch.float32)


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
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = ordinal_targets(labels, logits.shape[1] + 1)
        per_threshold = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        per_sample = per_threshold.mean(dim=1)
        sample_weights = self.class_weights[labels]
        return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
