from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = (3, 4, 5, 6, 7)


def coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert monotone P(y > k) logits into probabilities for L3..L7."""
    survival = torch.sigmoid(logits)
    probs = torch.cat(
        [1.0 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]], dim=1
    )
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def coral_decode(probabilities: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Decode CORAL probabilities by the ordinal RANK rule.

    The level index is ``#{k : P(y > k) > 0.5}`` (Cao, Mirjalili & Raschka),
    NOT ``argmax`` over the reconstructed categorical distribution.

    Why this matters: :func:`coral_logits_to_probs` builds interior classes as
    *differences of adjacent sigmoids*, so P(interior) is bounded by roughly
    ``tanh(gap / 4)`` while the two tail classes keep their full mass. With the
    default cut-point spacing (~0.69) every interior class is capped near 0.17
    while both tails reach ~0.26, so ``argmax`` provably can never return an
    interior level for ANY input. The first trained run emitted only L3, L7 and
    abstain because of this, on balanced L4-L7 data with no L3 at all.

    Returns ``(level_index, confidence)`` where confidence is the probability
    mass on the decoded level.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a [N, C] matrix")
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    # P(y > k) for k = 0 .. C-2, recovered from the categorical form.
    survival = 1.0 - np.cumsum(probs, axis=1)[:, :-1]
    predictions = (survival > 0.5).sum(axis=1).astype(np.int64)
    confidence = probs[np.arange(len(predictions)), predictions]
    return predictions, confidence


class OrderedCoralHead(nn.Module):
    """CORAL-style shared projection with strictly ordered learned cut-points."""

    def __init__(self, in_features: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(in_features, 1, bias=False)
        self.first_bias = nn.Parameter(torch.tensor(1.5))
        # softplus(0)=0.693 gave interior classes a ~0.17 probability ceiling.
        # softplus(1.5)=1.70 lifts it to ~0.40 so every level is reachable.
        self.bias_gaps_raw = nn.Parameter(torch.full((num_classes - 2,), 1.5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        gaps = F.softplus(self.bias_gaps_raw) + 1e-4
        biases = torch.cat(
            [self.first_bias.reshape(1), self.first_bias - torch.cumsum(gaps, dim=0)]
        )
        return self.score(self.dropout(features)) + biases


class TinyCNN(nn.Module):
    """Small test-only backbone; not a release model or benchmark baseline."""

    feature_dim = 32

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.features(pixel_values).flatten(1)


class OrdinalImageModel(nn.Module):
    def __init__(
        self,
        kind: str,
        name: str = "google/siglip2-base-patch16-224",
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.kind = kind
        self.name = name
        if kind == "siglip2":
            try:
                from transformers import AutoConfig, Siglip2VisionModel
            except ImportError as exc:
                raise ImportError("SigLIP-2 requires `pip install -e '.[models]'`") from exc
            if pretrained:
                self.backbone = Siglip2VisionModel.from_pretrained(name)
            else:
                full_config = AutoConfig.from_pretrained(name)
                self.backbone = Siglip2VisionModel(getattr(full_config, "vision_config", full_config))
            feature_dim = int(self.backbone.config.hidden_size)
        elif kind == "resnet18":
            try:
                import timm
            except ImportError as exc:
                raise ImportError("ResNet-18 requires `pip install -e '.[models]'`") from exc
            self.backbone = timm.create_model("resnet18", pretrained=pretrained, num_classes=0, global_pool="avg")
            feature_dim = int(self.backbone.num_features)
        elif kind == "tiny_cnn":
            self.backbone = TinyCNN()
            feature_dim = self.backbone.feature_dim
        else:
            raise ValueError(f"unknown model kind: {kind}")
        self.head = OrderedCoralHead(feature_dim, len(LEVELS), dropout)
        self._trainable_last_blocks = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.kind == "siglip2":
            output = self.backbone(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state[:, 0]
        else:
            features = self.backbone(pixel_values)
        return self.head(features)

    def set_trainable_backbone(self, last_blocks: int) -> None:
        self._trainable_last_blocks = last_blocks
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if last_blocks <= 0:
            return
        blocks = self._backbone_blocks()
        for block in blocks[-last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        # Final normalization should adapt with any unfrozen transformer block.
        if self.kind == "siglip2":
            tower = self._siglip_tower()
            for attr in ("post_layernorm", "head"):
                module = getattr(tower, attr, None)
                if isinstance(module, nn.Module):
                    for parameter in module.parameters():
                        parameter.requires_grad = True

    def _siglip_tower(self) -> nn.Module:
        """Return the module that actually holds the transformer stack.

        ``Siglip2VisionModel`` nests its encoder, post_layernorm and head one
        level down under ``.vision_model``; a bare vision config loaded without
        that wrapper exposes them directly. Accept both rather than assuming.
        """
        return getattr(self.backbone, "vision_model", self.backbone)

    def _backbone_blocks(self) -> list[nn.Module]:
        if self.kind == "siglip2":
            encoder = getattr(self._siglip_tower(), "encoder", None)
            layers = getattr(encoder, "layers", None)
            if layers is None:
                raise RuntimeError("Unsupported SigLIP-2 transformer layout: encoder.layers not found")
            return list(layers)
        if self.kind == "resnet18":
            return [module for name, module in self.backbone.named_children() if name.startswith("layer")]
        return [self.backbone.features[0], self.backbone.features[2]]

    def train(self, mode: bool = True) -> "OrdinalImageModel":
        super().train(mode)
        if mode:
            # Frozen stages stay in inference mode, including BatchNorm/dropout state.
            self.backbone.eval()
            if self._trainable_last_blocks > 0:
                for block in self._backbone_blocks()[-self._trainable_last_blocks:]:
                    block.train()
                if self.kind == "siglip2":
                    tower = self._siglip_tower()
                    for attr in ("post_layernorm", "head"):
                        module = getattr(tower, attr, None)
                        if isinstance(module, nn.Module) and any(p.requires_grad for p in module.parameters()):
                            module.train()
        return self

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"model_kind": self.kind, "model_name": self.name, "levels": list(LEVELS)}
