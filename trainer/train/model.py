from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = (3, 4, 5, 6, 7)

#: The decode rules :func:`coral_decode` can apply.  ``"rank"`` is the shipped
#: H-CORAL fix and the default; ``"argmax"`` is the pre-H-CORAL categorical
#: argmax, kept selectable so the two can be measured on identical weights
#: instead of argued about.
DECODE_RULES = ("rank", "argmax")


def _check_decode_rule(rule: str) -> str:
    if rule not in DECODE_RULES:
        raise ValueError(f"decode rule must be one of {DECODE_RULES}; got {rule!r}")
    return rule


def coral_decode_argmax(probabilities: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Decode by argmax over the reconstructed categorical distribution.

    This is the decode H-CORAL replaced.  It is kept as a selectable rule
    because it is the only way to measure the difference on identical weights:
    ``coral_logits_to_probs`` builds interior classes as differences of
    adjacent sigmoids, so their mass is capped at ``tanh(gap / 4)`` while the
    two tail classes are unbounded (see :func:`coral_interior_mass_ceiling`).
    With the cut-point spacing the head actually learns (~0.91 on the five v0.7
    release seeds) that cap is ~0.22, so argmax returns an interior level only
    when BOTH tails fall below it — which the calibrated v0.7 probabilities
    never do.  Measured on the five v0.7 release seeds' held-out predictions,
    argmax emits 0 interior levels in all five while the rank rule emits
    18-23.

    Returns ``(level_index, confidence)`` where confidence is the argmax mass.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a [N, C] matrix")
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    predictions = probs.argmax(axis=1).astype(np.int64)
    confidence = probs[np.arange(len(predictions)), predictions]
    return predictions, confidence


def coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert monotone P(y > k) logits into probabilities for L3..L7."""
    survival = torch.sigmoid(logits)
    probs = torch.cat(
        [1.0 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]], dim=1
    )
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def coral_decode(
    probabilities: "np.ndarray", rule: str = "rank"
) -> tuple["np.ndarray", "np.ndarray"]:
    """Decode CORAL probabilities into ``(level_index, confidence)``.

    ``rule`` selects the decode.  The default, ``"rank"``, is the ordinal rank
    rule ``level = #{k : P(y > k) > 0.5}`` (Cao, Mirjalili & Raschka) and is
    what the release ships.  ``"argmax"`` selects
    :func:`coral_decode_argmax`, the pre-H-CORAL categorical argmax, and exists
    only so the two decodes can be compared on identical weights.

    The rank rule is the default and the argmax alternative is NOT a
    replacement for it: see the module docstring notes on
    :func:`coral_decode_argmax` for why argmax cannot express an interior level
    at the cut-point spacing the head learns.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a [N, C] matrix")
    rule = _check_decode_rule(rule)
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    if rule == "argmax":
        return coral_decode_argmax(probs)
    # P(y > k) for k = 0 .. C-2, recovered from the categorical form.
    survival = 1.0 - np.cumsum(probs, axis=1)[:, :-1]
    predictions = (survival > 0.5).sum(axis=1).astype(np.int64)
    confidence = probs[np.arange(len(predictions)), predictions]
    return predictions, confidence


def coral_interior_mass_ceiling(
    probabilities: "np.ndarray", *, tolerance: float = 1e-2
) -> "float | None":
    """Largest class mass any interior level (L4/L5/L6) can ever reach.

    An interior class is a *difference of two adjacent sigmoids* whose logits
    are ``gap`` apart, so its mass is bounded by ``tanh(gap / 4)`` — the
    supremum of ``sigmoid(x) - sigmoid(x - gap)`` over ``x``, attained at
    ``x = gap / 2``.  The two tail classes carry no such bound.  This is the
    same ceiling that made ``argmax`` blind to L4/L5/L6; it still bounds how
    confident the model is allowed to *look* on an interior level, so anything
    that thresholds on class mass — the abstention gate — has to know it.  A
    threshold above this ceiling makes every interior decode abstain, which
    reproduces the H-CORAL symptom through the abstention gate instead of
    through argmax.

    ``gap`` is recovered from the probabilities alone, because that is all a
    threshold selector is given: ``survival[:, j] = 1 - cumsum(probs)[:, j]``
    is ``sigmoid((s + b_j) / T)``, so ``logit(survival[:, j]) -
    logit(survival[:, j+1])`` is the *temperature-scaled* gap and is identical
    in every row — the shared score ``s`` cancels.  The median over rows is
    used so a single saturated row cannot move it.

    Returns ``None`` when the rows do not share one gap vector, i.e. when
    ``probabilities`` is not a CORAL reconstruction (a softmax matrix, say)
    and the bound does not apply.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 3 or len(probs) < 2:
        return None
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    survival = 1.0 - np.cumsum(probs, axis=1)[:, :-1]
    if not np.isfinite(survival).all():
        return None
    clipped = survival.clip(1e-12, 1.0 - 1e-12)
    logged = np.log(clipped) - np.log1p(-clipped)
    differences = logged[:, :-1] - logged[:, 1:]
    # A logit is only well conditioned away from 0 and 1; the float32
    # probabilities the head emits have already lost the tail digits there.
    # Each gap is therefore estimated from the rows that condition *that* gap,
    # and only when every such row agrees is the matrix accepted as CORAL.
    well_conditioned = (survival > 1e-3) & (survival < 1.0 - 1e-3)
    gaps = np.empty(differences.shape[1], dtype=np.float64)
    for index in range(differences.shape[1]):
        usable = (
            np.isfinite(differences[:, index])
            & well_conditioned[:, index]
            & well_conditioned[:, index + 1]
        )
        if usable.sum() < 2:
            return None
        column = differences[usable, index]
        gap = float(np.median(column))
        if gap <= 0.0 or np.ptp(column) / max(abs(gap), 1e-9) > tolerance:
            return None  # not a CORAL reconstruction: the gaps are not shared
        gaps[index] = gap
    return float(np.tanh(gaps.min() / 4.0))


class OrderedCoralHead(nn.Module):
    """CORAL-style shared projection with strictly ordered learned cut-points."""

    def __init__(self, in_features: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(in_features, 1, bias=False)
        self.first_bias = nn.Parameter(torch.tensor(1.5))
        # softplus(0)=0.693 gave interior classes a ~0.17 probability ceiling.
        # softplus(1.5)=1.70 lifts it to ~0.40 so every level is reachable.
        # The ceiling is tanh(gap/4) — see coral_interior_mass_ceiling.  It
        # bounds not only argmax but also any confidence threshold applied to
        # class mass, so it is reported and enforced downstream.
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
