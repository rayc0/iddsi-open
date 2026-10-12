"""Candidate decode rules for the CORAL ordinal head, and a common scoring harness.

What this module exists to settle (measured, this lane)
-------------------------------------------------------
Wave 1 recorded that on the v0.7 **test** split (n=108) no seed emits L5 or L6
as a decision, per-level recall L5 0.00 / L6 0.00
(``lane_runs/an02_errors/analysis/structural_seed43.json``), and read that as a
representation/decode failure because argmax never tops L4 or L6 either.

Re-measured here from the five released ``best.pt`` checkpoints on both the
v0.7 validation split (n=110; L3 14 / L4 8 / L5 6 / L6 9 / L7 73) and the test
split (n=108; L3 10 / L4 11 / L5 7 / L6 10 / L7 70), the two halves of that
reading separate:

* the **shipped CORAL rank decode does produce L5 and L6 before abstention**, in
  every seed, on both splits — test pre-abstention L5 6-8 and L6 5-9 of 108.
  Only 0-3 of the 18-23 interior decodes per seed survive the shipped abstention
  threshold, because a CORAL interior class is a difference of two adjacent
  sigmoids and is capped at ``tanh(gap / 4)`` — 0.2228-0.2383 on these runs —
  while the two tail classes are unbounded.  The rank rule is not what makes
  L5/L6 unreachable; the abstention gate on decoded-level *mass* is.
* **argmax over the class posteriors produces L4 0 and L6 0 of 110** on
  validation in every seed (L5 1-3), for exactly that structural reason.  The
  "representation failure" reading is true of argmax and false of the shipped
  rank decode.

So the candidate rules below are compared on validation only, and the quantity
that decides between them is the post-abstention per-level recall together with
the dangerous-direction FNR — never the pre-abstention census alone.

Monotone ordinal structure
--------------------------
Every rule here is a *thresholded count over the cumulative probabilities*
``S_k = P(y > k)``.  Each ``S_k`` is monotone non-decreasing in the shared CORAL
score, so ``score -> #{k : S_k > t_k}`` is monotone non-decreasing for **any**
fixed vector ``t``: no thresholded-count rule can emit L5 for one image and L3
for a strictly thicker one.  ``argmax`` over the reconstructed categorical is
the one candidate that is *not* of this form; it is implemented and measured
here as rule (c) so the claim is checked rather than asserted.  The top-2 band
rule (d) is a reporting format over the same posterior vector and carries no
monotone guarantee, which the harness states rather than hides.

Confidence, and why it is not always the decoded-level mass
-----------------------------------------------------------
The shipped abstention gate thresholds on ``P(decoded level)``.  That quantity
is bounded above by ``tanh(gap / 4)`` for an interior decode and unbounded for a
tail decode, so a single threshold on it is structurally biased against L4/L5/L6
no matter where it is put.  Two confidence definitions are therefore implemented
and crossed with the decode rules:

``mass``
    ``P(decoded level)`` — the shipped definition.  Interior-capped.
``margin``
    the ordinal boundary margin ``min(S_{j-1} - t_{j-1}, t_j - S_j)`` — how far
    the cumulative probabilities sit from their own cut-points.  Not capped by
    the interior mass ceiling, and the natural confidence for a thresholded
    count.  It is not a probability, so its scale is arbitrary and only the
    val-selected threshold gives it meaning.

Safety rule
-----------
Every fitting entry point takes a ``split`` argument and refuses to run when the
name contains ``test`` (case-insensitive), reusing the guard in
:mod:`eval.abstention`.  Cut-points and thresholds are fitted on validation
only; the test split is scored with frozen parameters and is never used to
choose a rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from eval.abstention import split_is_test
from train.model import LEVELS, coral_decode, coral_interior_mass_ceiling

#: The gate the shipped calibration pipeline declares: dangerous
#: under-classification <= 5% at <= 30% abstention (>= 70% coverage).
DEFAULT_TARGET_UNDER_FNR = 0.05
DEFAULT_MAX_ABSTENTION = 0.30

#: Confidence definitions a rule may use (see the module docstring).
CONFIDENCE_MODES = ("mass", "margin", "maxprob", "bandmass")


def _refuse_if_test(split: str, what: str) -> None:
    if split_is_test(split):
        raise ValueError(
            f"refusing to {what} on split {split!r}: decode cut-points and abstention "
            "thresholds may only be fitted on validation data, never on any test split "
            "(a split whose name contains 'test')."
        )


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(LEVELS):
        raise ValueError(f"probabilities must be a [N, {len(LEVELS)}] matrix; got {probs.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("every probability row must sum to a positive value")
    return probs / row_sums


def survival_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """``P(y > k)`` for k = 0..C-2, the exact inverse of the CORAL reconstruction."""
    probs = _normalise(probabilities)
    return 1.0 - np.cumsum(probs, axis=1)[:, :-1]


def level_counts(values: Sequence[int]) -> dict[str, int]:
    array = np.asarray(values)
    return {f"L{int(LEVELS[i])}": int((array == i).sum()) for i in range(len(LEVELS))}


def broadcast_threshold(threshold: float | Sequence[float], n: int) -> np.ndarray:
    """Broadcast a scalar or per-row threshold to a length-``n`` array."""
    array = np.asarray(threshold, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n, float(array))
    if array.shape != (n,):
        raise ValueError(f"threshold must be a scalar or a length-{n} array; got shape {array.shape}")
    return array


# ---------------------------------------------------------------------------
# decoders: probability matrix -> level indices (+ band, where applicable)
# ---------------------------------------------------------------------------


def decode_thresholded_count(survival: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """``level = #{k : S_k > t_k}``.  Monotone in the shared score for any fixed ``t``."""
    survival = np.asarray(survival, dtype=np.float64)
    cutpoints = np.asarray(cutpoints, dtype=np.float64)
    if cutpoints.shape != (survival.shape[1],):
        raise ValueError(f"expected {survival.shape[1]} cut-points; got {cutpoints.shape}")
    return (survival > cutpoints[None, :]).sum(axis=1).astype(np.int64)


def decode_argmax(probabilities: np.ndarray) -> np.ndarray:
    """``level = argmax_j p_j``.  Not a thresholded count: no monotonicity guarantee."""
    probs = _normalise(probabilities)
    return probs.argmax(axis=1).astype(np.int64)


def decode_top2_band(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point decision plus the reported ordinal band from the top-2 posteriors.

    Returns ``(levels, band_lo, band_hi)``.  When the two largest posteriors sit
    on adjacent levels the band is ``[min, max]``; otherwise it collapses to the
    single most probable level, so the rule never reports a band wider than one
    level.  The point decision is the larger posterior.
    """
    probs = _normalise(probabilities)
    order = np.argsort(-probs, axis=1, kind="stable")
    i1 = order[:, 0].astype(np.int64)
    i2 = order[:, 1].astype(np.int64)
    adjacent = np.abs(i1 - i2) <= 1
    band_lo = np.where(adjacent, np.minimum(i1, i2), i1).astype(np.int64)
    band_hi = np.where(adjacent, np.maximum(i1, i2), i1).astype(np.int64)
    return i1, band_lo, band_hi


# ---------------------------------------------------------------------------
# confidences
# ---------------------------------------------------------------------------


def confidence_mass(probabilities: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """``P(decoded level)`` — the shipped definition, interior-capped by CORAL."""
    probs = _normalise(probabilities)
    return probs[np.arange(len(levels)), np.asarray(levels, dtype=np.int64)]


def confidence_margin(survival: np.ndarray, cutpoints: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Ordinal boundary margin of a thresholded-count decode.

    For a decode of level ``j`` the binding constraints are ``S_{j-1} > t_{j-1}``
    and ``S_j < t_j``.  The margin is the smaller of the two slacks, i.e. how
    far the cumulative probabilities are from flipping the decode.  Missing
    constraints (``j = 0`` has no left neighbour, ``j = C-1`` no right one) are
    dropped, so the margin is always strictly positive.
    """
    survival = np.asarray(survival, dtype=np.float64)
    cutpoints = np.asarray(cutpoints, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.int64)
    n, cuts = survival.shape
    rows = np.arange(n)
    # Left slack: S_{j-1} - t_{j-1}, defined for j >= 1.
    left = np.full(n, np.inf)
    has_left = levels >= 1
    left[has_left] = survival[rows[has_left], levels[has_left] - 1] - cutpoints[levels[has_left] - 1]
    # Right slack: t_j - S_j, defined for j <= C-2.
    right = np.full(n, np.inf)
    has_right = levels <= cuts - 1
    right[has_right] = cutpoints[levels[has_right]] - survival[rows[has_right], levels[has_right]]
    return np.minimum(left, right)


def confidence_bandmass(probabilities: np.ndarray, band_lo: np.ndarray, band_hi: np.ndarray) -> np.ndarray:
    """Total posterior mass inside the reported band — the honest confidence for a range."""
    probs = _normalise(probabilities)
    rows = np.arange(len(band_lo))
    return probs[rows, np.asarray(band_lo, dtype=np.int64)] + probs[rows, np.asarray(band_hi, dtype=np.int64)]


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


@dataclass
class DecodeOutput:
    """One rule's decode of a probability matrix.

    ``levels`` is the point decision used for recall and FNR; ``band_lo`` /
    ``band_hi`` carry the reported ordinal band (equal to ``levels`` for rules
    that emit a point).  ``confidence`` is the quantity the abstention gate
    thresholds.
    """

    levels: np.ndarray
    confidence: np.ndarray
    band_lo: np.ndarray
    band_hi: np.ndarray
    detail: dict[str, Any] = field(default_factory=dict)


class DecodeRule:
    """Interface: fit decode parameters on a calibration split, then decode."""

    name = "abstract"
    description = ""
    #: True when ``confidence`` is a CORAL decoded-level mass, and therefore
    #: bounded by ``tanh(gap / 4)`` for an interior decode.  Only then may the
    #: abstention selector clip its candidates to the reachable region.
    confidence_is_coral_mass = False

    def fit(self, probabilities, labels, *, split="val") -> "DecodeRule":
        raise NotImplementedError

    def decode(self, probabilities) -> DecodeOutput:
        raise NotImplementedError

    def parameters(self) -> dict[str, Any]:
        return {}


class RankDecodeRule(DecodeRule):
    """(a) The shipped CORAL rank decode: ``level = #{k : S_k > 0.5}``.  Control.

    ``confidence="mass"`` is exactly the shipped path.  ``confidence="margin"``
    keeps the shipped decode and changes only the quantity the abstention gate
    thresholds, which isolates the confidence definition from the decode rule.
    """

    name = "a_rank_decode_shipped"
    description = "shipped CORAL rank decode, level = #{k: P(y>k) > 0.5}"

    def __init__(self, confidence: str = "mass"):
        if confidence not in ("mass", "margin"):
            raise ValueError(f"RankDecodeRule confidence must be 'mass' or 'margin'; got {confidence!r}")
        self.confidence_mode = confidence
        self.name = "a_rank_decode_shipped" if confidence == "mass" else "a2_rank_decode_margin_conf"
        self.confidence_is_coral_mass = confidence == "mass"

    def fit(self, probabilities, labels, *, split="val"):
        _refuse_if_test(split, "fit the rank decode (no parameters)")
        return self

    def decode(self, probabilities) -> DecodeOutput:
        probs = _normalise(probabilities)
        survival = survival_from_probabilities(probs)
        cutpoints = np.full(survival.shape[1], 0.5)
        levels = decode_thresholded_count(survival, cutpoints)
        if self.confidence_mode == "mass":
            confidence = confidence_mass(probs, levels)
        else:
            confidence = confidence_margin(survival, cutpoints, levels)
        return DecodeOutput(
            levels=levels,
            confidence=np.asarray(confidence, dtype=np.float64),
            band_lo=levels,
            band_hi=levels,
            detail={"cutpoints": [float(v) for v in cutpoints], "confidence_mode": self.confidence_mode},
        )

    def parameters(self) -> dict[str, Any]:
        return {"cutpoints": 0.5, "confidence_mode": self.confidence_mode}


class CalibratedCutPointRule(DecodeRule):
    """(b) Per-level calibrated cut-points on the cumulative probabilities.

    ``level = #{k : S_k > t_k}`` with ``t`` fitted on the calibration split
    instead of the hard-coded 0.5.  The decode stays monotone for any fixed
    ``t`` because each ``S_k`` is monotone in the shared CORAL score.

    Fit: **marginal (per-level) calibration**.  For each cut-point ``k`` the
    target frequency is read off the calibration labels,
    ``q_k = mean(label > k)``, and ``t_k`` is set to the ``1 - q_k`` quantile of
    the observed ``S_k`` so that ``mean(S_k > t_k) ~ q_k``.  Each of the four
    cut-points is therefore estimated from all calibration events rather than
    from the rare level it borders, which matters here: the validation split has
    6 L5 and 9 L6 of 110, so any objective that reads a single level's recall
    off that support is fitting ~6 numbers with 4 free parameters.

    Why not a free grid search: one was implemented and measured first
    (``b3_free_grid_cutpoints`` in the comparison harness) and it overfits — the
    fitted vectors are non-monotone and unstable across seeds, and they buy
    interior recall by over-predicting one interior level while driving another
    to zero.  The marginal fit is the version worth recommending; the free fit
    is kept in the comparison so the overfitting is visible rather than
    asserted.

    ``confidence="margin"`` (the default) thresholds on the boundary margin,
    which is not capped by the CORAL interior mass ceiling.
    ``confidence="mass"`` keeps the shipped confidence so the effect of the
    cut-points alone is visible.
    """

    name = "b_calibrated_cutpoints"
    description = (
        "level = #{k: P(y>k) > t_k}, t_k set so mean(S_k > t_k) matches mean(label > k) on validation"
    )

    def __init__(
        self,
        confidence: str = "margin",
        objective: str = "marginal",
        grid: Sequence[float] | None = None,
    ):
        if confidence not in ("mass", "margin"):
            raise ValueError(f"CalibratedCutPointRule confidence must be 'mass' or 'margin'; got {confidence!r}")
        if objective not in ("marginal", "macro_recall_grid"):
            raise ValueError(f"unknown objective {objective!r}")
        self.confidence_mode = confidence
        self.objective = objective
        self.grid = np.asarray(
            grid if grid is not None else np.round(np.arange(0.05, 1.0001, 0.05), 4), dtype=np.float64
        )
        self.cutpoints: np.ndarray | None = None
        self.fit_report: dict[str, Any] = {}
        suffix = "" if objective == "marginal" else "_freegrid"
        conf_suffix = "" if confidence == "margin" else "_mass_conf"
        self.name = f"b_calibrated_cutpoints{suffix}{conf_suffix}"
        self.confidence_is_coral_mass = confidence == "mass"

    # -- fitting ----------------------------------------------------------

    def _fit_marginal(self, survival: np.ndarray, truth: np.ndarray) -> None:
        n_cuts = survival.shape[1]
        targets = np.array([(truth > k).mean() for k in range(n_cuts)], dtype=np.float64)
        cutpoints = np.empty(n_cuts, dtype=np.float64)
        for k in range(n_cuts):
            # t_k = the (1 - q_k) quantile of S_k, so the share of rows with
            # S_k > t_k equals the share of labels above k.
            cutpoints[k] = float(np.quantile(survival[:, k], 1.0 - targets[k]))
        self.cutpoints = cutpoints
        chosen = decode_thresholded_count(survival, cutpoints)
        self.fit_report = {
            "objective": "marginal per-level calibration: t_k = quantile(S_k, 1 - mean(label > k))",
            "n_calibration": int(len(truth)),
            "target_cumulative_frequencies": [float(v) for v in targets],
            "cutpoints": [float(v) for v in cutpoints],
            "calibration_decode_counts": level_counts(chosen),
            "calibration_accuracy": float((chosen == truth).mean()),
            "calibration_under_rate": float((chosen < truth).mean()),
            "calibration_macro_recall": float(
                np.mean(
                    [
                        ((chosen == level) & (truth == level)).sum() / max(int((truth == level).sum()), 1)
                        for level in np.unique(truth)
                    ]
                )
            ),
        }

    def _fit_macro_recall_grid(self, survival: np.ndarray, truth: np.ndarray) -> None:
        grid = self.grid
        mesh = np.meshgrid(*[grid] * survival.shape[1], indexing="ij")
        combos = np.stack([m.ravel() for m in mesh], axis=1)  # (G^(C-1), C-1)
        decoded = (survival[None, :, :] > combos[:, None, :]).sum(axis=2)  # (G^(C-1), N)
        present = np.unique(truth)
        hits = np.stack(
            [((decoded == level) & (truth[None, :] == level)).sum(axis=1) for level in present], axis=1
        )
        support = np.array([(truth == level).sum() for level in present], dtype=np.float64)
        macro_recall = (hits / support[None, :]).mean(axis=1)
        best = int(np.argmax(macro_recall))
        self.cutpoints = combos[best]
        chosen = decoded[best]
        self.fit_report = {
            "objective": "free grid search maximising macro per-level recall",
            "n_calibration": int(len(truth)),
            "grid_size": int(len(grid)),
            "combinations_evaluated": int(len(combos)),
            "cutpoints": [float(v) for v in self.cutpoints],
            "calibration_macro_recall": float(macro_recall[best]),
            "calibration_accuracy": float((chosen == truth).mean()),
            "calibration_under_rate": float((chosen < truth).mean()),
            "calibration_decode_counts": level_counts(chosen),
            "calibration_macro_recall_by_level": {
                f"L{int(level)}": float(hits[best, i] / support[i]) for i, level in enumerate(present)
            },
        }

    def fit(self, probabilities, labels, *, split="val"):
        _refuse_if_test(split, "fit decode cut-points")
        probs = _normalise(probabilities)
        survival = survival_from_probabilities(probs)
        truth = np.asarray(labels, dtype=np.int64)
        if self.objective == "marginal":
            self._fit_marginal(survival, truth)
        else:
            self._fit_macro_recall_grid(survival, truth)
        return self

    def decode(self, probabilities) -> DecodeOutput:
        if self.cutpoints is None:
            raise ValueError("CalibratedCutPointRule.decode called before fit")
        probs = _normalise(probabilities)
        survival = survival_from_probabilities(probs)
        levels = decode_thresholded_count(survival, self.cutpoints)
        if self.confidence_mode == "mass":
            confidence = confidence_mass(probs, levels)
        else:
            confidence = confidence_margin(survival, self.cutpoints, levels)
        return DecodeOutput(
            levels=levels,
            confidence=np.asarray(confidence, dtype=np.float64),
            band_lo=levels,
            band_hi=levels,
            detail={"cutpoints": [float(v) for v in self.cutpoints], "confidence_mode": self.confidence_mode},
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "cutpoints": None if self.cutpoints is None else [float(v) for v in self.cutpoints],
            "confidence_mode": self.confidence_mode,
        }


class ArgmaxDecodeRule(DecodeRule):
    """(c) Argmax over the five class posteriors instead of the rank rule.

    Argmax is *not* a thresholded count, so it has no monotonicity guarantee,
    and a CORAL interior class is capped at ``tanh(gap / 4)`` while the tails
    are unbounded — the structural reason it cannot reach L4/L6.  It is measured
    here rather than argued about.  ``confidence="maxprob"`` is the only
    meaningful mode.
    """

    name = "c_argmax_posteriors"
    description = "argmax over the 5 reconstructed class posteriors; confidence = max posterior"

    def __init__(self, confidence: str = "maxprob"):
        if confidence != "maxprob":
            raise ValueError("ArgmaxDecodeRule only supports confidence='maxprob'")
        self.confidence_mode = confidence

    def fit(self, probabilities, labels, *, split="val"):
        _refuse_if_test(split, "fit argmax (no parameters)")
        return self

    def decode(self, probabilities) -> DecodeOutput:
        probs = _normalise(probabilities)
        levels = decode_argmax(probs)
        return DecodeOutput(
            levels=levels,
            confidence=confidence_mass(probs, levels),
            band_lo=levels,
            band_hi=levels,
            detail={"confidence_mode": "maxprob"},
        )

    def parameters(self) -> dict[str, Any]:
        return {"confidence_mode": "maxprob"}


class Top2BandRule(DecodeRule):
    """(d) Report the top-2 ordinal band as a range when the two disagree.

    The point decision is the larger posterior; the reported band spans the two
    largest posteriors when they are adjacent and collapses to the point
    otherwise.  ``confidence="bandmass"`` is the total posterior mass inside the
    band, which is the honest confidence for a range answer (and is *not*
    capped by the CORAL interior ceiling, because a band of two adjacent levels
    adds two masses together).

    Carries no monotone guarantee: it is a reporting format over the same
    posterior vector, evaluated here so its cost is measured rather than
    assumed.
    """

    name = "d_top2_ordinal_band"
    description = "report [L_lo, L_hi] from the two largest posteriors when adjacent; confidence = band mass"

    def __init__(self, confidence: str = "bandmass"):
        if confidence != "bandmass":
            raise ValueError("Top2BandRule only supports confidence='bandmass'")
        self.confidence_mode = confidence

    def fit(self, probabilities, labels, *, split="val"):
        _refuse_if_test(split, "fit the top-2 band rule (no parameters)")
        return self

    def decode(self, probabilities) -> DecodeOutput:
        probs = _normalise(probabilities)
        levels, band_lo, band_hi = decode_top2_band(probs)
        return DecodeOutput(
            levels=levels,
            confidence=confidence_bandmass(probs, band_lo, band_hi),
            band_lo=band_lo,
            band_hi=band_hi,
            detail={"confidence_mode": "bandmass"},
        )

    def parameters(self) -> dict[str, Any]:
        return {"confidence_mode": "bandmass"}


def default_rules() -> list[DecodeRule]:
    """The candidate rules, each with its natural confidence.

    ``a2_rank_decode_margin_conf`` is included because the measurement above
    shows the shipped decode already reaches L5/L6 and it is the *confidence*
    definition that deletes them at the gate; without that variant the
    comparison cannot separate the two.  ``b3_free_grid_cutpoints`` is included
    because it was implemented and measured first and overfits — it is kept so
    that failure is visible rather than asserted.
    """
    return [
        RankDecodeRule(confidence="mass"),
        RankDecodeRule(confidence="margin"),
        CalibratedCutPointRule(confidence="margin", objective="marginal"),
        CalibratedCutPointRule(confidence="mass", objective="marginal"),
        CalibratedCutPointRule(confidence="margin", objective="macro_recall_grid"),
        ArgmaxDecodeRule(),
        Top2BandRule(),
    ]


# ---------------------------------------------------------------------------
# abstention threshold selection (validation only)
# ---------------------------------------------------------------------------


def candidate_thresholds(confidence: np.ndarray) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float64)
    return np.unique(np.concatenate(([0.0], confidence, [np.nextafter(confidence.max(), np.inf)])))


def select_abstention_threshold(
    labels: Sequence[int],
    predictions: Sequence[int],
    confidence: Sequence[float],
    *,
    split: str = "val",
    target_under_fnr: float = DEFAULT_TARGET_UNDER_FNR,
    max_abstention: float = DEFAULT_MAX_ABSTENTION,
    interior_mass_ceiling: float | None = None,
) -> dict[str, Any]:
    """Highest-coverage threshold satisfying the declared dangerous-error gate.

    Same selection semantics as
    :func:`eval.calibration.calibrate_abstention_threshold`, generalised from
    "CORAL probabilities" to any ``(predictions, confidence)`` pair so every
    candidate rule is gated identically:

    * candidates are the observed confidences (plus 0 and the supremum);
    * eligible candidates must cover at least ``1 - max_abstention``;
    * among the eligible, prefer the highest coverage whose dangerous
      under-classification rate over the covered subset is <= ``target_under_fnr``;
    * if none is, take the least under-classification among the eligible and
      report ``target_met: false`` — the gate is never silently satisfied.

    ``interior_mass_ceiling``, when given, clips the candidates to the region
    where an interior decode can survive at all.  This is the H-CORAL
    reachability constraint from :mod:`eval.calibration`: a threshold above
    ``tanh(gap / 4)`` abstains on *every* interior decode, so the gate could
    otherwise be "satisfied" by deleting L4/L5/L6 from the output entirely.  It
    is passed only for rules whose confidence is a CORAL interior-bounded mass.
    """

    _refuse_if_test(split, "select an abstention threshold")
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    conf = np.asarray(confidence, dtype=np.float64)
    if not (len(truth) == len(pred) == len(conf)) or not len(truth):
        raise ValueError("labels, predictions and confidence must be non-empty and equal length")
    candidates = candidate_thresholds(conf)
    clipped = False
    if interior_mass_ceiling is not None:
        reachable = candidates[candidates <= interior_mass_ceiling + 1e-12]
        if reachable.size:
            candidates = reachable
            clipped = True
    min_coverage = 1.0 - float(max_abstention)
    records = []
    for threshold in candidates:
        accepted = conf >= threshold
        coverage = float(accepted.mean())
        if not accepted.any():
            under_fnr = error_rate = float("nan")
        else:
            under_fnr = float((pred[accepted] < truth[accepted]).mean())
            error_rate = float((pred[accepted] != truth[accepted]).mean())
        records.append(
            {"threshold": float(threshold), "coverage": coverage, "under_fnr": under_fnr, "error_rate": error_rate}
        )
    eligible = [r for r in records if r["coverage"] + 1e-12 >= min_coverage]
    feasible = [r for r in eligible if np.isfinite(r["under_fnr"]) and r["under_fnr"] <= target_under_fnr + 1e-12]
    if feasible:
        chosen = max(feasible, key=lambda r: (r["coverage"], -r["threshold"]))
        target_met = True
        reason = "highest coverage satisfying the configured validation under-classification gate"
    elif eligible:
        chosen = min(eligible, key=lambda r: (r["under_fnr"], r["error_rate"], -r["coverage"]))
        target_met = False
        reason = "gate not met; least under-classification risk among candidates meeting minimum coverage"
    else:
        chosen = min(records, key=lambda r: -r["coverage"])
        target_met = False
        reason = "gate not met and no candidate reaches the minimum coverage"
    return {
        **chosen,
        "target_under_fnr": float(target_under_fnr),
        "minimum_coverage": float(min_coverage),
        "target_met": bool(target_met),
        "selection_reason": reason,
        "candidates_clipped_to_interior_ceiling": bool(clipped),
        "interior_mass_ceiling": None if interior_mass_ceiling is None else float(interior_mass_ceiling),
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_decode(
    labels: Sequence[int],
    output: DecodeOutput,
    threshold: float | Sequence[float],
    *,
    probabilities: np.ndarray | None = None,
    rule_name: str = "",
) -> dict[str, Any]:
    """Score one decode at one abstention threshold.

    Pure: no split check, mirroring :func:`eval.abstention.risk_coverage_row`.
    The caller is responsible for having frozen ``threshold`` (and any decode
    cut-points) on validation data; the fitting entry points in this module
    refuse to run on a split whose name contains ``test``.

    ``threshold`` may be a scalar or a per-row array.  The per-row form is what
    an out-of-fold protocol needs: each fold's threshold is selected on the
    other folds, so the pooled out-of-fold predictions are gated by a different
    threshold per row and a single scalar would be wrong.

    Denominators are stated explicitly everywhere: per-level recall is reported
    as ``correct / support`` with both integers, and the dangerous-direction
    quantities are reported three ways because they have three different
    denominators (H-ABSTBLIND):

    * ``dangerous_underclassification_fnr_covered`` — dangerous errors over the
      covered subset.  This is the gate-facing number (``fnr_covered`` in the
      hazard log); its denominator shrinks as coverage shrinks.
    * ``dangerous_underclassification_rate_all`` — dangerous errors over ALL
      rows, numerator gated on acceptance.  This is the abstention-blind
      variant: it improves when the model declines to answer, so it is reported
      for completeness and never used as the gate.
    * ``dangerous_fnr_detector`` — dangerous errors that slip through, over all
      dangerous errors regardless of acceptance.  The abstention-blind number
      and this one move in opposite directions, so both are shown.
    """

    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(output.levels, dtype=np.int64)
    conf = np.asarray(output.confidence, dtype=np.float64)
    n = int(len(truth))
    thresholds = np.asarray(broadcast_threshold(threshold, n), dtype=np.float64)
    accepted = conf >= thresholds

    per_level: dict[str, Any] = {}
    for index, level in enumerate(LEVELS):
        support = int((truth == index).sum())
        hit = int(((truth == index) & (pred == index) & accepted).sum())
        hit_pre = int(((truth == index) & (pred == index)).sum())
        predicted = int(((pred == index) & accepted).sum())
        per_level[f"L{int(level)}"] = {
            "support": support,
            "recall_post_abstention": (hit / support) if support else float("nan"),
            "recall_post_abstention_count": f"{hit}/{support}",
            "recall_pre_abstention": (hit_pre / support) if support else float("nan"),
            "recall_pre_abstention_count": f"{hit_pre}/{support}",
            "predicted_post_abstention": predicted,
            "precision_post_abstention": (hit / predicted) if predicted else float("nan"),
        }

    covered_true = truth[accepted]
    covered_pred = pred[accepted]
    dangerous = pred < truth
    dangerous_total = int(dangerous.sum())
    dangerous_covered = int((dangerous & accepted).sum())
    kappa = (
        float(cohen_kappa_score(covered_true, covered_pred, weights="quadratic"))
        if len(covered_true) > 1 and len(np.unique(covered_true)) > 1
        else float("nan")
    )
    macro_labels = sorted(set(int(v) for v in truth))
    predictions_with_abstain = np.where(accepted, pred, -1)
    band_lo = np.asarray(output.band_lo, dtype=np.int64)
    band_hi = np.asarray(output.band_hi, dtype=np.int64)
    in_band = (truth >= band_lo) & (truth <= band_hi)

    result = {
        "rule": rule_name,
        "n": n,
        "threshold": (float(threshold) if np.ndim(threshold) == 0 else [float(v) for v in np.ravel(threshold)]),
        "coverage": float(accepted.mean()),
        "accepted": int(accepted.sum()),
        "abstained": int((~accepted).sum()),
        "decoded_level_counts_pre_abstention": level_counts(pred),
        "decided_level_counts_post_abstention": level_counts(predictions_with_abstain[predictions_with_abstain >= 0]),
        "per_level_recall": per_level,
        "macro_recall_post_abstention": float(
            np.mean([v["recall_post_abstention"] for v in per_level.values() if np.isfinite(v["recall_post_abstention"])])
        ),
        "accuracy_covered": float((covered_pred == covered_true).mean()) if accepted.any() else float("nan"),
        "accuracy_all_rows": float((predictions_with_abstain == truth).mean()),
        "macro_f1_with_abstentions": float(
            f1_score(truth, predictions_with_abstain, labels=macro_labels, average="macro", zero_division=0)
        ),
        "weighted_kappa_covered": kappa,
        "dangerous_underclassification_fnr_covered": (
            float((covered_pred < covered_true).mean()) if accepted.any() else float("nan")
        ),
        "dangerous_underclassification_rate_all": float((dangerous & accepted).mean()),
        "dangerous_fnr_detector": (
            float(dangerous_covered / dangerous_total) if dangerous_total else float("nan")
        ),
        "dangerous_errors_total": dangerous_total,
        "dangerous_errors_covered": dangerous_covered,
        "overclassification_rate_covered": (
            float((covered_pred > covered_true).mean()) if accepted.any() else float("nan")
        ),
        "band_contains_truth_rate": float(in_band.mean()),
        "mean_band_width": float((band_hi - band_lo).mean()),
        "interior_levels_reachable": bool(((pred >= 1) & (pred <= len(LEVELS) - 2) & accepted).any()),
    }
    if probabilities is not None:
        probs = _normalise(probabilities)
        ceiling = coral_interior_mass_ceiling(probs)
        interior = (pred >= 1) & (pred <= len(LEVELS) - 2)
        result["interior_decodes_total"] = int(interior.sum())
        result["interior_decodes_accepted"] = int((interior & accepted).sum())
        result["interior_mass_ceiling"] = None if ceiling is None else float(ceiling)
        result["max_interior_confidence"] = float(conf[interior].max()) if interior.any() else float("nan")
    return result
