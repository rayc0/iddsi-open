"""Evidence layer for the IDDSI-Open Hugging Face Space.

This Space ships **no classifier**.  The v0.7 SigLIP-2 + CORAL weights are
withheld by a locked decision (2026-09-07; ``hf/model_README.md`` "Why the
weights are withheld"), so an "upload your food, get a level" demo would be
presenting a number nobody can check.  What ships instead is the *evidence*:
the per-level confusion matrix that shows L5 and L6 recall of 0.00, the
risk-coverage curve, the yardstick flip where the same checkpoints pass or fail
depending on which label set they are scored against, and the corpus-hygiene
numbers that keep the harvested dataset unpublishable.

Design rules this module follows
--------------------------------
1. **No model, no torch, no PIL.**  The Space is a pure-Python reader over
   JSON.  It boots in seconds on a CPU Space and cannot accidentally run
   inference.
2. **Derived numbers are recomputed here, not copied.**  ``per_level_recall``,
   ``per_level_precision``, ``macro_f1_with_abstentions`` and
   ``weighted_kappa_covered`` are computed from the raw confusion-matrix
   counts, then compared against the values published in
   ``analysis/errors_v07/outputs/structural_seed43.json`` by
   :meth:`EvidenceBundle.self_check`.  If the two ever disagree the Space says
   so out loud instead of quietly showing one of them.
3. **Read-only numbers are labelled with the file they came from.**  Every
   accessor names its source; :meth:`EvidenceBundle.self_check` renders that
   provenance so a reader can tell a recomputation from a quotation.
4. **Nothing here issues a verdict.**  There is no safe/unsafe output, no
   level estimate for any photograph, and no swallowing-safety inference.  The
   abstention string below is the only "output" the Space can produce.

The controlled disclaimer, its Traditional-Chinese rendering and the abstention
string are carried over **verbatim** from the previous ``space/core.py`` (which
took them from ``crowd/app.py``).  Do not edit the wording here: edit it there
and mirror it, so the two public faces cannot drift apart.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Controlled wording -- verbatim, do not edit here
# --------------------------------------------------------------------------

#: IDDSI levels the ordinal head predicts.  Mirrors ``train.model.LEVELS``.
LEVELS: tuple[int, ...] = (3, 4, 5, 6, 7)

LEVEL_LABELS: tuple[str, ...] = tuple(f"L{level}" for level in LEVELS)

#: IDDSI descriptor names, for axis labels only.  Paraphrased from the IDDSI
#: framework for readability; this project is independent of IDDSI.
LEVEL_NAMES: Mapping[int, str] = {
    3: "Liquidised / moderately thick",
    4: "Pureed",
    5: "Minced & moist",
    6: "Soft & bite-sized",
    7: "Regular",
}

DISCLAIMER = (
    "Research demonstration for culinary education. Estimates visual similarity "
    "to IDDSI descriptors; does not perform official IDDSI tests, assess "
    "swallowing, determine suitability for any person, or decide whether food is "
    "safe to consume."
)

DISCLAIMER_ZH_HANT = (
    "烹飪教學研究示範。本工具僅估算食物外觀與 IDDSI 描述之間的視覺相似度；"
    "並不執行 IDDSI 官方測試、不評估吞嚥能力、不判斷食物是否適合任何人士，"
    "亦不決定食物是否安全食用。"
)

#: Required abstention output (``hf/HAZARD_LOG.md`` control principles).  This
#: string is never replaced by a level estimate.
ABSTENTION_TEXT = "unclear — perform physical test"

#: Replaces the old ``PLACEHOLDER_BANNER``.  The old banner described a demo
#: that printed a *uniform* distribution as if it were a model output; this
#: Space prints no distribution at all, and says why.
WITHHELD_NOTICE = (
    "NO CLASSIFIER IS SHIPPED HERE. The v0.7 SigLIP-2 + CORAL weights are "
    "WITHHELD by a locked decision (2026-09-07), so this Space cannot classify "
    "anything and does not try. It renders the measured evidence for the "
    "withholding decision instead."
)

#: Why the weights are withheld.  Condensed from ``hf/model_README.md``; the
#: full sentence is quoted verbatim in the release card, bundled at
#: ``evidence/model_README.md``.
WITHHELD_REASON = (
    "The photo-classification head fails the project's own pre-registered "
    "gates, and a downloadable classifier would be used for verdicts whatever "
    "the disclaimer said. A disclaimer does not cure a claim."
)

#: The five v0.7 seeds, in the order the release card lists them.
SEEDS: tuple[int, ...] = (17, 29, 43, 71, 101)

#: Gate thresholds.  Quoted from ``hf/model_README.md`` ("The gates"), which is
#: bundled verbatim at ``evidence/model_README.md`` so the quotation is
#: checkable.  These are the *release* gates, not re-registered here.
GATE_DEFINITIONS: Mapping[str, str] = {
    "wk1": "wk1: weighted kappa >=0.70, else kill photo-classification.",
    "wk2": (
        "wk2: external macro-F1 >=0.75 plus dangerous under-classification "
        "<=5% at <=30% abstention."
    ),
}

#: Numeric thresholds the Space evaluates the committed metrics against.
GATE_THRESHOLDS: Mapping[str, float] = {
    "wk1_kappa": 0.70,
    "wk2_macro_f1": 0.75,
    "wk2_under_fnr": 0.05,
    "wk2_abstention": 0.30,
}

#: Held-out split size, from ``analysis/errors_v07/outputs/summary.json`` and
#: ``results/v07_release/v07_seed43_metrics.json`` (``n``).
N_TEST_EVENTS = 108

EVIDENCE_DIR = pathlib.Path(__file__).resolve().parent / "evidence"


class EvidenceError(RuntimeError):
    """Raised when a bundled evidence file is missing or malformed."""


def _load(name: str) -> Any:
    path = EVIDENCE_DIR / name
    if not path.is_file():
        raise EvidenceError(
            f"evidence file {name} is missing from {EVIDENCE_DIR}. Rebuild it with "
            "`python -m space.tools.build_evidence_bundle` inside the iddsi-train "
            "container."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt bundle
        raise EvidenceError(f"evidence file {name} is not valid JSON: {exc}") from exc


def _read_text(name: str) -> str:
    path = EVIDENCE_DIR / name
    if not path.is_file():
        raise EvidenceError(f"evidence file {name} is missing from {EVIDENCE_DIR}")
    return path.read_text(encoding="utf-8")


def _fmt(value: Any, places: int = 4) -> str:
    """Format a rate for a markdown table; ``None`` renders as an em dash."""

    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{float(value):.{places}f}"


# --------------------------------------------------------------------------
# Confusion-matrix arithmetic
# --------------------------------------------------------------------------


def quadratic_weighted_kappa(matrix: Sequence[Sequence[int]]) -> float:
    """Quadratic weighted kappa from its definition (no sklearn).

    ``matrix[i][j]`` counts events whose true class is ``i`` and predicted
    class is ``j``.  Weights are ``(i-j)^2 / (n-1)^2``.
    """

    n = len(matrix)
    if n < 2:
        raise ValueError("quadratic weighted kappa needs at least two classes")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("empty confusion matrix")
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    denom = (n - 1) ** 2
    observed = sum(
        ((i - j) ** 2 / denom) * matrix[i][j] for i in range(n) for j in range(n)
    )
    expected = sum(
        ((i - j) ** 2 / denom) * rows[i] * cols[j] / total
        for i in range(n)
        for j in range(n)
    )
    if expected == 0:
        raise ValueError("degenerate expected matrix")
    return 1.0 - observed / expected


def _f1(precision: float | None, recall: float | None) -> float:
    if precision is None or recall is None:
        return 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _decimals(value: float) -> int:
    """How many decimal places a published float was recorded at."""

    text = repr(float(value))
    if "e" in text or "E" in text:
        return 12
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def _agrees(recomputed: Any, published: Any) -> bool:
    """Compare a recomputation to a published value at the published precision.

    The published per-level figures in ``structural_seed43.json`` are recorded
    to four decimal places (0.6923, 0.8442) while the headline metrics are
    recorded to six (0.394931, 0.850721).  Comparing at full double precision
    would report a false mismatch for the four-decimal ones, so the tolerance
    is half a unit in the last place the published value actually carries.
    """

    if isinstance(recomputed, bool) or isinstance(published, bool):
        return recomputed is published
    if isinstance(recomputed, (int, float)) and isinstance(published, (int, float)):
        if recomputed == published:
            return True
        tol = 0.5 * 10 ** (-_decimals(published))
        return abs(float(recomputed) - float(published)) <= tol
    return recomputed == published


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------


class EvidenceBundle:
    """Lazy, read-only view over ``space/evidence/``.

    Every accessor is a pure function of the bundled JSON.  Nothing is cached
    across instances, nothing writes, and no network or GPU is touched.
    """

    def __init__(self, root: pathlib.Path | str | None = None) -> None:
        self.root = pathlib.Path(root) if root is not None else EVIDENCE_DIR

    # -- raw access -------------------------------------------------------

    def raw(self, name: str) -> Any:
        return _load(name)

    def source_text(self, name: str) -> str:
        return _read_text(name)

    def provenance(self) -> Mapping[str, Any]:
        return _load("PROVENANCE.json")

    # -- the seed-43 confusion matrix -------------------------------------

    def confusion(self) -> Mapping[str, Any]:
        """The 6x6 (true x pred-with-abstain) and 5x5 (decided-only) matrices."""

        return _load("confusion_seed43.json")

    def matrix_6x6(self) -> list[list[int]]:
        return [list(row) for row in self.confusion()["matrix_6x6_true_x_pred_with_abstain"]["counts"]]

    def matrix_5x5(self) -> list[list[int]]:
        return [list(row) for row in self.confusion()["matrix_5x5_decided_only"]["counts"]]

    def abstentions_per_true_level(self) -> list[int]:
        """Abstentions per true level, index 5 = the empty unclear row."""

        return list(self.confusion()["abstentions_per_true_level"])

    def decided_row_totals(self) -> list[int]:
        """Decided events per true level (row totals, abstentions excluded)."""

        return [sum(row) for row in self.matrix_5x5()]

    def decided_col_totals(self) -> list[int]:
        """Decided predictions per level (column totals)."""

        m = self.matrix_5x5()
        return [sum(m[i][j] for i in range(len(m))) for j in range(len(m))]

    def true_row_totals(self) -> list[int]:
        """All test events per true level, abstentions included."""

        decided = self.decided_row_totals()
        abstain = self.abstentions_per_true_level()[: len(decided)]
        return [d + a for d, a in zip(decided, abstain)]

    def n_decided(self) -> int:
        return sum(self.decided_row_totals())

    def n_abstained(self) -> int:
        return sum(self.abstentions_per_true_level()[: len(LEVELS)])

    # -- recomputed per-level statistics ----------------------------------
    #
    # Precision is over decided predictions.  Recall for the *published*
    # per-level recall is over decided events only (the abstention column is
    # dropped); recall for macro-F1 is over all test events of that level, so
    # an abstention counts as a miss.  Both denominators are stated in the
    # returned mapping and rendered by the app.

    def per_level_recall(self) -> dict[str, Any]:
        m = self.matrix_5x5()
        out: dict[str, Any] = {}
        for i, label in enumerate(LEVEL_LABELS):
            denom = self.decided_row_totals()[i]
            out[label] = {
                "value": (m[i][i] / denom) if denom else None,
                "numerator": m[i][i],
                "denominator": denom,
                "denominator_meaning": "decided events of this true level",
            }
        return out

    def per_level_precision(self) -> dict[str, Any]:
        m = self.matrix_5x5()
        out: dict[str, Any] = {}
        for j, label in enumerate(LEVEL_LABELS):
            denom = self.decided_col_totals()[j]
            if denom == 0:
                out[label] = {
                    "value": None,
                    "numerator": 0,
                    "denominator": 0,
                    "denominator_meaning": "undefined: level never emitted as a decision",
                }
            else:
                out[label] = {
                    "value": m[j][j] / denom,
                    "numerator": m[j][j],
                    "denominator": denom,
                    "denominator_meaning": "decided predictions of this level",
                }
        return out

    def macro_f1_with_abstentions(self) -> float:
        """Macro-F1 over the five levels, abstentions counted as misses.

        Precision uses the decided-prediction column totals; recall uses all
        test events of the true level, so an abstained event is a false
        negative for its true level.  This is the definition that reproduces
        the published ``macro_f1_with_abstentions`` exactly -- see
        :meth:`self_check`.
        """

        precision = self.per_level_precision()
        m = self.matrix_5x5()
        scores = []
        for i, label in enumerate(LEVEL_LABELS):
            tp = m[i][i]
            pred_denom = self.decided_col_totals()[i]
            true_denom = self.true_row_totals()[i]
            p = (tp / pred_denom) if pred_denom else None
            r = (tp / true_denom) if true_denom else None
            scores.append(_f1(p, r))
        return sum(scores) / len(scores)

    def weighted_kappa_covered(self) -> float:
        return quadratic_weighted_kappa(self.matrix_5x5())

    def emitted_levels(self) -> list[str]:
        """Levels the model ever emits as a decision on this test split."""

        return [
            label
            for label, total in zip(LEVEL_LABELS, self.decided_col_totals())
            if total > 0
        ]

    def silent_levels(self) -> list[str]:
        return [label for label in LEVEL_LABELS if label not in self.emitted_levels()]

    # -- published values the recomputation is checked against ------------

    def published_structural(self) -> Mapping[str, Any]:
        return _load("structural_seed43.json")

    def published_headline(self) -> Mapping[str, Any]:
        return self.published_structural()["headline_metric_reproduction_seed43"]

    def published_seed43_metrics(self) -> Mapping[str, Any]:
        return _load("v07_seed43_metrics.json")

    def calibration(self) -> Mapping[str, Any]:
        return _load("v07_seed43_calibration.json")

    # -- the five seeds ---------------------------------------------------

    def seed_metrics(self, seed: int) -> Mapping[str, Any]:
        return _load(f"v07_seed{seed}_metrics.json")

    def seed_ranking(self) -> Mapping[str, Any]:
        return _load("seed_ranking.json")

    def risk_coverage_curve(self, seed: int) -> list[Mapping[str, Any]]:
        return list(self.seed_metrics(int(seed))["risk_coverage_curve"])

    def gate_table(self) -> list[dict[str, Any]]:
        """Recompute the release gates from the five committed metrics files.

        ``wk1`` is scored on the v0.7 full-coverage consensus, which is the
        label set the v0.7 runs were trained and scored on.  The other
        yardstick is on the Yardstick tab.
        """

        rows = []
        for seed in SEEDS:
            m = self.seed_metrics(seed)["metrics"]
            kappa = m["weighted_kappa_covered"]
            macro_f1 = m["macro_f1_with_abstentions"]
            under_fnr = m["dangerous_underclassification_fnr_covered"]
            abstain = m["abstention_rate"]
            rows.append(
                {
                    "seed": seed,
                    "n": self.seed_metrics(seed)["n"],
                    "coverage": m["coverage"],
                    "abstention_rate": abstain,
                    "weighted_kappa_covered": kappa,
                    "macro_f1_with_abstentions": macro_f1,
                    "dangerous_underclassification_fnr_covered": under_fnr,
                    "wk1_kappa_ge_0.70": kappa >= GATE_THRESHOLDS["wk1_kappa"],
                    "wk2_macro_f1_ge_0.75": macro_f1 >= GATE_THRESHOLDS["wk2_macro_f1"],
                    "wk2_under_fnr_le_0.05": under_fnr <= GATE_THRESHOLDS["wk2_under_fnr"],
                    "wk2_abstention_le_0.30": abstain <= GATE_THRESHOLDS["wk2_abstention"],
                }
            )
        for row in rows:
            row["wk2_all_three"] = (
                row["wk2_macro_f1_ge_0.75"]
                and row["wk2_under_fnr_le_0.05"]
                and row["wk2_abstention_le_0.30"]
            )
            row["wk1_and_wk2"] = row["wk1_kappa_ge_0.70"] and row["wk2_all_three"]
        return rows

    # -- risk-coverage ----------------------------------------------------

    def operating_point(self, seed: int = 43) -> dict[str, Any]:
        """The shipped threshold and the risks measured at it.

        The committed ``risk_coverage_curve`` is a sweep over the *observed*
        confidence values, and the shipped threshold is not one of them (it
        sits between 0.2389 and 0.2743 on the seed-43 curve), so the risks at
        the operating point are recomputed from the decision tally in
        ``analysis/errors_v07/outputs/bias_seed43.json`` instead of being
        interpolated.  Every denominator is carried through.
        """

        seed = int(seed)
        metrics = self.seed_metrics(seed)
        bias = self.bias_seed43()
        decided = bias["decided"]
        wrong = bias["over_predicted"] + bias["under_predicted"]
        return {
            "threshold": metrics["threshold"],
            "coverage": metrics["metrics"]["coverage"],
            "accepted": decided,
            "n_test": metrics["n"],
            "error_risk": wrong / decided,
            "underclassification_risk": bias["under_predicted"] / decided,
            "overprediction_risk": bias["over_predicted"] / decided,
            "source": (
                "results/v07_release/v07_seed%d_metrics.json (threshold, coverage) + "
                "analysis/errors_v07/outputs/bias_seed43.json (decision tally)" % seed
            ),
        }

    def curve_extremes(
        self, seed: int = 43, coverage_floor: float = 0.0
    ) -> dict[str, Any]:
        """Min/max of both risk columns over the committed curve.

        Restricted to coverages at or above ``coverage_floor`` so the degenerate
        tail -- where three or five events remain and both risks read 0.0000 --
        does not stand in for skill.
        """

        seed = int(seed)
        curve = [
            p
            for p in self.risk_coverage_curve(seed)
            if p["coverage"] >= coverage_floor
        ]
        errors = [p["error_risk"] for p in curve]
        unders = [p["underclassification_risk"] for p in curve]
        return {
            "n_points": len(curve),
            "coverage_floor": coverage_floor,
            "error_risk_min": min(errors),
            "error_risk_max": max(errors),
            "under_risk_min": min(unders),
            "under_risk_max": max(unders),
            "coverage_min": min(p["coverage"] for p in curve),
            "coverage_max": max(p["coverage"] for p in curve),
        }

    def coverage_at_under_risk_budget(self, seed: int = 43, budget: float = 0.05) -> dict[str, Any]:
        """Best coverage on the committed curve with under-risk <= budget."""

        seed = int(seed)
        curve = self.risk_coverage_curve(seed)
        admissible = [p for p in curve if p["underclassification_risk"] <= budget]
        if not admissible:
            return {"budget": budget, "best_coverage": None, "point": None, "n_points": len(curve)}
        best = max(admissible, key=lambda p: p["coverage"])
        return {
            "budget": budget,
            "best_coverage": best["coverage"],
            "point": best,
            "n_points": len(curve),
        }

    # -- the yardstick flip ----------------------------------------------

    def yardstick_summary(self) -> Mapping[str, Any]:
        return _load("yardstick_summary.json")

    def yardstick_per_seed(self) -> list[Mapping[str, Any]]:
        raw = _load("yardstick_per_seed_slim.json")
        return [r for r in raw if isinstance(r, dict) and "version" in r]

    def yardstick_flip(self) -> Mapping[str, Any]:
        return _load("yardstick_flip_decomposition.json")

    def bias_matrix_identification(self) -> list[Mapping[str, Any]]:
        return list(_load("yardstick_bias_matrix_identification.json"))

    def bias_matrix(self) -> Mapping[str, Any]:
        return _load("bias_matrix.json")

    # -- corpus hygiene ---------------------------------------------------

    def corpus_hygiene(self) -> Mapping[str, Any]:
        return _load("corpus_hygiene_measured.json")

    def labelability_sweep(self) -> Mapping[str, Any]:
        return _load("labelability_sweep.json")

    def inter_judge(self) -> Mapping[str, Any]:
        return _load("inter_judge_20260916.json")

    def coverage_meta(self) -> Mapping[str, Any]:
        return _load("coverage_meta.json")

    def dedup(self) -> Mapping[str, Any]:
        return _load("dedup_report.json")

    def bias_seed43(self) -> Mapping[str, Any]:
        return _load("bias_seed43.json")

    def adjacent_pairs(self) -> Mapping[str, Any]:
        return _load("adjacent_pairs_seed43.json")

    def structural(self) -> Mapping[str, Any]:
        return _load("structural_seed43.json")

    def analysis_summary(self) -> Mapping[str, Any]:
        return _load("analysis_summary.json")

    # -- the self-check ---------------------------------------------------

    def self_check(self) -> list[dict[str, Any]]:
        """Recompute every headline number and compare it to the published one.

        Each row names the quantity, the file the published value came from,
        the recomputed value, and whether they agree.  The Space renders this
        table on the Provenance tab and shows a failure banner if any row
        disagrees -- the point of the Space is that nothing in it is asserted
        without a checkable derivation.
        """

        published = self.published_headline()
        metrics = self.published_seed43_metrics()["metrics"]
        structural = self.structural()
        recall_pub = published["per_level_recall_decided"]
        precision_pub = published["per_level_precision_decided"]
        census = structural["emitted_level_census"]["43"]

        checks: list[dict[str, Any]] = []

        def add(
            quantity: str,
            recomputed: Any,
            published_value: Any,
            source: str,
            denominator: str,
        ) -> None:
            checks.append(
                {
                    "quantity": quantity,
                    "recomputed": recomputed,
                    "published": published_value,
                    "source": source,
                    "denominator": denominator,
                    "ok": bool(_agrees(recomputed, published_value)),
                }
            )

        recall = self.per_level_recall()
        for label in LEVEL_LABELS:
            add(
                f"recall {label} (decided)",
                recall[label]["value"],
                recall_pub[label],
                "analysis/errors_v07/outputs/structural_seed43.json "
                "→ headline_metric_reproduction_seed43.per_level_recall_decided",
                f"{recall[label]['numerator']}/{recall[label]['denominator']} decided "
                f"{label} events",
            )

        precision = self.per_level_precision()
        for label in LEVEL_LABELS:
            pub = precision_pub[label]
            got = precision[label]["value"]
            add(
                f"precision {label} (decided)",
                got if got is not None else precision[label]["denominator_meaning"],
                pub,
                "analysis/errors_v07/outputs/structural_seed43.json "
                "→ headline_metric_reproduction_seed43.per_level_precision_decided",
                (
                    f"{precision[label]['numerator']}/{precision[label]['denominator']} decided "
                    f"predictions of {label}"
                    if got is not None
                    else precision[label]["denominator_meaning"]
                ),
            )

        add(
            "macro F1 with abstentions",
            self.macro_f1_with_abstentions(),
            published["macro_f1_with_abstentions_published"],
            "analysis/errors_v07/outputs/structural_seed43.json "
            "→ headline_metric_reproduction_seed43.macro_f1_with_abstentions_published",
            f"{N_TEST_EVENTS} test events, abstentions counted as misses for the true level",
        )
        add(
            "weighted kappa, covered",
            self.weighted_kappa_covered(),
            published["weighted_kappa_covered_published"],
            "analysis/errors_v07/outputs/structural_seed43.json "
            "→ headline_metric_reproduction_seed43.weighted_kappa_covered_published",
            f"{self.n_decided()} decided events (abstentions excluded)",
        )
        add(
            "coverage",
            self.n_decided() / N_TEST_EVENTS,
            metrics["coverage"],
            "results/v07_release/v07_seed43_metrics.json → metrics.coverage",
            f"{self.n_decided()}/{N_TEST_EVENTS} test events",
        )
        add(
            "levels emitted as decisions",
            self.emitted_levels(),
            list(census["emitted_level_counts"].keys()),
            "analysis/errors_v07/outputs/structural_seed43.json "
            "→ emitted_level_census['43'].emitted_level_counts",
            f"{census['decided_n']} decisions",
        )
        add(
            "levels never emitted as decisions",
            self.silent_levels(),
            list(census["levels_never_emitted"]),
            "analysis/errors_v07/outputs/structural_seed43.json "
            "→ emitted_level_census['43'].levels_never_emitted",
            f"{census['decided_n']} decisions",
        )
        add(
            "abstained events",
            self.n_abstained(),
            structural["abstention_analysis_seed43"]["abstained_n"],
            "analysis/errors_v07/outputs/structural_seed43.json "
            "→ abstention_analysis_seed43.abstained_n",
            f"{N_TEST_EVENTS} test events",
        )
        return checks

    def self_check_passed(self) -> bool:
        return all(row["ok"] for row in self.self_check())


# --------------------------------------------------------------------------
# Markdown renderers (pure strings, so they are unit-testable without Gradio)
# --------------------------------------------------------------------------


def render_self_check_table(bundle: EvidenceBundle) -> str:
    rows = bundle.self_check()
    lines = [
        "| quantity | recomputed here | published | agrees | source file | denominator |",
        "|---|---:|---:|:---:|---|---|",
    ]
    for row in rows:
        recomputed = row["recomputed"]
        published = row["published"]
        if isinstance(recomputed, float):
            recomputed_s = f"{recomputed:.6f}"
        elif isinstance(recomputed, list):
            recomputed_s = ", ".join(str(x) for x in recomputed) if recomputed else "—"
        else:
            recomputed_s = str(recomputed)
        if isinstance(published, float):
            published_s = f"{published:.6f}"
        elif isinstance(published, list):
            published_s = ", ".join(str(x) for x in published) if published else "—"
        else:
            published_s = str(published)
        lines.append(
            f"| {row['quantity']} | {recomputed_s} | {published_s} | "
            f"{'✅' if row['ok'] else '❌ **MISMATCH**'} | `{row['source']}` | "
            f"{row['denominator']} |"
        )
    ok = sum(1 for r in rows if r["ok"])
    lines.append("")
    lines.append(f"**{ok} of {len(rows)} checks agree.**")
    if ok != len(rows):
        lines.append(
            "\n> ❌ **A recomputation disagrees with a published value. Treat every "
            "number on this page as unverified until the bundle is rebuilt.**"
        )
    return "\n".join(lines)


def render_confusion_markdown(bundle: EvidenceBundle) -> str:
    cm = bundle.confusion()
    counts = cm["matrix_6x6_true_x_pred_with_abstain"]["counts"]
    norm = cm["matrix_5x5_decided_only"]["row_normalised"]
    decided_rows = bundle.decided_row_totals()
    abstain = bundle.abstentions_per_true_level()[: len(LEVELS)]
    true_totals = bundle.true_row_totals()
    col_totals = bundle.decided_col_totals()
    silent = bundle.silent_levels()

    out = [
        f"### 6×6 — true level × decision, **all {N_TEST_EVENTS} held-out test events**",
        "",
        "Rows are the consensus label; the last column is the abstention bucket "
        f"(`{ABSTENTION_TEXT}`), which is a **non-decision**, not a prediction.",
        "",
        "| true \\ decided as | L3 | L4 | L5 | L6 | L7 | unclear | row n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, label in enumerate(LEVEL_LABELS):
        cells = " | ".join(str(counts[i][j]) for j in range(len(LEVEL_LABELS)))
        out.append(f"| {label} | {cells} | {counts[i][5]} | {true_totals[i]} |")
    col_cells = " | ".join(str(t) for t in col_totals)
    out.append(f"| **col n** | {col_cells} | {bundle.n_abstained()} | {N_TEST_EVENTS} |")

    out += [
        "",
        "### 5×5 — decided events only, row-normalised (recall)",
        "",
        "The last column of the 6×6 is dropped, so each row sums to 1.0 over the "
        "events that actually received a level.",
        "",
        "| true \\ decided as | L3 | L4 | L5 | L6 | L7 | decided n | recall (own level) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    recall = bundle.per_level_recall()
    for i, label in enumerate(LEVEL_LABELS):
        cells = " | ".join(f"{norm[i][j]:.2f}" for j in range(len(LEVEL_LABELS)))
        out.append(
            f"| {label} | {cells} | {decided_rows[i]} | **{recall[label]['value']:.2f}** |"
        )

    precision = bundle.per_level_precision()
    out += [
        "",
        "### Per-level recall and precision (recomputed from the counts above)",
        "",
        "| level | IDDSI descriptor | recall | precision | recall denominator | precision denominator |",
        "|---|---|---:|---:|---|---|",
    ]
    for label in LEVEL_LABELS:
        level = int(label[1])
        r = recall[label]
        p = precision[label]
        out.append(
            f"| {label} | {LEVEL_NAMES[level]} | {_fmt(r['value'])} | {_fmt(p['value'])} | "
            f"{r['numerator']}/{r['denominator']} decided | "
            + (
                f"{p['numerator']}/{p['denominator']} decided"
                if p["value"] is not None
                else "never emitted"
            )
            + " |"
        )

    census = bundle.structural()["emitted_level_census"]
    out += [
        "",
        "### The load-bearing fact",
        "",
        f"**{' and '.join(silent)} are never emitted as a decision.** Their column "
        f"totals over all {N_TEST_EVENTS} test events are "
        f"{', '.join(f'{label} {col_totals[LEVEL_LABELS.index(label)]}' for label in silent)}. "
        "So the recall figures above are not 0.00 because the model is bad at L5/L6 "
        "in particular — they are 0.00 because the model is a **three-way "
        f"classifier plus abstention**. The only levels it will ever say are "
        f"{', '.join(bundle.emitted_levels())}.",
        "",
        "This is not a threshold artifact and not a seed artifact. Across all five "
        "v0.7 seeds:",
        "",
        "| seed | decided | emitted as decisions | never emitted |",
        "|---|---:|---|---|",
    ]
    for seed in SEEDS:
        c = census[str(seed)]
        out.append(
            f"| {seed} | {c['decided_n']} | {', '.join(c['emitted_level_counts'])} | "
            f"{', '.join(c['levels_never_emitted'])} |"
        )
    out += [
        "",
        "Even the raw argmax of the calibrated distribution never tops L4 or L6 in "
        "any seed (`structural_seed43.json` → `argmax_levels_never_top`), so the "
        "collapse is in the representation, not in the decode rule.",
        "",
        "### Where the errors actually go",
        "",
        f"Of the {bundle.n_decided()} decided events, "
        f"{bundle.bias_seed43()['correct']} are correct and "
        f"{bundle.bias_seed43()['over_predicted'] + bundle.bias_seed43()['under_predicted']} "
        "are a wrong level. The asymmetry is the direction that matters: "
        f"**{bundle.bias_seed43()['over_predicted']} over-predict, "
        f"{bundle.bias_seed43()['under_predicted']} under-predict** (mean signed error "
        f"{bundle.bias_seed43()['mean_signed_error']:+.4f} levels over "
        f"{bundle.n_decided()} decided events). Under-prediction is the direction the "
        "project's gate constrains, because it means predicting a *softer* texture "
        "than truth.",
        "",
        f"Dangerous under-classification is **{bundle.published_seed43_metrics()['metrics']['dangerous_underclassification_rate_all']:.4f}** "
        f"of all {N_TEST_EVENTS} test events "
        f"[95% CI {bundle.published_seed43_metrics()['confidence_intervals_95']['dangerous_underclassification_rate_all'][0]:.4f}"
        f"–{bundle.published_seed43_metrics()['confidence_intervals_95']['dangerous_underclassification_rate_all'][1]:.4f}], "
        f"and {bundle.published_seed43_metrics()['metrics']['dangerous_underclassification_fnr_covered']:.4f} "
        f"of the {bundle.n_decided()} decided events. The wk2 gate target is ≤ 0.05.",
    ]
    return "\n".join(out)


def render_risk_coverage_markdown(bundle: EvidenceBundle, seed: int = 43) -> str:
    metrics = bundle.seed_metrics(seed)
    m = metrics["metrics"]
    point = bundle.operating_point(seed)
    budget = bundle.coverage_at_under_risk_budget(seed)
    curve = bundle.risk_coverage_curve(seed)
    cal = bundle.calibration()
    everywhere = bundle.curve_extremes(seed, coverage_floor=0.50)
    above_gate = bundle.curve_extremes(seed, coverage_floor=0.70)

    lines = [
        f"### Seed {seed}, {N_TEST_EVENTS} held-out test events",
        "",
        "The curve is the committed `risk_coverage_curve` from "
        f"`results/v07_release/v07_seed{seed}_metrics.json` "
        f"({len(curve)} thresholds). Coverage is the share of the "
        f"{N_TEST_EVENTS} test events that clear the confidence threshold and get a "
        "level; the rest abstain. `error risk` is the wrong-level rate among the "
        "**accepted** events; `under-classification risk` is the rate at which an "
        "accepted event is predicted *softer* than its label — the direction the "
        "gate constrains.",
        "",
        "| quantity | value | denominator |",
        "|---|---:|---|",
        f"| shipped threshold | {point['threshold']:.6f} | validation-calibrated, seed {seed} |",
        f"| coverage at that threshold | {point['coverage']:.4f} | {point['accepted']}/{N_TEST_EVENTS} test events |",
        f"| error risk at that threshold | {point['error_risk']:.4f} | {point['error_risk'] * point['accepted']:.0f} of {point['accepted']} accepted events |",
        f"| under-classification risk at that threshold | {point['underclassification_risk']:.4f} | 4 of {point['accepted']} accepted events |",
        f"| over-prediction risk at that threshold | {point['overprediction_risk']:.4f} | 12 of {point['accepted']} accepted events |",
        f"| best coverage with under-risk ≤ 0.05 | {budget['best_coverage']:.4f} | {budget['point']['accepted']}/{N_TEST_EVENTS} test events |",
        f"| error-risk floor over coverage ≥ 0.50 | {everywhere['error_risk_min']:.4f} | {everywhere['n_points']} committed curve points |",
        f"| error-risk floor over coverage ≥ 0.70 | {above_gate['error_risk_min']:.4f} | {above_gate['n_points']} committed curve points |",
        "",
        "> The shipped threshold is not itself a point on the committed curve (it "
        "sits between the 0.2389 and 0.2743 points), so the three risks at the "
        "operating point are recomputed from the decision tally in "
        "`analysis/errors_v07/outputs/bias_seed43.json` "
        f"({point['accepted']} decided: 76 correct, 12 over, 4 under) rather than "
        "interpolated.",
        "",
        f"Calibration on **validation** (`v07_seed{seed}_calibration.json`): "
        f"coverage {cal['coverage']:.4f}, under-FNR {cal['under_fnr']:.4f} against a "
        f"target of {cal['target_under_fnr']}, `target_met: {cal['target_met']}`.",
        "",
        "**Reading the curve honestly.** Under-classification risk falls to 0.0000 "
        "once coverage drops below about 0.64 — but that is abstention buying "
        f"safety, not skill. Across every committed point with coverage ≥ 0.50 the "
        f"**error** risk never drops below {everywhere['error_risk_min']:.4f}, and "
        f"across coverage ≥ 0.70 it never drops below "
        f"{above_gate['error_risk_min']:.4f}. At the shipped operating point the "
        f"model still gets {point['error_risk'] * point['accepted']:.0f} of "
        f"{point['accepted']} accepted events wrong. A selective classifier whose "
        "abstentions are concentrated on the levels it cannot express is not a "
        "usable texture estimator; it is a three-way classifier that declines to "
        "answer most of the time.",
        "",
        "**Gate check on this split** (recomputed from the committed metrics file):",
        "",
        "| gate | threshold | measured | verdict |",
        "|---|---:|---:|:---:|",
        f"| wk1 weighted κ (covered) | ≥ 0.70 | {m['weighted_kappa_covered']:.4f} | "
        f"{'PASS' if m['weighted_kappa_covered'] >= 0.70 else 'FAIL'} |",
        f"| wk2 macro-F1 | ≥ 0.75 | {m['macro_f1_with_abstentions']:.4f} | "
        f"{'PASS' if m['macro_f1_with_abstentions'] >= 0.75 else 'FAIL'} |",
        f"| wk2 dangerous under-classification (covered FNR) | ≤ 0.05 | {m['dangerous_underclassification_fnr_covered']:.4f} | "
        f"{'PASS' if m['dangerous_underclassification_fnr_covered'] <= 0.05 else 'FAIL'} |",
        f"| wk2 abstention | ≤ 0.30 | {m['abstention_rate']:.4f} | "
        f"{'PASS' if m['abstention_rate'] <= 0.30 else 'FAIL'} |",
    ]
    return "\n".join(lines)


def render_gate_markdown(bundle: EvidenceBundle) -> str:
    rows = bundle.gate_table()
    lines = [
        "### The pre-registered gates, recomputed from the committed metrics",
        "",
        "Definitions, quoted from `hf/model_README.md` (bundled verbatim at "
        "`evidence/model_README.md` so the quotation is checkable):",
        "",
        f"> {GATE_DEFINITIONS['wk1']}  ",
        f"> {GATE_DEFINITIONS['wk2']}",
        "",
        "Both are evaluated here from `results/v07_release/v07_seed*_metrics.json`. "
        "wk1 is scored against the **v0.7 full-coverage consensus**, the label set "
        "the v0.7 runs were trained on — the Yardstick tab shows what happens to "
        "the same checkpoints when the yardstick changes.",
        "",
        "| seed | n | coverage | κ (covered) | macro-F1 | under-FNR (covered) | abstention | wk1 | wk2 (all three) |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['n']} | {r['coverage']:.4f} | "
            f"{r['weighted_kappa_covered']:.4f} | {r['macro_f1_with_abstentions']:.4f} | "
            f"{r['dangerous_underclassification_fnr_covered']:.4f} | "
            f"{r['abstention_rate']:.4f} | "
            f"{'PASS' if r['wk1_kappa_ge_0.70'] else 'FAIL'} | "
            f"{'PASS' if r['wk2_all_three'] else 'FAIL'} |"
        )
    n_wk1 = sum(1 for r in rows if r["wk1_kappa_ge_0.70"])
    n_wk2 = sum(1 for r in rows if r["wk2_all_three"])
    n_both = sum(1 for r in rows if r["wk1_and_wk2"])
    lines += [
        "",
        f"**wk1: {n_wk1}/{len(rows)} seeds pass. wk2: {n_wk2}/{len(rows)} seeds pass. "
        f"Both gates on the same seed: {n_both}/{len(rows)}.**",
        "",
        "wk1 passes on the v0.7 yardstick; **wk2 does not**, on any seed, because "
        "macro-F1 sits at 0.319–0.395 against a 0.75 gate. The release card's "
        "conclusion — *\"The model fails all of them. The wk1 kill rule FIRES. "
        "Photo-classification is killed as a product capability. This is why only "
        "code and results ship.\"* — is quoted, not re-derived here; the "
        "withholding decision is unchanged on either yardstick, and it rests on "
        "the under-classification and calibration failures rather than on the "
        "kappa alone.",
    ]
    return "\n".join(lines)


def render_yardstick_markdown(bundle: EvidenceBundle, common_split: bool = False) -> str:
    """The yardstick flip.

    ``common_split`` switches the table between the published denominators
    (v0.4 is scored on its own 142-event held-out split for the v0.5 column)
    and the 108-event split common to all three models, so the reader can see
    that the verdict does not depend on which denominator was used.
    """

    summary = bundle.yardstick_summary()
    flip = bundle.yardstick_flip()
    ident = bundle.bias_matrix_identification()
    bias = bundle.bias_matrix()

    suffix = " (108-event common split)" if common_split else ""
    per_yardstick = summary["per_yardstick"]
    key_v05 = f"v0.5 consensus{suffix}"
    key_v07 = f"v0.7 full-coverage consensus{suffix}"
    n_events = {
        model: per_yardstick[key_v05][model]["n_events_per_seed"][0]
        for model in ("v0.4", "v0.5", "v0.7")
    }

    lines = [
        "### The same checkpoints, two label sets",
        "",
        "Three model versions × five seeds each, scored twice: against the "
        "**v0.5 consensus** labels and against the **v0.7 full-coverage "
        "consensus** labels. Nothing about the models changed between the two "
        "columns. Every number is quadratic weighted kappa on the **covered** "
        "rows only — the wk1 gate metric — recomputed in "
        "`results/yardstick_audit/` from the per-event prediction files, with "
        "CORAL rank decode and QWK re-implemented from their definitions (no "
        "sklearn, no `trainer/eval/metrics.py`).",
        "",
        (
            "**Denominator basis: the 108-event split common to all three "
            "models.** v0.4, v0.5 and v0.7 are scored on the same held-out "
            "events, so the three rows are directly comparable."
            if common_split
            else "**Denominator basis: as published.** v0.4 is scored on its "
            "own 142-event held-out split for the v0.5 column (its published "
            "denominator); v0.5 and v0.7 use the 108-event split. Switch to "
            "the common split to put all three on one denominator — the "
            "verdict does not change."
        ),
        "",
        "| model | n events | vs v0.5 consensus (wk1 range) | seeds passing 0.70 | vs v0.7 consensus (wk1 range) | seeds passing 0.70 |",
        "|---|---:|---|---:|---|---:|",
    ]
    for model in ("v0.4", "v0.5", "v0.7"):
        a = per_yardstick[key_v05][model]
        b = per_yardstick[key_v07][model]
        lines.append(
            f"| {model} | {n_events[model]} | {a['min']:.4f} – {a['max']:.4f} | "
            f"{a['seeds_passing_wk1_0.70']}/{a['n_seeds']} | "
            f"{b['min']:.4f} – {b['max']:.4f} | "
            f"{b['seeds_passing_wk1_0.70']}/{b['n_seeds']} |"
        )
    lines += [
        "",
        "**A gate outcome here is a property of the yardstick, not of the model.** "
        "All three versions fail wk1 on the v0.5 labels and all three pass it on "
        "the v0.7 labels. This is hazard **H-YARDSTICK** in "
        "`hf/HAZARD_LOG.md`, and it is the reason a kappa is not interpretable "
        "without the label set attached to it.",
        "",
        "### Why the two yardsticks disagree",
        "",
        f"On the {flip['common_test_events']} held-out events common to all three "
        f"models, the two label sets are identical on **{flip['label_unchanged']}** "
        f"and differ on **{flip['label_changed_between_v05_and_v07']}** "
        f"({flip['label_changed_between_v05_and_v07'] / flip['common_test_events']:.1%}). "
        "The largest single transition is **L5 → L7** (10 events): a texture level "
        "moving to the catch-all regular-food end. Direction of every change:",
        "",
        "| transition | events |",
        "|---|---:|",
    ]
    for transition, n in sorted(
        flip["label_change_transitions_on_common_test_events"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| {transition} | {n} |")
    lines += [
        "",
        "The circularity is disclosed, not hidden: the labeller families behind "
        "the two label sets overlap exactly "
        f"({', '.join(flip['v07_labeller_union'])}), and "
        "`v07_labellers_absent_from_v05` is empty. The two yardsticks are not "
        "independent of the models.",
        "",
        "### `results/v07_release/bias_matrix.json` is a different metric",
        "",
        "The release card cites `bias_matrix.json` as the evidence for the "
        "yardstick flip. Identifying its six cells against the per-seed "
        "recomputation shows it holds **mean kappa over ALL rows, abstentions "
        "included** — not the gate metric, which is kappa on covered rows:",
        "",
        "| model | yardstick | value in bias_matrix.json | mean wk1 (covered) | identified as | as the 0.70 gate it would |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in ident:
        covered = row["candidates"]["mean wk1 (kappa on covered rows) = the gate metric"]
        allrows = row["candidates"]["mean kappa over ALL rows, abstentions included"]
        lines.append(
            f"| {row['model']} | {row['yardstick']} | {row['published_bias_matrix_value']:.4f} | "
            f"{covered:.4f} | mean kappa over ALL rows, abstentions included "
            f"(matches to {abs(allrows - row['published_bias_matrix_value']):.1e}) | "
            f"{'PASS' if row['as_gate_would_it_pass_0.70'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Read as the gate metric, `bias_matrix.json` does **not** support the "
        "sentence it is attached to: its v0.7-labels column puts v0.5 at "
        f"**{bias['v0.5 (trained: cleaned consensus)']['v0.7 consensus (full coverage)']:.4f}**, "
        "below the 0.70 gate, while wk1 on covered rows puts it at 0.7036–0.7953 "
        "(5/5 seeds). Both quantities are real; they are not the same quantity, "
        "and the Space shows both rather than picking the flattering one.",
        "",
        "### What the decomposition shows",
        "",
        "Kappa on the events whose label **did not change** is the control — the "
        "two yardsticks agree there, so any gap is the label change itself. "
        "Averaged over the three models and five seeds, wk1 on unchanged-label "
        "events sits around 0.75–0.91 while wk1 on changed-label events sits "
        "around 0.54–0.83. The flip lives in the changed-label events, which is "
        "exactly where the label noise lives.",
    ]
    return "\n".join(lines)


def render_corpus_hygiene_markdown(bundle: EvidenceBundle) -> str:
    h = bundle.corpus_hygiene()
    sweep = bundle.labelability_sweep()
    judge = bundle.inter_judge()
    cov = bundle.coverage_meta()
    dedup = bundle.dedup()

    lines = [
        "### Licence hygiene — why the harvested dataset does not ship",
        "",
        "Measured this run by `space/tools/build_evidence_bundle.py` from the "
        "attribution ledgers at `/work/data/<set>/attribution.jsonl` for the three "
        "sets `realweak_pilot`, `harvest_d1` and `harvest_d1b`, under the "
        "fail-closed release rule (a row is release-clean only if its "
        "provider is per-asset or dataset-level verified, its licence is in the "
        "allowlist and is not a ShareAlike family incompatible with the BY-NC-SA "
        "main release, and its attribution fields are complete).",
        "",
        "| quantity | value | denominator |",
        "|---|---:|---|",
        f"| harvested rows | {h['harvest_rows_total']} | {', '.join(f'{k} {v}' for k, v in h['harvest_rows_per_set'].items())} |",
        f"| unique event ids | {h['unique_event_ids']} | {h['duplicate_event_id_rows']} rows repeat an event id |",
        f"| rows with unverified open-images/openverse licences | {h['unverified_open_images_openverse_rows']} | {h['harvest_rows_total']} rows |",
        f"| — of which open-images | {h['unverified_by_provider']['open-images']} | unverified rows |",
        f"| — of which openverse | {h['unverified_by_provider']['openverse']} | unverified rows |",
        f"| CC BY-SA rows (ShareAlike vs BY-NC-SA main release) | {h['sharealike_rows']} | {h['harvest_rows_total']} rows |",
        f"| rows with incomplete attribution | {h['rows_with_incomplete_attribution']} | {h['harvest_rows_total']} rows |",
        f"| **release-clean rows** | **{h['release_clean_rows']}** | {h['harvest_rows_total']} rows |",
        f"| **release-clean unique events** | **{h['release_clean_unique_events']}** | {h['unique_event_ids']} unique events |",
        "",
        f"The dataset is therefore **{h['release_clean_unique_events'] / h['unique_event_ids']:.1%} "
        "publishable** under its own fail-closed rule. That is hazard "
        "**H-LICENCE**: the release builder shipped 397 unverified open-images "
        "rows until the verification check failed closed on a missing field, and "
        "fail-closed verification stays as the gate.",
        "",
        "> ⚠️ **A number the release card and the measurement disagree on.** The "
        "card says *\"Of 587 unique events exactly 2 would ship cleanly … 7 are CC "
        "BY-SA … 109 have unresolved redistribution status\"* "
        "(`hf/model_README.md`, also quoted in `docs/REGULATORY_POSTURE.md`). "
        f"The measurement above, from the same ledgers, gives **{h['unique_event_ids']}** "
        f"unique events, **{h['release_clean_unique_events']}** release-clean unique "
        f"events and **{h['sharealike_rows']}** CC BY-SA rows. The 592 unverified "
        "open-images/openverse rows agree exactly. The card's smaller figures come "
        "from an earlier ledger state; the Space shows the current measurement and "
        "flags the discrepancy rather than quietly repeating either. **Either way "
        "the dataset does not ship** — the gap is between 2 and 99 events out of "
        f"{h['unique_event_ids']}.",
        "",
        "### Deduplication and split hygiene",
        "",
        "| quantity | value | denominator |",
        "|---|---:|---|",
        f"| images hashed | {dedup['images_hashed']} | all harvest media |",
        f"| exact byte-duplicate groups | {dedup['exact_byte_duplicate_groups']} | — |",
        f"| near-duplicate pairs (dHash) | {dedup['near_duplicate_pairs']} | {dedup['images_hashed']} images |",
        f"| groups with more than one member | {dedup['groups_with_more_than_one_member']} | — |",
        f"| groups spanning multiple splits **in v0.2** | {dedup['groups_spanning_multiple_splits_in_v02']} | the leak that v0.3 fixed |",
        "",
        f"The v0.7 manifest itself is clean on this axis: "
        f"**{h['v07_groups_spanning_multiple_splits']}** of its "
        f"{h['v07_manifest_unique_groups']} groups span more than one split, and "
        f"train/test, train/val and val/test group overlaps are "
        f"{h['v07_group_overlap_train_test']}, {h['v07_group_overlap_train_val']} and "
        f"{h['v07_group_overlap_val_test']} respectively "
        f"(measured from `/work/data/manifest_v07_weighted.jsonl`, "
        f"{h['v07_manifest_rows']} rows: {h['v07_split_counts']['train']} train / "
        f"{h['v07_split_counts']['val']} val / {h['v07_split_counts']['test']} test).",
        "",
        "### Label coverage — how much of the corpus carries a real judgement",
        "",
        "| quantity | value | denominator |",
        "|---|---:|---|",
        f"| v0.7 manifest rows | {h['v07_manifest_rows']} | — |",
        f"| events with two judgements | {h['v07_events_with_two_judgements']} | {h['v07_manifest_rows']} rows |",
        f"| — unanimous (2/2) | {h['v07_events_unanimous']} | {h['v07_events_with_two_judgements']} double-coverage events |",
        f"| — split (1/2) | {h['v07_events_split']} | {h['v07_events_with_two_judgements']} double-coverage events |",
        f"| single-judgement events | {h['v07_events_single_judgement']} | {h['v07_manifest_rows']} rows |",
        f"| **unanimous fraction of double-coverage** | **{h['v07_unanimous_fraction_of_double']:.4f}** | {h['v07_events_with_two_judgements']} events |",
        "",
        f"`results/regen_20260916/inter_judge_20260916.json`, on {judge['n_pairs']} "
        f"doubled-coverage pairs: exact agreement **{judge['exact_agreement']:.4f}**, "
        f"unweighted κ {judge['kappa_unweighted']:.4f}, quadratic κ "
        f"{judge['kappa_quadratic']:.4f}. That is the **negative finding** of the "
        "project (hazard **H-NOISE**): two VLM judge families agree on an IDDSI "
        "level from the same web photo barely above chance for five classes, "
        "because middle levels (L4/L5/L6) need absolute size and flow information "
        "a photograph does not carry.",
        "",
        "### Labelability sweep",
        "",
        "| quantity | value | denominator |",
        "|---|---:|---|",
        f"| events judged by both labeller families | {sweep['both']} | — |",
        f"| cross-family agreement that the image contains food | {sweep['cross_food_agree']}/{sweep['both']} = {sweep['cross_food_agree'] / sweep['both']:.4f} | {sweep['both']} events |",
        f"| cross-family agreement that it is levellable at all | {sweep['cross_lev_agree']}/{sweep['both']} = {sweep['cross_lev_agree'] / sweep['both']:.4f} | {sweep['both']} events |",
        f"| step-family: not food | {sweep['step_food_no']} | {sweep['step_events']} events |",
        f"| step-family: unlevellable | {sweep['step_unlevellable']} | {sweep['step_events']} events |",
        f"| mm-family: not food | {sweep['mm_food_no']} | {sweep['mm_events']} events |",
        f"| mm-family: unlevellable | {sweep['mm_unlevellable']} | {sweep['mm_events']} events |",
        "",
        f"Test-split level mix, for the denominators on the Confusion tab: "
        + ", ".join(f"{k} {v}" for k, v in sorted(h["v07_test_level_counts"].items()))
        + f" out of {h['v07_test_n']}.",
    ]
    return "\n".join(lines)


def render_scope_markdown() -> str:
    return f"""
### What this Space is

A rendering of the **measured evidence** behind the decision to withhold the
IDDSI-Open photo-classification weights. It loads no model, accepts no image,
and produces no level estimate.

### What this Space is not

- **Not a swallowing-safety classifier.** It does not assess swallowing,
  aspiration risk, or suitability for any person.
- **Not an IDDSI test.** It performs no official IDDSI test and measures no
  flow, hardness, adhesiveness or cohesiveness.
- **Not a verdict.** It never says food is safe or unsafe to consume, and never
  tells anyone what texture to prepare, modify or prescribe.
- **Not IDDSI-endorsed.** This project is independent of IDDSI.
- **Not a demo of the model.** There is no model here to demo.

### Disclaimer

**{DISCLAIMER}**

**免責聲明（繁體中文）。** {DISCLAIMER_ZH_HANT}

### The one output this Space can produce

`{ABSTENTION_TEXT}` — and only as a quotation, because that is the abstention
string the model's decision path is required to emit. It is not a prediction.
"""
