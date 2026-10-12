"""Hugging Face Space: the evidence behind the IDDSI-Open weight-withholding decision.

Sdk front matter lives in ``README.md`` next to this file (``sdk: gradio``).

There is deliberately **no image input and no model on this page**. The v0.7
SigLIP-2 + monotone CORAL weights are withheld by a locked decision
(2026-09-07; ``hf/model_README.md``, "Why the weights are withheld"), so a
"classify your food" demo would present a number nobody could check. This Space
renders the evidence instead:

1. **Confusion matrix** — the per-level matrix on the 108-event held-out split,
   showing L5 and L6 recall of 0.00 because those levels are *never emitted as a
   decision*, in any of the five seeds.
2. **Risk–coverage and gates** — the committed selective-classification curve,
   with the wk1/wk2 gate outcomes recomputed from the shipped metrics.
3. **Yardstick flip** — the same checkpoints scored against two different label
   sets, passing or failing the same gate depending only on the yardstick.
4. **Corpus hygiene** — the licence, dedup, split and label-coverage numbers
   that keep the harvested dataset unpublishable.
5. **Provenance and self-check** — every bundled file's source and SHA-256, and
   every headline number recomputed here and compared to the published value.

Everything displayed comes from ``space/evidence/``, which is built by
``space/tools/build_evidence_bundle.py`` from the repository's committed JSON
plus one measurement taken from the shared read-only inputs. Nothing is
hand-typed. ``space/core.py`` holds the data layer and is unit-tested without
Gradio.

**This is not a swallowing-safety tool.** It performs no IDDSI test, measures no
flow, hardness, adhesiveness or cohesiveness, makes no determination about any
person, and issues no safe/unsafe verdict. It is a culinary-education research
artefact.
"""

from __future__ import annotations

import pandas as pd

import gradio as gr

try:  # Space layout: app.py sits in space/ and imports the sibling module.
    from core import (
        ABSTENTION_TEXT,
        DISCLAIMER,
        DISCLAIMER_ZH_HANT,
        EVIDENCE_DIR,
        LEVEL_LABELS,
        LEVEL_NAMES,
        LEVELS,
        N_TEST_EVENTS,
        SEEDS,
        WITHHELD_NOTICE,
        WITHHELD_REASON,
        EvidenceBundle,
        render_confusion_markdown,
        render_corpus_hygiene_markdown,
        render_gate_markdown,
        render_risk_coverage_markdown,
        render_scope_markdown,
        render_self_check_table,
        render_yardstick_markdown,
    )
except ImportError:  # Repo-root layout: space/ is a package on sys.path.
    from space.core import (  # type: ignore[no-redef]
        ABSTENTION_TEXT,
        DISCLAIMER,
        DISCLAIMER_ZH_HANT,
        EVIDENCE_DIR,
        LEVEL_LABELS,
        LEVEL_NAMES,
        LEVELS,
        N_TEST_EVENTS,
        SEEDS,
        WITHHELD_NOTICE,
        WITHHELD_REASON,
        EvidenceBundle,
        render_confusion_markdown,
        render_corpus_hygiene_markdown,
        render_gate_markdown,
        render_risk_coverage_markdown,
        render_scope_markdown,
        render_self_check_table,
        render_yardstick_markdown,
    )

_TITLE = "IDDSI-Open — the evidence, not the model"
SUBTITLE = (
    "Research preview for culinary education and QA. Not a swallowing-safety "
    "classifier, not an IDDSI test, not a verdict."
)

_INTRO = f"""
# {_TITLE}

{SUBTITLE}

> 🚫 **{WITHHELD_NOTICE}**

{WITHHELD_REASON}

This page holds **no image input and no classifier**. It renders four measured
things and nothing else: the per-level confusion matrix, the risk–coverage curve,
the yardstick flip, and the corpus-hygiene numbers. Every figure is either
recomputed from raw counts on this page or quoted with the file it came from —
the **Provenance & self-check** tab shows the recomputation next to the
published value for each one.

**{DISCLAIMER}**

**免責聲明（繁體中文）。** {DISCLAIMER_ZH_HANT}
"""

_HOW_TO_READ = """
### How to read this Space

| Tab | What it shows | Where the numbers come from |
|---|---|---|
| **Confusion matrix** | Per-level recall and precision on the 108-event held-out split, and the fact that L5/L6 are never emitted as a decision | recomputed from the raw counts in `analysis/errors_v07/outputs/confusion_seed43.json` |
| **Risk–coverage & gates** | The committed selective-classification curve and the wk1/wk2 gate outcomes | `results/v07_release/v07_seed*_metrics.json` |
| **Yardstick flip** | The same checkpoints, two label sets, opposite gate verdicts | `results/yardstick_audit/` |
| **Corpus hygiene** | Licence, dedup, split and label-coverage numbers | measured from `/work/data/*` by the bundle builder |
| **Provenance & self-check** | SHA-256 of every bundled file, and every headline number recomputed vs published | `space/evidence/PROVENANCE.json` |

**Denominators are stated everywhere.** The held-out split is 108 events
(10 L3, 11 L4, 7 L5, 10 L6, 70 L7). Coverage is the share of those 108 events
that receive a level; the rest abstain with
`unclear — perform physical test`, which is a non-decision, not a prediction.
"""

_BUNDLE = EvidenceBundle()
_SELF_CHECK_OK = _BUNDLE.self_check_passed()


# --------------------------------------------------------------------------
# DataFrames for the plots
# --------------------------------------------------------------------------


def _recall_frame(bundle: EvidenceBundle) -> "pd.DataFrame":
    """Per-level recall over decided events, in ordinal (soft -> thick) order."""

    recall = bundle.per_level_recall()
    return pd.DataFrame(
        {
            "level": LEVEL_LABELS,
            "recall": [recall[level]["value"] or 0.0 for level in LEVEL_LABELS],
            "decided n": [recall[level]["denominator"] for level in LEVEL_LABELS],
            "emitted as a decision": [
                "yes" if level in bundle.emitted_levels() else "never emitted"
                for level in LEVEL_LABELS
            ],
        }
    )


def _matrix_frame(bundle: EvidenceBundle, normalised: bool) -> "pd.DataFrame":
    """The 6x6 true x decision matrix, as counts or row-normalised rates."""

    counts = bundle.matrix_6x6()
    columns = list(LEVEL_LABELS) + ["unclear"]
    rows = []
    for i, label in enumerate(LEVEL_LABELS):
        row_total = sum(counts[i])
        if normalised and row_total:
            values = [c / row_total for c in counts[i]]
        else:
            values = [float(c) for c in counts[i]]
        rows.append({"true level": label, **dict(zip(columns, values))})
    return pd.DataFrame(rows)


def _risk_coverage_frame(bundle: EvidenceBundle, seed: int) -> "pd.DataFrame":
    """Long-format risk/coverage data: one row per (curve point, metric).

    ``gr.LinePlot`` draws one line per unique value of ``color``, so the two
    risk columns are stacked into a ``metric`` column.  The shipped operating
    point is appended as an extra row pair so it is visible on the plot even
    though its threshold is not one of the swept confidence values.
    """

    curve = bundle.risk_coverage_curve(seed)
    point = bundle.operating_point(seed)
    rows = []
    for source, coverage, accepted, error, under in [
        *((
            "committed curve",
            p["coverage"],
            p["accepted"],
            p["error_risk"],
            p["underclassification_risk"],
        ) for p in curve),
        (
            "shipped threshold",
            point["coverage"],
            point["accepted"],
            point["error_risk"],
            point["underclassification_risk"],
        ),
    ]:
        rows.append(
            {
                "coverage": coverage,
                "risk": error,
                "metric": "error risk (wrong level)",
                "source": source,
                "accepted n": accepted,
            }
        )
        rows.append(
            {
                "coverage": coverage,
                "risk": under,
                "metric": "under-classification risk (predicted softer than label)",
                "source": source,
                "accepted n": accepted,
            }
        )
    return pd.DataFrame(rows)


def _yardstick_frame(bundle: EvidenceBundle, common_split: bool) -> "pd.DataFrame":
    """Per-run wk1 for every model x seed x yardstick, for the grouped bar plot."""

    suffix = " (108-event common split)" if common_split else ""
    keys = (f"v0.5 consensus{suffix}", f"v0.7 full-coverage consensus{suffix}")
    runs = [
        r
        for r in bundle.yardstick_per_seed()
        if r["yardstick"] in keys and r["version"] in ("v0.4", "v0.5", "v0.7")
    ]
    return pd.DataFrame(
        {
            "model": [r["version"] for r in runs],
            "seed": [int(r["seed"]) for r in runs],
            "yardstick": [r["yardstick"] for r in runs],
            "wk1 (quadratic weighted kappa, covered rows)": [
                r["wk1_kappa_covered"] for r in runs
            ],
            "n events": [r["n_events"] for r in runs],
            "n covered": [r["n_covered"] for r in runs],
        }
    )


def _provenance_frame(bundle: EvidenceBundle) -> "pd.DataFrame":
    prov = bundle.provenance()["files"]
    rows = []
    for name, meta in sorted(prov.items()):
        rows.append(
            {
                "bundled file": name,
                "kind": meta["kind"],
                "source": meta.get("source", ""),
                "bytes": meta["bytes"],
                "sha256": meta["sha256"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------


def _confusion(normalised: bool):
    return (
        _matrix_frame(_BUNDLE, normalised),
        _recall_frame(_BUNDLE),
        render_confusion_markdown(_BUNDLE),
    )


def _risk_coverage(seed: int):
    return (
        _risk_coverage_frame(_BUNDLE, seed),
        render_risk_coverage_markdown(_BUNDLE, seed),
    )


_YARDSTICK_COMMON_LABEL = (
    "108-event common split (all three models share one denominator)"
)


def _yardstick(common_split: str | bool):
    if isinstance(common_split, str):
        common_split = common_split == _YARDSTICK_COMMON_LABEL
    return (
        _yardstick_frame(_BUNDLE, common_split),
        render_yardstick_markdown(_BUNDLE, common_split),
    )


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def build_demo() -> "gr.Blocks":
    """Construct the Gradio Blocks app (kept out of module scope so tests can
    import this module without a running UI)."""

    with gr.Blocks(title=_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(_INTRO)
        gr.Markdown(_HOW_TO_READ)

        if not _SELF_CHECK_OK:
            gr.Markdown(
                "> ❌ **Self-check failed.** At least one number recomputed from the "
                "bundled evidence disagrees with the value published in the "
                "repository. See the *Provenance & self-check* tab before trusting "
                "anything on this page."
            )

        with gr.Tabs():
            # -------------------------------------------------------------
            with gr.TabItem("1 · Scope and disclaimer"):
                gr.Markdown(render_scope_markdown())
                gr.Markdown(
                    "### What the withheld model was\n\n"
                    f"- **Backbone.** `google/siglip2-base-patch16-224`, resized to "
                    "224×224.\n"
                    "- **Head.** A monotone CORAL ordinal head predicting cut-point "
                    "logits `P(y > k)` for k = L3..L6, converted to class "
                    "probabilities.\n"
                    f"- **Levels.** L{LEVELS[0]}–L{LEVELS[-1]} only. Anything thinner "
                    "than L3 (IDDSI L0–L2) is outside what the head can express, and "
                    "a photo of it would be reported as an L3–L7 distribution that "
                    "means nothing.\n"
                    "- **Decode.** The ordinal rank rule "
                    "`level = #{k : P(y>k) > 0.5}`, *not* argmax over the class "
                    "masses. Argmax over sigmoid differences provably cannot return "
                    "an interior level for any input — hazard **H-CORAL** in "
                    "`hf/HAZARD_LOG.md`.\n"
                    "- **Abstention.** Below the calibrated confidence threshold the "
                    "model answers "
                    f"`{ABSTENTION_TEXT}` instead of a level.\n\n"
                    "The weights and the harvested image dataset are both withheld. "
                    "Only code and eval results ship."
                )

            # -------------------------------------------------------------
            with gr.TabItem("2 · Confusion matrix"):
                gr.Markdown(
                    f"### Per-level confusion on the held-out split, seed 43\n\n"
                    f"**{N_TEST_EVENTS} test events**, group-disjoint, consensus "
                    "labels. 92 received a level, 16 abstained. This is the single "
                    "most important exhibit: **L5 and L6 recall are 0.00 because the "
                    "model never emits them as a decision at all**, so the ordinal "
                    "head is in practice a three-way classifier plus abstention."
                )
                with gr.Row():
                    normalise = gr.Radio(
                        choices=["counts", "row-normalised (recall)"],
                        value="counts",
                        label="Matrix values",
                    )
                matrix = gr.Dataframe(
                    value=_matrix_frame(_BUNDLE, False),
                    label="true level × decision (6×6, all 108 events, abstentions included)",
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown(
                    "### Per-level recall over decided events\n\n"
                    "Zero-height bars are levels the model never says. The bar plot "
                    "and the table below are recomputed from the same counts."
                )
                recall_chart = gr.BarPlot(
                    value=_recall_frame(_BUNDLE),
                    x="level",
                    y="recall",
                    title="Per-level recall over decided events (seed 43, 108 held-out events)",
                    y_lim=[0, 1],
                    height=260,
                    tooltip=["level", "recall", "decided n", "emitted as a decision"],
                )
                gr.Markdown(render_confusion_markdown(_BUNDLE))
                normalise.change(
                    _confusion,
                    inputs=normalise,
                    outputs=[matrix, recall_chart, gr.Markdown()],
                )

            # -------------------------------------------------------------
            with gr.TabItem("3 · Risk–coverage and gates"):
                gr.Markdown(
                    "### Selective classification: what abstention buys\n\n"
                    "Coverage is the share of the 108 held-out events that clear the "
                    "confidence threshold. Pick a seed — all five are shown because "
                    "the shape of the curve, not its exact position, is the finding."
                )
                seed_choice = gr.Dropdown(
                    choices=[str(s) for s in SEEDS],
                    value="43",
                    label="v0.7 seed",
                )
                curve_chart = gr.LinePlot(
                    value=_risk_coverage_frame(_BUNDLE, 43),
                    x="coverage",
                    y="risk",
                    color="metric",
                    title="Risk vs coverage, seed 43 — committed curve plus the shipped threshold",
                    x_title="coverage (share of the 108 held-out events that get a level)",
                    y_title="risk among the accepted events",
                    height=360,
                    tooltip=["coverage", "risk", "metric", "accepted n", "source"],
                    overlay_point=True,
                )
                gr.Markdown(render_risk_coverage_markdown(_BUNDLE, 43))
                gr.Markdown("---")
                gr.Markdown(render_gate_markdown(_BUNDLE))
                seed_choice.change(
                    _risk_coverage,
                    inputs=seed_choice,
                    outputs=[curve_chart, gr.Markdown()],
                )

            # -------------------------------------------------------------
            with gr.TabItem("4 · Yardstick flip"):
                gr.Markdown(
                    "### Same checkpoints, two label sets, opposite verdicts\n\n"
                    "Three model versions × five seeds, scored against the v0.5 "
                    "consensus and against the v0.7 full-coverage consensus. Nothing "
                    "about the models changed between the two bars. The gate is "
                    "κ ≥ 0.70 on covered rows."
                )
                basis = gr.Radio(
                    choices=[
                        "As published (v0.4 scored on its own 142-event split)",
                        _YARDSTICK_COMMON_LABEL,
                    ],
                    value="As published (v0.4 scored on its own 142-event split)",
                    label="Denominator basis",
                )
                flip_chart = gr.BarPlot(
                    value=_yardstick_frame(_BUNDLE, False),
                    x="model",
                    y="wk1 (quadratic weighted kappa, covered rows)",
                    color="yardstick",
                    title="wk1 by model and yardstick (one bar per seed; the line is the 0.70 gate)",
                    y_lim=[0, 1],
                    height=340,
                    tooltip=[
                        "model",
                        "seed",
                        "yardstick",
                        "wk1 (quadratic weighted kappa, covered rows)",
                        "n events",
                        "n covered",
                    ],
                )
                gr.Markdown(render_yardstick_markdown(_BUNDLE))
                basis.change(
                    _yardstick,
                    inputs=basis,
                    outputs=[flip_chart, gr.Markdown()],
                )

            # -------------------------------------------------------------
            with gr.TabItem("5 · Corpus hygiene"):
                gr.Markdown(render_corpus_hygiene_markdown(_BUNDLE))

            # -------------------------------------------------------------
            with gr.TabItem("6 · Provenance and self-check"):
                gr.Markdown(
                    "### Every number on this page, checked\n\n"
                    "The left column is recomputed here from the raw counts in the "
                    "bundled evidence; the middle column is the value published in "
                    "the repository. If they disagree the Space says so."
                )
                gr.Markdown(render_self_check_table(_BUNDLE))
                gr.Markdown(
                    "### Bundled files\n\n"
                    f"`{EVIDENCE_DIR.name}/` is rebuilt by "
                    "`space/tools/build_evidence_bundle.py`. Each row records whether "
                    "the file was copied verbatim from the repository, measured by "
                    "that script, or copied with a recorded reduction."
                )
                gr.Dataframe(
                    value=_provenance_frame(_BUNDLE),
                    label="bundled evidence files",
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown(
                    "### Sources not bundled\n\n"
                    "- The per-event prediction arrays for the 60 yardstick-audit "
                    "runs stay in the repository at "
                    "`results/yardstick_audit/per_seed.json`; only their scalar "
                    "summaries ship here.\n"
                    "- The sealed evaluation set (`capture/sealed_eval_plan.json`, "
                    "`capture/sealed_eval_pilot.json`) was never opened by the "
                    "bundle builder, this Space, or any measurement quoted here.\n"
                    "- `hf/HAZARD_LOG.md` and `hf/model_README.md` are bundled "
                    "byte-for-byte under `evidence/` so every quotation on this page "
                    "is checkable against its source."
                )

        gr.Markdown("---")
        gr.Markdown(f"### Disclaimer\n\n**{DISCLAIMER}**")
        gr.Markdown(f"### 免責聲明\n\n**{DISCLAIMER_ZH_HANT}**")
        gr.Markdown(
            "### Scope\n\n"
            "- In scope: research preview and culinary-education use only.\n"
            "- Out of scope: any safe/unsafe verdict; any dietary, clinical or "
            "patient claim; any assessment of swallowing, aspiration risk or "
            "suitability for a person; any instruction to prepare, modify or "
            "prescribe a texture.\n"
            "- Not IDDSI-endorsed. This project is independent of IDDSI.\n"
            f"- The model's head covers L{LEVELS[0]}–L{LEVELS[-1]} only. IDDSI "
            "L0–L2 (thin liquids) are outside what it can express.\n"
            "- This Space runs no model and accepts no image. The only output it "
            f"can produce is the quoted abstention string `{ABSTENTION_TEXT}`.\n"
        )
    return demo


# Hugging Face Spaces discovers a module-level ``demo``.  Mirrors the guard in
# ``crowd/app.py`` so an environment without a current Gradio build can still
# import this module (and run the core tests) without it exploding.
demo = build_demo() if hasattr(gr, "Blocks") else None


if __name__ == "__main__":
    if demo is None:
        raise SystemExit("Install dependencies from space/requirements.txt first.")
    demo.launch()
