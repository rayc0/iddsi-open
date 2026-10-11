"""Hugging Face Space: try the IDDSI-Open SigLIP-2 ordinal texture model.

Sdk front matter lives in ``README.md`` next to this file (``sdk: gradio``).

What this Space does
--------------------
Loads the SigLIP-2 base vision tower plus the monotone CORAL ordinal head,
accepts one uploaded food photograph, and returns the L3-L7 probability
distribution together with the controlled disclaimer.

What it does NOT do
-------------------
It performs no official IDDSI test, makes no swallowing-safety assessment,
makes no determination about any person, and issues no safe/unsafe verdict.  It
is a culinary-education research demonstration.  ``space/core.py`` is
fail-closed: if anything is missing or broken it abstains with ``unclear —
perform physical test`` and emits no level.
Weights status (2026-10-12)
---------------------------
The v0.7 weights are WITHHELD (``hf/model_README.md``, "Why the weights are
withheld").  With no checkpoint reachable this Space therefore runs the
``placeholder_no_weights`` path: a uniform L3-L7 distribution, a prominent
PLACEHOLDER banner, and no decoded level.  Point ``IDDSI_SPACE_CHECKPOINT`` at
a checkpoint file to switch it to the weighted path -- the app already
implements it, and ``space/tests/test_space_core.py`` exercises it against a
locally-built random-init checkpoint.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

try:  # Space layout: app.py sits in space/ and imports the sibling module.
    from core import (
        ABSTENTION_TEXT,
        DEFAULT_ABSTENTION_THRESHOLD,
        DISCLAIMER,
        DISCLAIMER_ZH_HANT,
        LEVELS,
        PLACEHOLDER_BANNER,
        describe_configuration,
        predict,
    )
except ImportError:  # Repo-root layout: space/ is a package on sys.path.
    from space.core import (  # type: ignore[no-redef]
        ABSTENTION_TEXT,
        DEFAULT_ABSTENTION_THRESHOLD,
        DISCLAIMER,
        DISCLAIMER_ZH_HANT,
        LEVELS,
        PLACEHOLDER_BANNER,
        describe_configuration,
        predict,
    )

_CONFIG = describe_configuration()
_TITLE = "IDDSI-Open — visual texture-level research demo (L3–L7)"
_DESCRIPTION = f"""
Upload one food photograph. The demo reports a probability distribution over
**IDDSI texture levels L3–L7** — L3 Liquidised/moderately thick, L4 Pureed,
L5 Minced & moist, L6 Soft & bite-sized, L7 Regular — estimated from
**appearance only**.

**This is not a swallowing-safety tool.** It performs no IDDSI test, measures no
flow, hardness, adhesiveness or cohesiveness, and makes no determination about
any person. {DISCLAIMER}

*{DISCLAIMER_ZH_HANT}*
"""

_PLACEHOLDER_NOTE = f"""
> ⚠️ **{PLACEHOLDER_BANNER}**
"""

_HOW_IT_WORKS = f"""
### How the estimate is produced

1. **Backbone.** The photo is resized to 224×224 and encoded by
   `google/siglip2-base-patch16-224`.
2. **Ordinal head.** A CORAL head predicts monotone cut-point logits
   `P(y > k)` for k = L3..L6, converted to class probabilities.
3. **Decode.** The level is read with the ordinal **rank rule**
   `level = #{{k : P(y>k) > 0.5}}`, *not* argmax over the class masses. Argmax
   over sigmoid differences provably cannot return an interior level for any
   input — hazard **H-CORAL** in `hf/HAZARD_LOG.md`.
4. **Abstention.** If the confidence on the decoded level is below
   {DEFAULT_ABSTENTION_THRESHOLD:.4f} the demo answers **{ABSTENTION_TEXT}**
   instead of a level. The threshold is the most conservative of the five v0.7
   validation calibrations (`results/v07_release/v07_seed101_calibration.json`).

### Why middle levels are the hard part

L3/L4 is a *viscosity* boundary and L4/L5 is an *absolute particle-size*
boundary. Neither is recoverable from a still photograph: a photograph carries
no scale fiducial and no flow. Independent labeller families agree on an IDDSI
level from a web photo only about three times in four (~74.5%,
251/337 doubled-coverage events), so every number here is measured against
label noise, not ground truth.

### What is actually validated

Nothing in this demo is validated against physical IDDSI tests. The
physically-tested desk-specimen protocol (300 specimens, one operator,
`docs/CAPTURE_PROTOCOL.md`) has never been run. Until it is, treat the output as
a research signal about visual similarity — never as a texture determination.
"""


def _distribution_frame(output) -> "pd.DataFrame":
    """One row per IDDSI level for ``gr.BarPlot`` (which takes a DataFrame).

    Rows are kept in soft -> thick order (L3 first) so the chart reads as an
    ordinal scale rather than a sorted category list.
    """

    return pd.DataFrame(
        {
            "level": [f"L{level}" for level in output.levels],
            "probability": [
                round(float(output.probabilities.get(level, 0.0)), 4)
                for level in output.levels
            ],
        }
    )


def run(image) -> tuple[str, object, str, str]:
    """Gradio callback. Returns (summary markdown, bar plot, level text, disclaimer)."""

    output = predict(image)
    return (
        output.to_markdown(),
        _distribution_frame(output),
        output.verdict_plain,
        DISCLAIMER,
    )


def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks app (kept out of module scope so tests can
    import this module without a running UI)."""

    with gr.Blocks(title=_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {_TITLE}")
        gr.Markdown(_DESCRIPTION)
        gr.Markdown(_PLACEHOLDER_NOTE)

        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(
                    label="Food photograph",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                )
                submit = gr.Button("Estimate texture level", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown(label="Result")
                chart = gr.BarPlot(
                    x="level",
                    y="probability",
                    title="L3–L7 probability distribution",
                    y_lim=[0, 1],
                    height=260,
                    tooltip=["level", "probability"],
                )
                verdict = gr.Textbox(
                    label="Decoded level",
                    interactive=False,
                    value=ABSTENTION_TEXT,
                )

        submit.click(run, inputs=photo, outputs=[summary, chart, verdict])
        photo.change(run, inputs=photo, outputs=[summary, chart, verdict])

        gr.Markdown("---")
        gr.Markdown(f"### Disclaimer\n\n**{DISCLAIMER}**")
        gr.Markdown(f"### 免責聲明\n\n**{DISCLAIMER_ZH_HANT}**")
        with gr.Accordion("How this works, and what it cannot tell you", open=False):
            gr.Markdown(_HOW_IT_WORKS)
            gr.Markdown(
                "### Scope\n\n"
                "- In scope: research preview and culinary-education use only.\n"
                "- Out of scope: any safe/unsafe verdict; any dietary, clinical or "
                "patient claim; any assessment of swallowing, aspiration risk or "
                "suitability for a person; any instruction to prepare, modify or "
                "prescribe a texture.\n"
                "- Not IDDSI-endorsed. This project is independent of IDDSI.\n"
                f"- The head offers L{LEVELS[0]}–L{LEVELS[-1]} only. Anything thinner "
                "than L3 (IDDSI L0–L2) is outside what this model can express, and a "
                "photo of it is reported as an L3-L7 distribution that means nothing.\n"
            )
            gr.Markdown(
                "### Resolved configuration\n\n"
                f"- Levels: L{LEVELS[0]}–L{LEVELS[-1]}\n"
                f"- Decode rule: `{_CONFIG['decode_rule']}`\n"
                f"- Abstention threshold: `{_CONFIG['abstain_threshold']}`\n"
                f"- Checkpoint env var: `{_CONFIG['checkpoint_env']}` → "
                f"`{_CONFIG['default_checkpoint']}` "
                f"(resolved: **{_CONFIG['checkpoint_resolves']}**)\n"
                f"- Backbone env var: `{_CONFIG['backbone_env']}` → "
                f"`{_CONFIG['default_backbone']}`\n"
                f"- CORAL implementation: `{_CONFIG['coral_source']}` "
                "(asserted numerically identical by `space/tests/test_space_core.py`)"
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
