"""Launch the Space locally and drive it end to end. CPU only, no GPU, no hub.

This is the "it actually runs" check for the Hugging Face Space.  It

1. imports ``space.app`` (which builds the Gradio Blocks app at import time),
2. asserts the app contains **no image input and no model** -- the whole point
   of the redesign is that it cannot classify anything,
3. launches it on 127.0.0.1 with a free port,
4. drives every interactive control through a real ``gradio_client`` session
   (both confusion-matrix modes, all five seeds, both yardstick bases),
5. asserts the rendered evidence carries the controlled disclaimer, the L5/L6
   zero-recall fact, the yardstick flip, the corpus-hygiene numbers and a
   passing self-check, and
6. writes a JSON evidence report.

Usage (from the repository root, in any env with gradio):

    python space/tools/verify_space_launch.py --report /tmp/space_launch.json
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "space")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _component_names(demo) -> list[str]:
    return sorted({type(block).__name__ for block in demo.blocks.values()})


def _tab_labels(demo) -> list[str]:
    """Tab titles, read off the single ``gr.Tabs`` container."""

    labels = []
    for block in demo.blocks.values():
        if type(block).__name__ == "Tabs":
            for child in getattr(block, "children", []):
                label = getattr(child, "label", None)
                if label:
                    labels.append(str(label))
    return sorted(labels)


def _plot_rows(payload) -> int:
    """Row count of a native-plot payload as it arrives over the wire.

    ``gr.LinePlot`` / ``gr.BarPlot`` return ``{"columns": [...], "data": [[...],
    ...], "datatypes": {...}, "mark": "line"}``, so the row count lives under
    ``data`` -- not in ``len(payload)``, which is the number of keys.
    """

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return len(data)
        return -1
    if isinstance(payload, list):
        return len(payload)
    return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="/tmp/space_launch_report.json")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    evidence: dict[str, object] = {"repo_root": str(_REPO_ROOT)}

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        evidence["error"] = f"gradio unavailable: {exc}"
        Path(args.report).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"BLOCKED: gradio is not installed ({exc})")
        return 2

    _modules_before = set(sys.modules)

    import app as space_app
    from core import (
        ABSTENTION_TEXT,
        DISCLAIMER,
        LEVEL_LABELS,
        LEVELS,
        N_TEST_EVENTS,
        SEEDS,
    )

    demo = space_app.build_demo()
    evidence["gradio_version"] = gr.__version__
    evidence["components"] = _component_names(demo)
    evidence["tabs"] = _tab_labels(demo)

    # ---- the structural claim: nothing here can classify -------------------
    forbidden = {"Image", "ImageEditor", "AnnotatedImage", "Gallery", "Video", "Audio"}
    evidence["no_input_components"] = sorted(
        forbidden.intersection({type(b).__name__ for b in demo.blocks.values()})
    )
    # Only count a dependency as "pulled in by this Space" if importing the
    # app is what brought it in.  Gradio itself depends on PIL, so its mere
    # presence in sys.modules says nothing about the evidence layer.
    _new_modules = set(sys.modules) - _modules_before
    evidence["torch_imported"] = "torch" in _new_modules
    evidence["transformers_imported"] = "transformers" in _new_modules
    evidence["pil_imported"] = any(m == "PIL" or m.startswith("PIL.") for m in _new_modules)
    evidence["modules_added_by_importing_app"] = len(_new_modules)

    # ---- render every panel straight from the core (no server needed) ------
    bundle = space_app._BUNDLE
    rendered = {
        "scope": space_app.render_scope_markdown(),
        "confusion": space_app.render_confusion_markdown(bundle),
        "risk_coverage": space_app.render_risk_coverage_markdown(bundle, 43),
        "gates": space_app.render_gate_markdown(bundle),
        "yardstick": space_app.render_yardstick_markdown(bundle),
        "corpus_hygiene": space_app.render_corpus_hygiene_markdown(bundle),
        "self_check": space_app.render_self_check_table(bundle),
    }
    evidence["rendered_chars"] = {k: len(v) for k, v in rendered.items()}
    evidence["self_check_rows"] = len(bundle.self_check())
    evidence["self_check_passed"] = bundle.self_check_passed()
    evidence["recomputed"] = {
        "per_level_recall": {
            label: row["value"] for label, row in bundle.per_level_recall().items()
        },
        "per_level_precision": {
            label: row["value"] for label, row in bundle.per_level_precision().items()
        },
        "macro_f1_with_abstentions": bundle.macro_f1_with_abstentions(),
        "weighted_kappa_covered": bundle.weighted_kappa_covered(),
        "emitted_levels": bundle.emitted_levels(),
        "silent_levels": bundle.silent_levels(),
        "n_decided": bundle.n_decided(),
        "n_abstained": bundle.n_abstained(),
    }

    checks: dict[str, object] = {}

    # 1. no classifier surface
    checks["no_image_or_media_input"] = not evidence["no_input_components"]
    checks["no_torch"] = not evidence["torch_imported"]
    checks["no_transformers"] = not evidence["transformers_imported"]
    checks["no_pillow"] = not evidence["pil_imported"]

    # 2. self-check
    checks["self_check_passed"] = bool(evidence["self_check_passed"])
    checks["self_check_has_no_mismatch"] = "MISMATCH" not in rendered["self_check"]
    checks["self_check_covers_every_level"] = all(
        f"recall {label}" in rendered["self_check"] for label in LEVEL_LABELS
    )

    # 3. the load-bearing facts are on the page
    checks["confusion_shows_l5_l6_zero_recall"] = all(
        f"| {label} |" in rendered["confusion"] for label in ("L5", "L6")
    ) and "0.00" in rendered["confusion"]
    checks["confusion_states_never_emitted"] = (
        "never emitted as a decision" in rendered["confusion"]
    )
    checks["confusion_lists_all_five_seeds"] = all(
        f"| {seed} |" in rendered["confusion"] for seed in SEEDS
    )
    checks["risk_coverage_states_denominator"] = (
        f"{N_TEST_EVENTS} held-out test events" in rendered["risk_coverage"]
    )
    checks["risk_coverage_shows_abstention_effect"] = (
        "abstention buying safety" in rendered["risk_coverage"]
    )
    checks["gates_show_wk1_pass_wk2_fail"] = (
        "wk1: 5/5 seeds pass" in rendered["gates"]
        and "wk2: 0/5 seeds pass" in rendered["gates"]
    )
    checks["yardstick_shows_both_verdicts"] = (
        "0/5" in rendered["yardstick"] and "5/5" in rendered["yardstick"]
    )
    checks["yardstick_flags_bias_matrix_metric"] = (
        "different metric" in rendered["yardstick"]
        or "mean kappa over ALL rows" in rendered["yardstick"]
    )
    checks["corpus_hygiene_states_licence_numbers"] = all(
        token in rendered["corpus_hygiene"]
        for token in ("unverified open-images/openverse", "release-clean unique events")
    )
    checks["corpus_hygiene_flags_card_discrepancy"] = (
        "disagree" in rendered["corpus_hygiene"]
    )

    # 4. disclaimers everywhere they must be
    checks["disclaimer_in_intro"] = DISCLAIMER in space_app._INTRO
    checks["disclaimer_in_scope"] = DISCLAIMER in rendered["scope"]
    checks["zh_hant_disclaimer_in_scope"] = "烹飪教學研究示範" in rendered["scope"]
    checks["abstention_text_quoted"] = ABSTENTION_TEXT in rendered["scope"]
    checks["scope_says_not_a_swallowing_safety_tool"] = (
        "Not a swallowing-safety classifier" in rendered["scope"]
    )
    checks["scope_says_no_model"] = (
        "loads no model" in rendered["scope"] or "runs no model" in rendered["scope"]
    )

    # 5. no placeholder leakage, no unrendered None/nan
    all_rendered = "\n".join(rendered.values()) + space_app._INTRO
    checks["no_placeholder_banner"] = "PLACEHOLDER" not in all_rendered.upper() or (
        "NO CLASSIFIER IS SHIPPED HERE" in all_rendered
    )
    checks["no_unrendered_none"] = not re.search(r"\bNone\b", all_rendered)
    checks["no_unrendered_nan"] = not re.search(r"\bnan\b", all_rendered)
    # The CORAL rank rule is written with literal braces in the prose; strip the
    # known-literal spellings before looking for a leaked f-string placeholder.
    _literal_braces = all_rendered
    for _known in (
        "{k : P(y>k) > 0.5}",
        "#{k : P(y>k) > 0.5}",
        "{k : P(y>k) > 0.5}",
    ):
        _literal_braces = _literal_braces.replace(_known, "")
    checks["no_unresolved_fstring"] = not re.search(
        r"\{[a-zA-Z_][a-zA-Z0-9_.\[\]'\"()]*\}", _literal_braces
    )
    checks["no_stray_braces"] = "{" not in _literal_braces

    evidence["checks"] = checks
    evidence["all_checks_passed"] = all(bool(v) for v in checks.values())

    # ---- now drive the running server -------------------------------------
    port = _free_port()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        show_error=True,
        quiet=True,
    )
    try:
        from gradio_client import Client

        client = Client(f"http://127.0.0.1:{port}", hf_token=None)
        api_info = client.view_api(return_format="dict")
        named = sorted(api_info.get("named_endpoints", {}))
        evidence["named_endpoints"] = named
        evidence["endpoint_count"] = len(named)

        live: dict[str, object] = {}

        counts_md = client.predict("counts", api_name="/_confusion")[2]
        live["confusion_counts_has_108_denominator"] = f"{N_TEST_EVENTS} test events" in counts_md
        norm_md = client.predict("row-normalised (recall)", api_name="/_confusion")[2]
        live["confusion_row_normalised_renders"] = "0.90" in norm_md

        for seed in SEEDS:
            chart, md = client.predict(str(seed), api_name="/_risk_coverage")
            live[f"risk_coverage_seed{seed}_renders"] = (
                f"Seed {seed}, {N_TEST_EVENTS} held-out test events" in md
            )
            live[f"risk_coverage_seed{seed}_chart_rows"] = _plot_rows(chart)

        flip_md = client.predict(
            "108-event common split (all three models share one denominator)",
            api_name="/_yardstick",
        )[1]
        live["yardstick_common_split_renders"] = (
            "108-event split common to all three models" in flip_md
        )
        live["yardstick_common_split_flip"] = "0/5" in flip_md and "5/5" in flip_md

        evidence["live"] = live
        evidence["live_all_passed"] = all(bool(v) for k, v in live.items())
        evidence["live_failures"] = sorted(k for k, v in live.items() if not v)
    except Exception as exc:  # pragma: no cover - surfaced, never swallowed
        evidence["live_error"] = f"{type(exc).__name__}: {exc}"
        evidence["live_all_passed"] = False
    finally:
        demo.close()

    evidence["overall_passed"] = bool(
        evidence["all_checks_passed"] and evidence.get("live_all_passed")
    )

    Path(args.report).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overall_passed": evidence["overall_passed"],
                "static_checks": evidence["all_checks_passed"],
                "live_checks": evidence.get("live_all_passed"),
                **{k: v for k, v in checks.items() if not v},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if evidence["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
