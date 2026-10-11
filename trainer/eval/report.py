from __future__ import annotations

from pathlib import Path
from typing import Any


DISCLAIMER = (
    "Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; "
    "does not perform official IDDSI tests, assess swallowing, determine suitability for any person, "
    "or decide whether food is safe to consume."
)

YARDSTICK_NOTE = (
    "Label set for every kappa in this report (H-YARDSTICK): {label_set}. "
    "A weighted kappa is not interpretable without the label set it was measured against - the same "
    "checkpoints fail the wk1>=0.70 gate on one label set and pass it on another. "
    "For reference, the v0.4/v0.5 numbers were measured against the v0.5 consensus and fail the "
    "wk1>=0.70 gate; the same models measured against the v0.7 full-coverage consensus pass "
    "(v0.7: 0.838-0.860)."
)

YARDSTICK_UNSTATED = (
    "Label set for every kappa in this report (H-YARDSTICK): NOT STATED by the caller. "
    "Pass label_set= to render_markdown_report. A weighted kappa without its label set is not "
    "interpretable: the same checkpoints fail the wk1>=0.70 gate on the v0.5 consensus and pass it "
    "on the v0.7 full-coverage consensus (v0.7: 0.838-0.860)."
)


def _fmt(value: float) -> str:
    return "NA" if value != value else f"{value:.4f}"


def render_markdown_report(
    result: dict[str, Any],
    *,
    title: str,
    synthetic: bool,
    calibration: dict[str, Any] | None = None,
    human: dict[str, Any] | None = None,
    hub: dict[str, Any] | None = None,
    label_set: str | None = None,
) -> str:
    lines = [f"# {title}", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA — PIPELINE DEVELOPMENT ONLY — DO NOT RELEASE OR REPORT AS PERFORMANCE.**", ""]
    lines += [f"> {DISCLAIMER}", "", "## Evaluation scope", ""]
    lines += [
        f"- Events: {result['n']}",
        f"- Confidence threshold calibrated on validation data: {_fmt(result['threshold'])}",
        "- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.",
        "- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.",
        "- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.",
        "",
        YARDSTICK_NOTE.format(label_set=label_set) if label_set else YARDSTICK_UNSTATED,
    ]
    # H-CORAL reachability.  Without the pre-abstention decode counts, a
    # threshold that abstains on every L4/L5/L6 prediction is indistinguishable
    # in this report from a decoder that cannot emit L4/L5/L6 at all.
    pre = result.get("decoded_level_counts_pre_abstention")
    if pre:
        lines.append(
            "- Rank-rule decode before abstention: "
            + ", ".join(f"{name} {pre[name]}" for name in ("L3", "L4", "L5", "L6", "L7") if name in pre)
        )
    lines.append(
        "- Interior levels L4-L6 still reachable after abstention: "
        + (
            f"yes ({result.get('interior_decodes_accepted', 0)} of "
            f"{result.get('interior_decodes_total', 0)} interior decodes cleared the threshold)"
            if result.get("interior_levels_reachable")
            else "NO — every L4/L5/L6 decode fell below the confidence threshold"
        )
    )
    lines += [
        "",
        "## Metrics", "",
        "| Metric | Estimate | Bootstrap 95% CI |", "|---|---:|---:|",
    ]
    for name, value in result["metrics"].items():
        interval = result["confidence_intervals_95"].get(name)
        interval_text = f"[{_fmt(interval[0])}, {_fmt(interval[1])}]" if interval else "not computed"
        lines.append(f"| {name} | {_fmt(value)} | {interval_text} |")
    lines += ["", "## Per-level confusion matrix", ""]
    columns = [column.replace("pred_", "") for column in result["confusion_columns"]]
    lines += ["| Truth ↓ / prediction → | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    # Render every row the metrics object carries. Slicing to 5 silently dropped
    # the `true_unclear` row, so the printed table stopped summing to `n` above
    # while metrics.json kept the count.
    for name, row in zip(result["confusion_rows"], result["confusion_matrix"]):
        lines.append(f"| {name.replace('true_', '')} | " + " | ".join(str(value) for value in row) + " |")
    if calibration:
        lines += ["", "## Validation calibration", ""]
        lines += [
            f"- Temperature: {_fmt(float(calibration['temperature']))}",
            f"- Threshold selection target met: `{str(calibration['target_met']).lower()}`",
            f"- Selection: {calibration['selection_reason']}",
        ]
    if human:
        lines += ["", "## Human inter-rater baseline", ""]
        lines += [
            f"- Paired ratings: {human['n']}",
            f"- Raw agreement: {_fmt(human['raw_agreement'])}",
            f"- Quadratic weighted kappa: {_fmt(human['weighted_kappa'])}",
            (
                "- Label set for that kappa (H-YARDSTICK): agreement between the two human raters, "
                "not a model scored against a label set, so it characterises the yardstick itself. "
                "It is not comparable to the model kappa above."
            ),
        ]
        if "rater_vs_physical_test" in human:
            lines += [
                "",
                "| Photo-only rater vs physical-test label | Raw agreement | Weighted kappa | Dangerous under-classification FNR |",
                "|---|---:|---:|---:|",
            ]
            for name, values in human["rater_vs_physical_test"].items():
                lines.append(
                    f"| {name} | {_fmt(values['raw_agreement'])} | {_fmt(values['weighted_kappa'])} | "
                    f"{_fmt(values['dangerous_underclassification_fnr'])} |"
                )
        lines += ["", f"- {human['note']}"]
    lines += [
        "", "## Risk–coverage data", "",
        (
            "The machine-readable `metrics.json` contains the full risk–coverage curve. "
            "Threshold selection on a test set is prohibited."
        ),
    ]
    if hub:
        lines += ["", "## Release provenance", ""]
        lines += [
            f"- Fetched from the Hub: `{hub.get('repo')}` at revision `{hub.get('revision')}` "
            f"(resolved commit `{hub.get('resolved_commit')}`).",
            f"- Served by endpoint `{hub.get('endpoint')}` — {hub.get('endpoint_provenance')}.",
            f"- Requested at {hub.get('requested_at_utc')} UTC; {hub.get('n_files')} file(s), "
            f"{hub.get('total_bytes')} bytes.",
        ]
        if hub.get("commit_mismatch"):
            lines += [
                "- **The snapshot that served these bytes is not the commit the endpoint reported.** "
                "Re-pin the revision before quoting any figure from this report.",
            ]
        for item in hub.get("files", []):
            lines += [
                f"- `{item['filename']}` — {item['bytes']} bytes, sha256 `{item['sha256']}`"
                + (", already in local cache" if item.get("already_cached") else ""),
            ]
        lines += [
            "",
            "These bytes came from the published release, not from a local checkout, so the "
            "figures above are computed on exactly what a third party can download.",
        ]
    lines += [
        "", "## Interpretation boundary", "",
        (
            "Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure "
            "flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical "
            "IDDSI test."
        ),
        "", "This project is not the official IDDSI website and is not endorsed by IDDSI.", "",
    ]
    return "\n".join(lines)


def write_markdown_report(path: str | Path, *args: Any, **kwargs: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(*args, **kwargs), encoding="utf-8")
