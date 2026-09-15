from __future__ import annotations

from pathlib import Path
from typing import Any


DISCLAIMER = (
    "Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; "
    "does not perform official IDDSI tests, assess swallowing, determine suitability for any person, "
    "or decide whether food is safe to consume."
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
    for name, row in zip(result["confusion_rows"][:5], result["confusion_matrix"][:5]):
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
