"""Launch the Space locally and drive it end to end. CPU only, no GPU, no hub.

This is the "it actually runs" check for the Hugging Face Space. It

1. imports ``space.app`` (which builds the Gradio Blocks app at import time),
2. launches it on 127.0.0.1 with a free port,
3. sends a real photograph-shaped PNG through the running server,
4. asserts the response carries the controlled disclaimer and abstains rather
   than naming a level, and
5. writes a JSON evidence report.

Usage (from the repository root, inside the iddsi-train container or any env
with gradio + PIL):

    python space/tools/verify_space_launch.py --report /tmp/space_launch.json
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "space"), str(_REPO_ROOT / "trainer")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_photo(path: Path) -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(20261012)
    array = (rng.random((224, 224, 3)) * 255).astype("uint8")
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _normalise_chart(value) -> list[tuple[str, float]]:
    """Coerce whatever the BarPlot endpoint returns into (label, value) pairs.

    ``gr.BarPlot`` is fed a pandas DataFrame; over the wire it can arrive as a
    list of row dicts, a dict of columns, or a list of row lists depending on
    the gradio-client version.  All three are accepted.
    """

    if isinstance(value, dict):
        columns = value.get("columns")
        data = value.get("data")
        if columns and data is not None:  # gradio native-plot payload
            columns = list(columns)
            return [
                (str(row[columns.index("level")]), float(row[columns.index("probability")]))
                for row in data
            ]
        if "headers" in value and "data" in value:  # list-of-lists with a header row
            headers = list(value["headers"])
            return [
                (str(row[headers.index("level")]), float(row[headers.index("probability")]))
                for row in value["data"]
            ]
        labels = value.get("level") or value.get("x")
        numbers = value.get("probability") or value.get("y")
        if labels is None or numbers is None:
            raise AssertionError(f"unrecognised chart payload keys: {sorted(value)}")
        return [(str(label), float(number)) for label, number in zip(labels, numbers)]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [(str(row["level"]), float(row["probability"])) for row in value]
    if isinstance(value, list):
        return [(str(row[0]), float(row[1])) for row in value]
    raise AssertionError(f"unrecognised chart payload type: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="/tmp/space_launch_report.json")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    evidence: dict[str, object] = {"repo_root": str(_REPO_ROOT)}

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        evidence["error"] = f"gradio unavailable: {exc}"
        Path(args.report).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"BLOCKED: gradio is not installed ({exc})")
        return 2

    from app import build_demo  # noqa: PLC0415  (space/ is on sys.path)

    demo = build_demo()
    evidence["gradio_version"] = gr.__version__
    evidence["components"] = sorted(
        {type(block).__name__ for block in demo.blocks.values()}
    )

    port = _free_port()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        show_error=True,
        quiet=True,
    )

    from gradio_client import Client, handle_file  # noqa: PLC0415

    client = Client(f"http://127.0.0.1:{port}", hf_token=None)
    api_info = client.view_api(return_format="dict")
    named = sorted(api_info.get("named_endpoints", {}))
    evidence["named_endpoints"] = named

    with tempfile.TemporaryDirectory() as tmp:
        photo = _write_photo(Path(tmp) / "photo.png")
        target = next(
            (name for name in named if name.endswith("/predict") or name == "/predict"),
            named[0] if named else "/predict",
        )
        result = client.predict(handle_file(str(photo)), api_name=target)
        evidence["endpoint_used"] = target
        evidence["raw_result_types"] = [type(item).__name__ for item in result]
        summary, chart, verdict = result[0], result[1], result[2]

    from core import ABSTENTION_TEXT, DISCLAIMER, LEVELS  # noqa: PLC0415

    rows = _normalise_chart(chart)
    labels = [label for label, _ in rows]
    values = [number for _, number in rows]

    checks = {
        "summary_contains_disclaimer": DISCLAIMER in summary,
        "summary_contains_zh_hant_disclaimer": "烹飪教學研究示範" in summary,
        "summary_contains_placeholder_banner": "PLACEHOLDER" in summary,
        "verdict_is_abstention": verdict.strip() == ABSTENTION_TEXT,
        "chart_has_one_row_per_level": len(rows) == len(LEVELS),
        "chart_rows_are_levels_in_order": labels == [f"L{level}" for level in LEVELS],
        "chart_probabilities_sum_to_one": abs(sum(values) - 1.0) < 1e-6,
        "chart_probabilities_are_uniform": all(
            abs(number - 1.0 / len(LEVELS)) < 1e-9 for number in values
        ),
        "no_level_named_in_verdict": not any(f"L{level}" in verdict for level in LEVELS),
    }
    evidence["checks"] = checks
    evidence["all_checks_passed"] = all(checks.values())
    evidence["chart_rows"] = [{"label": label, "value": number} for label, number in rows]
    evidence["verdict"] = verdict
    evidence["summary_excerpt"] = summary[:1200]

    Path(args.report).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"all_checks_passed": evidence["all_checks_passed"], **checks}, indent=2))
    demo.close()
    return 0 if evidence["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
