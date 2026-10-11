from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.utils import write_json

from . import hub_source
from .human import human_inter_rater_baseline, load_human_ratings
from .metrics import evaluate_predictions
from .report import write_markdown_report


def _add_hub_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the Hub-consumption options shared by every artifact argument."""
    parser.add_argument(
        "--hub-repo",
        default=None,
        help=(
            "Hub repo for hub: references (default: IDDSI_HUB_REPO or "
            f"{hub_source.RELEASE_REPO})"
        ),
    )
    parser.add_argument(
        "--hub-revision",
        default=None,
        help=(
            "Pinned revision for hub: references: a release tag such as v0.7 "
            f"(default: IDDSI_HUB_REVISION or {hub_source.DEFAULT_REVISION})"
        ),
    )
    parser.add_argument(
        "--hub-endpoint",
        default=None,
        help=(
            "Hub endpoint for hub: references. Default order: IDDSI_HF_ENDPOINT, "
            "HF_ENDPOINT, then a reachability probe of huggingface.co followed "
            f"by {hub_source.MIRROR_ENDPOINT} (the CN mirror)"
        ),
    )
    parser.add_argument(
        "--hub-cache-dir",
        default=None,
        help=f"HF cache directory for hub: references (default: IDDSI_HUB_CACHE or the huggingface_hub default)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated IDDSI L3-L7 predictions")
    parser.add_argument("--predictions", required=True, help="NPZ with probabilities and labels arrays, or hub:<repo>@<rev>:<rfilename>")
    parser.add_argument("--threshold", required=True, help="Validation-calibrated confidence threshold, a metrics JSON, or a hub: reference")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--human-ratings", help="Optional CSV: event_id,rater_a_level,rater_b_level")
    parser.add_argument("--synthetic", action="store_true")
    _add_hub_arguments(parser)
    args = parser.parse_args()

    hub_kwargs = {
        "repo": args.hub_repo,
        "revision": args.hub_revision,
        "endpoint": args.hub_endpoint,
        "cache_dir": args.hub_cache_dir,
    }
    predictions_path, predictions_receipt = hub_source.resolve_local(
        args.predictions, kind="predictions", **hub_kwargs
    )
    threshold_path, threshold_receipt = hub_source.resolve_local(
        args.threshold, kind="threshold", **hub_kwargs
    )
    receipt = hub_source.merge_receipts([predictions_receipt, threshold_receipt])
    if receipt is not None:
        print(receipt.summary_line())

    # A threshold may be given as a number, as a metrics JSON carrying a
    # "threshold" key, or as a hub: reference to such a JSON.
    raw_threshold = args.threshold
    try:
        threshold = float(raw_threshold)
    except ValueError:
        threshold = hub_source.threshold_from_file(threshold_path)

    arrays = np.load(predictions_path, allow_pickle=False)
    result = evaluate_predictions(
        arrays["probabilities"], arrays["labels"], threshold,
        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
    )
    human = (
        human_inter_rater_baseline(
            load_human_ratings(args.human_ratings),
            arrays["event_ids"].tolist() if "event_ids" in arrays else None,
            arrays["labels"] if "event_ids" in arrays else None,
        )
        if args.human_ratings else None
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    write_markdown_report(
        output / "report.md",
        result,
        title="IDDSI-Open held-out evaluation",
        synthetic=args.synthetic,
        human=human,
        hub=receipt.to_dict() if receipt is not None else None,
    )
    if receipt is not None:
        hub_source.write_receipt(output / "hub_receipt.json", receipt)
    print(json.dumps({
        "report": str(output / "report.md"),
        "metrics": str(output / "metrics.json"),
        **({"hub_receipt": str(output / "hub_receipt.json")} if receipt is not None else {}),
    }))


if __name__ == "__main__":
    main()
