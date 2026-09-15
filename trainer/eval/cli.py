from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.utils import write_json

from .human import human_inter_rater_baseline, load_human_ratings
from .metrics import evaluate_predictions
from .report import write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated IDDSI L3-L7 predictions")
    parser.add_argument("--predictions", required=True, help="NPZ with probabilities and labels arrays")
    parser.add_argument("--threshold", required=True, type=float, help="Validation-calibrated confidence threshold")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--human-ratings", help="Optional CSV: event_id,rater_a_level,rater_b_level")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    arrays = np.load(args.predictions, allow_pickle=False)
    result = evaluate_predictions(
        arrays["probabilities"], arrays["labels"], args.threshold,
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
    )
    print(json.dumps({"report": str(output / "report.md"), "metrics": str(output / "metrics.json")}))


if __name__ == "__main__":
    main()
