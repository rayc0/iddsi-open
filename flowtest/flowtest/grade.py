"""Command line interface: python -m flowtest.grade VIDEO --out result.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import FlowTestConfig, grade_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic IDDSI 10 mL / 10 s Flow-Test video grader")
    parser.add_argument("video", type=Path, help="input video")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument(
        "--fiducial-optional",
        action="store_true",
        help="grade without the ID-1 fiducial: uncalibrated mode, flagged with the 'uncalibrated_scale_mode' code",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = FlowTestConfig(fiducial_required=not args.fiducial_optional)
    try:
        result = grade_video(args.video, config=config)
    except (OSError, ValueError) as error:
        payload = {
            "video": str(args.video),
            "status": "error",
            "error": str(error),
        }
        exit_code = 2
    else:
        payload = result.to_dict()
        exit_code = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
