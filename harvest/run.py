from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .http import HTTPClient
from .label import Qwen3VLLabeler
from .pipeline import Harvester
from .sources import (
    Nutrition5kSource,
    OpenImagesSource,
    OpenverseSource,
    WikimediaCommonsSource,
)


SOURCE_NAMES = ("wikimedia", "openverse", "open-images", "nutrition5k")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest licence-eligible, privacy-screened real-weak food images."
    )
    parser.add_argument("--max", type=int, default=2000, help="maximum total accepted rows")
    parser.add_argument("--out", type=Path, required=True, help="output dataset directory")
    parser.add_argument(
        "--sources",
        default=",".join(SOURCE_NAMES),
        help=f"comma-separated eligible sources: {', '.join(SOURCE_NAMES)}",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--labeler",
        choices=("qwen", "defer"),
        default="qwen",
        help="'defer' skips VLM weak-labelling (no torch needed); label later on GPU",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument(
        "--provider-cap",
        type=float,
        default=0.4,
        help="maximum fraction of the target accepted rows from one provider (default: 0.4)",
    )
    parser.add_argument("--max-download-bytes", type=int, default=25_000_000)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument(
        "--http-min-interval",
        type=float,
        default=1.0,
        help="minimum seconds between requests to the same host (default: 1.0)",
    )
    parser.add_argument(
        "--user-agent",
        default="iddsi-open-realweak/0.1 (LinguaLeap Tech Limited; research dataset)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = [item.strip() for item in args.sources.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SOURCE_NAMES))
    if unknown:
        print(f"ERROR: unknown source(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if not selected:
        print("ERROR: at least one source is required", file=sys.stderr)
        return 2
    if not 0.0 < args.provider_cap <= 1.0:
        print("ERROR: --provider-cap must be greater than 0 and at most 1", file=sys.stderr)
        return 2
    if args.http_min_interval < 0:
        print("ERROR: --http-min-interval must be non-negative", file=sys.stderr)
        return 2
    http = HTTPClient(
        user_agent=args.user_agent,
        timeout=args.http_timeout,
        min_interval=args.http_min_interval,
    )
    factories = {
        "wikimedia": lambda: WikimediaCommonsSource(http),
        "openverse": lambda: OpenverseSource(http),
        "open-images": lambda: OpenImagesSource(http),
        "nutrition5k": lambda: Nutrition5kSource(http),
    }
    sources = [factories[name]() for name in selected]
    if args.labeler == "defer":
        from .label import DeferLabeler
        labeler = DeferLabeler()
    else:
        labeler = Qwen3VLLabeler(args.model, device=args.device)
    harvester = Harvester(
        http=http,
        sources=sources,
        labeler=labeler,
        phash_threshold=args.phash_threshold,
        max_download_bytes=args.max_download_bytes,
        candidate_multiplier=args.candidate_multiplier,
        provider_cap=args.provider_cap,
    )
    try:
        summary = harvester.run(maximum=args.max, output=args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary.__dict__, sort_keys=True))
    return 0 if summary.existing + summary.accepted >= summary.target else 3


if __name__ == "__main__":
    raise SystemExit(main())
