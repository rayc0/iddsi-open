from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .client import ExplanationClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain structured visual outputs without changing the classifier decision")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Structured classifier JSON")
    parser.add_argument("--event-id", help="Select one event when --input is a list")
    parser.add_argument("--language", choices=["yue", "zh", "en"], default="yue")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not args.event_id:
            parser.error("--event-id is required when --input contains a list")
        matches = [item for item in data if str(item.get("event_id")) == args.event_id]
        if len(matches) != 1:
            parser.error(f"expected one matching event_id, found {len(matches)}")
        data = matches[0]
    text = ExplanationClient(**config["explain"]).generate(data, args.language)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
