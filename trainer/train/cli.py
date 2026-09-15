from __future__ import annotations

import argparse
import json

from .config import load_config
from .engine import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDDSI L3-L7 ordinal visual baselines")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    summaries = run_experiment(load_config(args.config))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

