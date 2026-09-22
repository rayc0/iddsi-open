#!/usr/bin/env python3
"""W35: machine-readable parser for the output of ops/health_check.sh.

Usage:
    bash ops/health_check.sh | python3 ops/parse_health.py
    python3 ops/parse_health.py /path/to/saved_output.txt

Sections are delimited by ``=== SECTION: <name> ===`` lines. Machine lines are
TAB-separated and start with a keyword (CONTAINER, MODEL, GPU_MEM, ...). Lines
starting with ``#`` are human commentary and ignored. Missing or degraded
sections parse to their default (empty / null) values instead of failing.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, Optional

SECTION_RE = re.compile(r"^=== SECTION: ([A-Za-z0-9_]+) ===\s*$")


def _int_or_none(value: str) -> Optional[int]:
    value = value.strip()
    if not value or value.upper() in ("N/A", "NA", "UNAVAILABLE", "[N/A]"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _clean(value: str) -> str:
    return value.strip().strip("[]")


def parse(text: str) -> Dict[str, Any]:
    """Parse health_check.sh output text into a dict (JSON-serializable)."""
    result: Dict[str, Any] = {
        "containers": {},
        "iddsi_train_running": False,
        "deepseek_v4_running": False,
        "models": {},
        "flux": {
            "safetensors_found": False,
            "safetensors_bytes": None,
            "temp_files": None,
        },
        "gpu": {
            "total_mb": None,
            "used_mb": None,
            "free_mb": None,
            "compute_apps": [],
            "host_ram": None,
        },
        "heartbeat_age_minutes": None,
        "heartbeat_log": None,
        "last_night_summary": None,
        "sections_seen": [],
    }
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        match = SECTION_RE.match(line)
        if match:
            current = match.group(1)
            if current not in result["sections_seen"]:
                result["sections_seen"].append(current)
            continue
        if current is None:
            continue
        parts = line.split("\t")
        keyword = parts[0]

        if current == "containers" and keyword == "CONTAINER" and len(parts) >= 3:
            name, status = parts[1], parts[2]
            result["containers"][name] = status
            if name == "iddsi-train" and status.startswith("Up"):
                result["iddsi_train_running"] = True
            elif name == "iddsi-train":
                result["iddsi_train_running"] = False
            if name == "deepseek-v4":
                result["deepseek_v4_running"] = status.startswith("Up")

        elif current == "models":
            if keyword == "MODEL" and len(parts) >= 4:
                result["models"][parts[1]] = {
                    "size_bytes": _int_or_none(parts[2]) or 0,
                    "status": parts[3],
                }
            elif keyword == "FLUX_SAFETENSORS" and len(parts) >= 3:
                result["flux"]["safetensors_found"] = parts[1] == "FOUND"
                result["flux"]["safetensors_bytes"] = _int_or_none(parts[2])
            elif keyword == "FLUX_TEMP_FILES" and len(parts) >= 2:
                result["flux"]["temp_files"] = _int_or_none(parts[1])

        elif current == "gpu":
            if keyword == "GPU_MEM" and len(parts) >= 2:
                fields = [f for f in _clean(parts[1]).split(",") if f != ""]
                if len(fields) >= 3:
                    result["gpu"]["total_mb"] = _int_or_none(fields[0])
                    result["gpu"]["used_mb"] = _int_or_none(fields[1])
                    result["gpu"]["free_mb"] = _int_or_none(fields[2])
            elif keyword == "GPU_APP" and len(parts) >= 4:
                result["gpu"]["compute_apps"].append(
                    {
                        "pid": _int_or_none(parts[1]),
                        "process_name": parts[2],
                        "used_mb": _int_or_none(parts[3]),
                    }
                )
            elif keyword == "HOST_RAM" and len(parts) >= 7:
                result["gpu"]["host_ram"] = {
                    "total": parts[1],
                    "used": parts[2],
                    "free": parts[3],
                    "shared": parts[4],
                    "buff_cache": parts[5],
                    "available": parts[6],
                }

        elif current == "heartbeat" and keyword == "HEARTBEAT" and len(parts) >= 2:
            if parts[1] != "none":
                result["heartbeat_log"] = parts[1]
                result["heartbeat_age_minutes"] = (
                    _int_or_none(parts[2]) if len(parts) >= 3 else None
                )

        elif current == "night_summary" and keyword == "NIGHT_SUMMARY" and len(parts) >= 2:
            result["last_night_summary"] = None if parts[1] == "none" else parts[1]

    return result


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        if argv[1] in ("-h", "--help"):
            print(__doc__)
            return 0
        with open(argv[1], "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    print(json.dumps(parse(text), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
