"""W35 tests for ops/parse_health.py.

Run from repo root:
    python -m pytest ops/tests/test_parse_health.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from parse_health import parse  # noqa: E402

# --- Embedded sample outputs -------------------------------------------------

HAPPY_PATH = """=== SECTION: containers ===
CONTAINER\tiddsi-train\tUp 18 hours
CONTAINER\tdeepseek-v4\tUp About an hour
CONTAINER\tollama\tUp 32 hours
CONTAINER\tgrafana\tExited (3) 2 hours ago
#   (no containers listed)
=== SECTION: models ===
#   hub root: /home/tun/iddsi/hf/hub
MODEL\tmodels--timm--resnet18.a1_in1k\t47185920\tPRESENT
#   models--timm--resnet18.a1_in1k -> 45M (expected >= 47000000 bytes)
MODEL\tmodels--google--siglip2-base-patch16-224\t1610612736\tPRESENT
MODEL\tmodels--Qwen--Qwen3-VL-2B-Instruct\t4294967296\tPARTIAL
MODEL\tmodels--black-forest-labs--FLUX.1-schnell\t12288\tPARTIAL
FLUX_SAFETENSORS\tMISSING\t0
#   WARNING: no FLUX.1-schnell *.safetensors >= 23,000,000,000 bytes found
FLUX_TEMP_FILES\t2
#   leftover: /home/tun/iddsi/hf/hub/.flux1-schnell.safetensors.part
#   leftover: /home/tun/iddsi/hf/hub/.flux1-schnell.safetensors.incomplete
=== SECTION: gpu ===
#   GB10 (GB200-class) unified-memory Spark
GPU_MEM\tN/A,N/A,N/A
#   top GPU consumers:
GPU_APP\t2236\t/srv/whisper/whisper.cpp/build/bin/whisper-server\t3635
GPU_APP\t490743\tllama-server\t100501
HOST_RAM\t121Gi\t117Gi\t1.6Gi\t647Mi\t4.7Gi\t4.5Gi
=== SECTION: heartbeat ===
HEARTBEAT\t/home/tun/iddsi/night_pilot.log\t42
#   newest log /home/tun/iddsi/night_pilot.log modified 42 min ago
=== SECTION: night_summary ===
NIGHT_SUMMARY\tNIGHT_2026-09-01.md
#   first lines of /home/tun/iddsi/reports/NIGHT_2026-09-01.md:
#   # Night summary 2026-09-01
#   400 images generated
=== END ===
"""

DEGRADED_PATH = """=== SECTION: containers ===
[unavailable: docker ps failed (daemon down or permission?)
=== SECTION: models ===
[unavailable: hub dir /home/tun/iddsi/hf/hub not found]
=== SECTION: gpu ===
GPU_MEM\tunavailable
GPU_APP\t2236\tllama-server\t3635
HOST_RAM\tunavailable
=== SECTION: heartbeat ===
HEARTBEAT\tnone
#   no heartbeat logs found
=== SECTION: night_summary ===
NIGHT_SUMMARY\tnone
=== END ===
"""

# Sections entirely absent (script crashed early / truncated pipe).
TRUNCATED = """=== SECTION: containers ===
CONTAINER\tiddsi-train\tExited (1) 5 minutes ago
"""


# --- Happy path --------------------------------------------------------------

def test_happy_path_containers():
    r = parse(HAPPY_PATH)
    assert r["containers"]["iddsi-train"] == "Up 18 hours"
    assert r["containers"]["deepseek-v4"] == "Up About an hour"
    assert r["containers"]["grafana"] == "Exited (3) 2 hours ago"
    assert r["iddsi_train_running"] is True
    assert r["deepseek_v4_running"] is True


def test_stopped_flags():
    text = HAPPY_PATH.replace(
        "CONTAINER\tdeepseek-v4\tUp About an hour",
        "CONTAINER\tdeepseek-v4\tExited (0) 3 hours ago",
    )
    r = parse(text)
    assert r["deepseek_v4_running"] is False
    assert r["iddsi_train_running"] is True


def test_happy_path_models():
    r = parse(HAPPY_PATH)
    m = r["models"]
    assert set(m) == {
        "models--timm--resnet18.a1_in1k",
        "models--google--siglip2-base-patch16-224",
        "models--Qwen--Qwen3-VL-2B-Instruct",
        "models--black-forest-labs--FLUX.1-schnell",
    }
    assert m["models--timm--resnet18.a1_in1k"] == {
        "size_bytes": 47185920,
        "status": "PRESENT",
    }
    assert m["models--black-forest-labs--FLUX.1-schnell"]["status"] == "PARTIAL"
    assert m["models--black-forest-labs--FLUX.1-schnell"]["size_bytes"] == 12288


def test_happy_path_flux():
    r = parse(HAPPY_PATH)
    assert r["flux"]["safetensors_found"] is False
    assert r["flux"]["safetensors_bytes"] == 0
    assert r["flux"]["temp_files"] == 2


def test_flux_safetensors_found():
    text = HAPPY_PATH.replace(
        "FLUX_SAFETENSORS\tMISSING\t0",
        "FLUX_SAFETENSORS\tFOUND\t23657872615",
    )
    r = parse(text)
    assert r["flux"]["safetensors_found"] is True
    assert r["flux"]["safetensors_bytes"] == 23657872615


def test_happy_path_gpu_na_unified_memory():
    r = parse(HAPPY_PATH)
    assert r["gpu"]["total_mb"] is None
    assert r["gpu"]["used_mb"] is None
    assert r["gpu"]["free_mb"] is None
    assert len(r["gpu"]["compute_apps"]) == 2
    assert r["gpu"]["compute_apps"][0]["pid"] == 2236
    assert r["gpu"]["compute_apps"][0]["process_name"] == "/srv/whisper/whisper.cpp/build/bin/whisper-server"
    assert r["gpu"]["compute_apps"][0]["used_mb"] == 3635
    assert r["gpu"]["compute_apps"][1]["used_mb"] == 100501


def test_happy_path_gpu_with_numbers():
    text = HAPPY_PATH.replace(
        "GPU_MEM\tN/A,N/A,N/A", "GPU_MEM\t131072,117000,14072"
    )
    r = parse(text)
    assert r["gpu"]["total_mb"] == 131072
    assert r["gpu"]["used_mb"] == 117000
    assert r["gpu"]["free_mb"] == 14072


def test_happy_path_host_ram():
    r = parse(HAPPY_PATH)
    assert r["gpu"]["host_ram"]["total"] == "121Gi"
    assert r["gpu"]["host_ram"]["available"] == "4.5Gi"


def test_happy_path_heartbeat_and_night():
    r = parse(HAPPY_PATH)
    assert r["heartbeat_age_minutes"] == 42
    assert r["heartbeat_log"] == "/home/tun/iddsi/night_pilot.log"
    assert r["last_night_summary"] == "NIGHT_2026-09-01.md"
    assert r["sections_seen"] == [
        "containers",
        "models",
        "gpu",
        "heartbeat",
        "night_summary",
    ]


# --- Degraded paths ----------------------------------------------------------

def test_degraded_unavailable_lines_ignored():
    r = parse(DEGRADED_PATH)
    assert r["containers"] == {}
    assert r["iddsi_train_running"] is False
    assert r["deepseek_v4_running"] is False
    assert r["models"] == {}
    assert r["flux"]["safetensors_found"] is False
    assert r["flux"]["temp_files"] is None
    assert r["gpu"]["total_mb"] is None
    assert r["gpu"]["host_ram"] is None
    assert len(r["gpu"]["compute_apps"]) == 1
    assert r["heartbeat_age_minutes"] is None
    assert r["heartbeat_log"] is None
    assert r["last_night_summary"] is None


def test_truncated_output_missing_sections():
    r = parse(TRUNCATED)
    assert r["iddsi_train_running"] is False  # Exited, not Up
    assert r["models"] == {}
    assert r["gpu"]["compute_apps"] == []
    assert r["heartbeat_age_minutes"] is None
    assert r["last_night_summary"] is None
    assert r["sections_seen"] == ["containers"]


def test_empty_input():
    r = parse("")
    assert r == {
        "containers": {},
        "iddsi_train_running": False,
        "deepseek_v4_running": False,
        "models": {},
        "flux": {"safetensors_found": False, "safetensors_bytes": None, "temp_files": None},
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


def test_result_is_json_serializable():
    import json

    json.dumps(parse(HAPPY_PATH))
    json.dumps(parse(DEGRADED_PATH))
    json.dumps(parse(TRUNCATED))
