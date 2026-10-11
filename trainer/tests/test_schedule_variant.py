"""Equivalence and shape tests for ``schedule_variant``.

``train/engine.py`` has no LR scheduler, so lane exp02_schedule added one in a
separate driver instead of editing the shared engine. These tests pin down the
two properties that make the lane's comparison meaningful:

1. with ``training.scheduler.type: constant`` the new driver reproduces the
   shipped engine's outputs exactly (same seed, same data, same artefacts), so
   any difference measured between the control and the variant arm is caused by
   the schedule and nothing else;
2. with ``type: cosine`` the applied learning rate really is a linear warmup
   over 10% of the phase's steps followed by a cosine decay to zero.

Both run on CPU with the tiny fixture model, so they need no GPU slot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fixtures.generate import generate_fixture
from schedule_variant import run_seed as scheduled_run_seed
from train.engine import run_seed as engine_run_seed


def _fixture_cfg(tmp_path: Path, name: str, scheduler: dict, epochs: int = 3) -> dict:
    manifest = generate_fixture(tmp_path / "fixture", per_level_per_split=2, size=24)
    return {
        "experiment": {"name": name, "output_dir": str(tmp_path / "runs"), "seeds": [5]},
        "data": {
            "manifest": str(manifest), "image_size": 24, "num_workers": 0, "batch_size": 4,
            "level_field": "level", "split_field": "split", "group_field": "group_id", "synthetic": True,
        },
        "model": {"kind": "tiny_cnn", "name": "test", "dropout": 0.0, "pretrained": False},
        "training": {
            "weight_decay": 0.0, "class_balance_beta": 0.99, "gradient_clip_norm": 1.0,
            "amp": False, "device": "cpu", "scheduler": scheduler,
            "phases": [
                {"name": "frozen_encoder_head", "epochs": epochs, "lr": 0.01, "unfreeze_last_blocks": 0},
                {"name": "unfreeze_last_two_blocks", "epochs": epochs, "lr": 0.001, "unfreeze_last_blocks": 2},
            ],
        },
        "calibration": {
            "target_under_fnr": 0.4, "max_abstention": 0.4,
            "temperature_min": 0.5, "temperature_max": 2.0, "temperature_steps": 5,
        },
        "evaluation": {"bootstrap_samples": 5, "bootstrap_seed": 2},
    }


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_constant_schedule_matches_shipped_engine(tmp_path: Path):
    """Same seed + constant schedule => byte-identical artefacts to the engine."""
    engine_cfg = _fixture_cfg(tmp_path / "engine", "engine_arm", {"type": "constant"})
    driver_cfg = _fixture_cfg(tmp_path / "driver", "driver_arm", {"type": "constant"})

    engine_summary = engine_run_seed(engine_cfg, 5)
    driver_summary = scheduled_run_seed(driver_cfg, 5)

    engine_dir = Path(engine_summary["output_dir"])
    driver_dir = Path(driver_summary["output_dir"])

    assert engine_summary["metrics"] == driver_summary["metrics"]
    assert engine_summary["best_epoch"] == driver_summary["best_epoch"]
    for artefact in ("history.json", "calibration.json", "metrics.json", "test_structured_outputs.json"):
        assert _load(engine_dir / artefact) == _load(driver_dir / artefact), artefact


def test_cosine_schedule_changes_the_run(tmp_path: Path):
    """The same fixture under cosine+warmup must not merely re-label the control."""
    control = _fixture_cfg(tmp_path / "control", "cos_control", {"type": "constant"})
    variant = _fixture_cfg(tmp_path / "variant", "cos_variant", {"type": "cosine", "warmup_fraction": 0.1})

    control_summary = scheduled_run_seed(control, 5)
    variant_summary = scheduled_run_seed(variant, 5)
    assert control_summary["metrics"] != variant_summary["metrics"]


def test_cosine_trace_is_warmup_then_decay(tmp_path: Path):
    """Applied LR must ramp linearly for 10% of the phase, then cosine to ~0."""
    epochs = 20
    cfg = _fixture_cfg(tmp_path, "trace", {"type": "cosine", "warmup_fraction": 0.1}, epochs=epochs)
    summary = scheduled_run_seed(cfg, 5)
    trace = _load(Path(summary["output_dir"]) / "schedule_trace.json")

    assert len(trace) == 2 * epochs
    phase_one = [row["lr_at_epoch_start"] for row in trace if row["phase"] == "frozen_encoder_head"]
    peak = 0.01
    total_steps = trace[0]["total_steps"]
    warmup_steps = trace[0]["warmup_steps"]
    assert total_steps == epochs * trace[0]["steps_per_epoch"]
    assert warmup_steps == max(1, round(0.1 * total_steps))

    # Linear warmup: each optimizer step adds the same increment, so the rate in
    # force at the start of consecutive epochs rises by steps_per_epoch / warmup.
    steps_per_epoch = trace[0]["steps_per_epoch"]
    expected_first = peak / warmup_steps
    assert phase_one[0] == pytest.approx(expected_first, rel=1e-9)
    warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
    ramp = phase_one[:warmup_epochs]
    increments = [b - a for a, b in zip(ramp, ramp[1:])]
    assert increments and all(abs(step - increments[0]) < 1e-12 for step in increments)
    assert increments[0] == pytest.approx(peak * steps_per_epoch / warmup_steps, rel=1e-9)

    # Cosine decay after warmup, ending at min_lr_fraction * peak = 0. The first
    # epoch that starts at or beyond the warmup horizon is where the peak sits.
    decay = phase_one[warmup_epochs:]
    assert decay and decay[0] == pytest.approx(peak, rel=1e-9)
    assert all(b <= a + 1e-15 for a, b in zip(decay, decay[1:])), "decay must be monotone non-increasing"
    assert max(decay) <= peak + 1e-12
    assert trace[-1]["lr_at_epoch_end"] == pytest.approx(0.0, abs=1e-9)

    # The cosine midpoint sits at about half the peak, as cos(pi/2) = 0.
    mid = decay[len(decay) // 2]
    assert mid == pytest.approx(peak * 0.5, rel=0.35)

    # Phase two is re-warmed from its own (much smaller) peak LR.
    phase_two = [row["lr_at_epoch_start"] for row in trace if row["phase"] == "unfreeze_last_two_blocks"]
    assert phase_two[0] == pytest.approx(0.001 / warmup_steps, rel=1e-9)
    assert phase_two[-1] < phase_two[0]
    assert all(math.isfinite(value) for value in phase_one + phase_two)


def test_unknown_scheduler_type_is_rejected(tmp_path: Path):
    cfg = _fixture_cfg(tmp_path, "bad", {"type": "triangular"}, epochs=1)
    with pytest.raises(ValueError):
        scheduled_run_seed(cfg, 5)
